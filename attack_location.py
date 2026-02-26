import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import scipy.io as scp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from model import highwayNet
from utils import ngsimDataset

NUM_DS_IDS = 6
DS_LABELS  = [
    'US-101 (07:50)', 'US-101 (08:05)', 'US-101 (08:20)',
    'I-80 (16:00)',   'I-80 (17:00)',   'I-80 (17:15)',
]

MODEL_ARGS = {
    'use_cuda':             torch.cuda.is_available(),
    'encoder_size':         64,
    'decoder_size':         128,
    'in_length':            16,
    'out_length':           25,
    'grid_size':            (13, 3),
    'soc_conv_depth':       64,
    'conv_3x1_depth':       16,
    'dyn_embedding_size':   32,
    'input_embedding_size': 32,
    'num_lat_classes':      3,
    'num_lon_classes':      2,
    'use_maneuvers':        True,
    'train_flag':           False,
}

_soc_emb = (((MODEL_ARGS['grid_size'][0] - 4) + 1) // 2) * MODEL_ARGS['conv_3x1_depth']
REPR_DIM  = _soc_emb + MODEL_ARGS['dyn_embedding_size']

ATTACK_BATCH_SIZE   = 256
ATTACK_LR           = 1e-2
ATTACK_EPOCHS       = 30
ATTACK_WEIGHT_DECAY = 1e-4
EXTRACT_BATCH_SIZE  = 512


# ── Dataset ───────────────────────────────────────────────────────────────────

class ngsimDatasetWithDsId(ngsimDataset):
    """
    Extends ngsimDataset to return dsId (0-indexed, 0–5) as an
    additional output — the private location attribute.
    """

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0]) - 1
        hist, fut, neighbors, lat_enc, lon_enc = super().__getitem__(idx)
        return hist, fut, neighbors, lat_enc, lon_enc, ds_id

    def collate_fn(self, samples):
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _ in samples:
            nbr_batch_size += sum(len(nbrs[i]) != 0 for i in range(len(nbrs)))

        maxlen = self.t_h // self.d_s + 1

        nbrs_batch    = torch.zeros(maxlen, nbr_batch_size, 2)
        mask_batch    = torch.zeros(
            len(samples), self.grid_size[1], self.grid_size[0], self.enc_size
        ).byte()
        hist_batch    = torch.zeros(maxlen, len(samples), 2)
        fut_batch     = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        loc_batch     = torch.zeros(len(samples), dtype=torch.long)

        count = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, ds_id) in enumerate(samples):
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])

            fut_batch[0:len(fut), sampleId, 0]     = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1]     = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1

            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            loc_batch[sampleId] = ds_id

            for nbr_id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos = nbr_id % self.grid_size[0]
                    mask_batch[
                        sampleId, nbr_id // self.grid_size[0], pos, :
                    ] = torch.ones(self.enc_size).byte()
                    count += 1

        return (
            hist_batch, nbrs_batch, mask_batch,
            lat_enc_batch, lon_enc_batch,
            fut_batch, op_mask_batch,
            loc_batch,
        )


# ── Encoder ───────────────────────────────────────────────────────────────────

class highwayNetEncoder(highwayNet):
    """
    Thin wrapper around highwayNet that exposes the encoder representation.
    """

    @torch.no_grad()
    def get_representation(self, hist, nbrs, masks):
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(
            self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2]))
        )

        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        masks    = masks.bool()
        soc_enc  = torch.zeros_like(masks).float()
        soc_enc  = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc  = soc_enc.permute(0, 3, 2, 1)
        soc_enc  = self.soc_maxpool(
            self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc))))
        )
        soc_enc  = soc_enc.view(-1, self.soc_embedding_size)

        return torch.cat((soc_enc, hist_enc), dim=1)


# ── Attack model ──────────────────────────────────────────────────────────────

class LocationAttackMLP(nn.Module):
    """Lightweight MLP attack model: z → dsId (2 hidden layers, no dropout)."""

    def __init__(self, repr_dim: int = REPR_DIM, num_classes: int = NUM_DS_IDS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64),       nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Core functions ────────────────────────────────────────────────────────────

def load_fl_model(checkpoint_path: str, args: dict, device) -> highwayNetEncoder:
    model = highwayNetEncoder(args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'FL model loaded: {checkpoint_path}  |  params: {n_params:,}  |  repr_dim: {REPR_DIM}')
    return model


def extract_representations(encoder, dataset: ngsimDatasetWithDsId, indices: list,
                             device, batch_size: int = EXTRACT_BATCH_SIZE,
                             split_name: str = '') -> TensorDataset:
    """Extract (z, dsId) pairs for the given list of dataset indices."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=dataset.collate_fn,
    )

    label_counts = np.zeros(NUM_DS_IDS, dtype=int)
    for idx in indices:
        label_counts[int(dataset.D[idx, 0]) - 1] += 1

    tag = f' [{split_name}]' if split_name else ''
    print(f'  {len(indices)} samples{tag}')
    for i, cnt in enumerate(label_counts):
        print(f'    dsId={i} ({DS_LABELS[i]}): {cnt} ({cnt/len(indices)*100:.1f}%)')

    z_list, label_list = [], []
    encoder.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, loc = batch
            z = encoder.get_representation(hist.to(device), nbrs.to(device), mask.to(device))
            z_list.append(z.cpu())
            label_list.append(loc.cpu())
            if (batch_idx + 1) % 50 == 0:
                print(f'    ... {(batch_idx+1)*batch_size}/{len(indices)} extracted')

    z_all      = torch.cat(z_list,     dim=0)
    labels_all = torch.cat(label_list, dim=0)
    print(f'  Extraction complete: z shape={tuple(z_all.shape)}')
    return TensorDataset(z_all, labels_all)


def train_attack_model(
    train_ds:     TensorDataset,
    val_ds:       TensorDataset,
    device,
    epochs:       int   = ATTACK_EPOCHS,
    batch_size:   int   = ATTACK_BATCH_SIZE,
    lr:           float = ATTACK_LR,
    weight_decay: float = ATTACK_WEIGHT_DECAY,
) -> tuple:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = LocationAttackMLP(REPR_DIM, NUM_DS_IDS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.2,         # 20% of training spent warming up
        anneal_strategy='cos',
    )
    # Class-balanced loss to handle dsId imbalance
    label_counts = torch.bincount(train_ds.tensors[1], minlength=NUM_DS_IDS).float()
    class_weights = (label_counts.sum() / (NUM_DS_IDS * label_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f'Training attack model for {epochs} epochs | '
          f'train={len(train_ds)} val={len(val_ds)} | batch={batch_size} | lr={lr}')
    print(f'  {"Epoch":>5}  {"Train Loss":>10}  {"Train Acc":>9}  {"Val Loss":>8}  {"Val Acc":>7}')
    print(f'  {"─"*5}  {"─"*10}  {"─"*9}  {"─"*8}  {"─"*7}')

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for z_batch, label_batch in train_loader:
            z_batch     = z_batch.to(device)
            label_batch = label_batch.to(device)
            logits      = model(z_batch)
            loss        = criterion(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            t_loss    += loss.item() * len(z_batch)
            t_correct += (logits.argmax(1) == label_batch).sum().item()
            t_total   += len(z_batch)
        epoch_train_loss = t_loss / t_total
        epoch_train_acc  = t_correct / t_total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for z_batch, label_batch in val_loader:
                z_batch     = z_batch.to(device)
                label_batch = label_batch.to(device)
                logits      = model(z_batch)
                loss        = criterion(logits, label_batch)
                v_loss    += loss.item() * len(z_batch)
                v_correct += (logits.argmax(1) == label_batch).sum().item()
                v_total   += len(z_batch)

        epoch_val_loss = v_loss / v_total
        epoch_val_acc  = v_correct / v_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f'  {epoch+1:5d}  {epoch_train_loss:10.4f}  {epoch_train_acc*100:8.2f}%'
              f'  {epoch_val_loss:8.4f}  {epoch_val_acc*100:6.2f}%')

    return model, history


def evaluate_attack_model(
    attack_model: LocationAttackMLP,
    test_ds:      TensorDataset,
    device,
    batch_size:   int = ATTACK_BATCH_SIZE,
) -> dict:
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    attack_model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for z_batch, label_batch in loader:
            logits = attack_model(z_batch.to(device))
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label_batch.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    accuracy   = (all_preds == all_labels).mean()

    per_class_acc = {}
    for c in range(NUM_DS_IDS):
        mask = all_labels == c
        per_class_acc[c] = (all_preds[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0.0

    return {
        'accuracy':      accuracy,
        'per_class_acc': per_class_acc,
        'all_preds':     all_preds,
        'all_labels':    all_labels,
        'all_probs':     all_probs,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_and_accuracy(train_history: dict, results: dict) -> None:
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(train_history['train_acc']) + 1)

    ax = axes[0]
    ax.plot(epochs_range, train_history['train_loss'],
            color='steelblue', linewidth=2, label='Train loss')
    ax.plot(epochs_range, train_history['val_loss'],
            color='orange', linewidth=2, linestyle='--', label='Val loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Attack Model Loss Curve')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, [a * 100 for a in train_history['train_acc']],
            color='steelblue', linewidth=2, label='Train accuracy')
    ax.plot(epochs_range, [a * 100 for a in train_history['val_acc']],
            color='orange', linewidth=2, linestyle='--', label='Val accuracy')
    ax.axhline(100 / NUM_DS_IDS, color='red', linestyle=':', linewidth=1.5,
               label=f'Random baseline ({100/NUM_DS_IDS:.1f}%)')
    ax.axhline(results['accuracy'] * 100, color='green', linestyle='--', linewidth=1.5,
               label=f'Test accuracy ({results["accuracy"]*100:.1f}%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Attack Model Accuracy Curve')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Attack Model Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attack_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: attack_accuracy.png')


def plot_confusion_matrix(results: dict) -> None:
    cm      = confusion_matrix(results['all_labels'], results['all_preds'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_labels = ['US101-1', 'US101-2', 'US101-3', 'I80-1', 'I80-2', 'I80-3']

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (counts)', 'Confusion Matrix (row-normalised)'],
        ['d', '.2f'],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=short_labels, yticklabels=short_labels,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Predicted dsId'); ax.set_ylabel('True dsId'); ax.set_title(title)

    plt.suptitle('Attack Model Confusion — Location Inference', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('attack_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: attack_confusion_matrix.png')

    short_labels = ['US101-1', 'US101-2', 'US101-3', 'I80-1', 'I80-2', 'I80-3']
    print('Classification Report:')
    print(classification_report(results['all_labels'], results['all_preds'],
                                target_names=short_labels, digits=4))




def print_summary(results: dict, fl_checkpoint: str) -> None:
    random_baseline = 1.0 / NUM_DS_IDS
    attack_acc      = results['accuracy']
    leakage_gap     = attack_acc - random_baseline

    if leakage_gap > 0.30:
        severity = 'HIGH';     comment = 'Strong location signal — TAPPFL protection recommended.'
    elif leakage_gap > 0.10:
        severity = 'MODERATE'; comment = 'Meaningful leakage present — privacy protection advisable.'
    else:
        severity = 'LOW';      comment = 'Leakage close to random — limited location information.'

    print('=' * 60)
    print('  PRIVACY LEAKAGE EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Model checkpoint : {fl_checkpoint}')
    print(f'  Representation   : z ∈ ℝ^{REPR_DIM}  (encoder output)')
    print(f'  Private attribute: dsId  (highway segment, 6 classes)')
    print(f'  Attack model     : LocationAttackMLP (4-layer deep MLP)')
    print()
    print(f'  Random baseline  : {random_baseline*100:.2f}%  (1/6)')
    print(f'  Attack accuracy  : {attack_acc*100:.2f}%')
    print(f'  Leakage gap      : +{leakage_gap*100:.2f}% above random')
    print()
    print(f'  Leakage severity : {severity}')
    print(f'  Assessment       : {comment}')
    print()
    print('  Per-segment breakdown:')
    for c in range(NUM_DS_IDS):
        acc  = results['per_class_acc'][c]
        gap  = acc - random_baseline
        flag = '⚠' if gap > 0.2 else ' '
        print(f'    {flag} dsId={c} {DS_LABELS[c]:<22}: {acc*100:.2f}%  (gap: +{gap*100:.2f}%)')
    print('=' * 60)
    print()
    print('  Next step: Run TAPPFL training and re-run this script on the')
    print('  TAPPFL-protected representations to quantify privacy improvement.')


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Privacy attack: location inference from FL representations')
    parser.add_argument('--checkpoint',  default='pretrained_models/cslstm_central_best.tar',
                        help='Path to the FL model checkpoint (.tar)')
    parser.add_argument('--train-data',   default='NGSIM/data/TrainSet.mat',
                        help='Path to TrainSet.mat')
    parser.add_argument('--epochs',       type=int,   default=ATTACK_EPOCHS)
    parser.add_argument('--batch-size',   type=int,   default=ATTACK_BATCH_SIZE)
    parser.add_argument('--lr',           type=float, default=ATTACK_LR)
    parser.add_argument('--subset-ratio', type=float, default=0.2,
                        help='Fraction of TrainSet to use in total (0.0–1.0)')
    parser.add_argument('--save-model',   default='attack_model.pt',
                        help='Path to save the trained attack model')
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device       : {device}')
    print(f'Repr dim (z) : {REPR_DIM}')

    # Load encoder
    encoder = load_fl_model(args.checkpoint, MODEL_ARGS, device)

    # Build 70 / 10 / 20 index split from TrainSet.mat
    dataset   = ngsimDatasetWithDsId(args.train_data)
    n_total   = int(len(dataset) * args.subset_ratio)
    all_idx   = np.random.permutation(len(dataset))[:n_total].tolist()
    n_test    = int(n_total * 0.20)
    n_val     = int(n_total * 0.10)
    test_idx  = all_idx[:n_test]
    val_idx   = all_idx[n_test:n_test + n_val]
    train_idx = all_idx[n_test + n_val:]
    print(f'\nTrainSet subset: {n_total} samples  '
          f'(train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)})')

    # Extract representations
    print('\nExtracting train representations...')
    train_repr_ds = extract_representations(encoder, dataset, train_idx, device,
                                            split_name='train')
    print('\nExtracting val representations...')
    val_repr_ds   = extract_representations(encoder, dataset, val_idx,   device,
                                            split_name='val')
    print('\nExtracting test representations...')
    test_repr_ds  = extract_representations(encoder, dataset, test_idx,  device,
                                            split_name='test')

    # Train attack model
    print()
    attack_model, train_history = train_attack_model(
        train_repr_ds, val_repr_ds, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # Evaluate
    print('\nEvaluating on test representations...')
    results = evaluate_attack_model(attack_model, test_repr_ds, device, args.batch_size)

    print(f'\n{"─"*50}')
    print(f'  Overall Attack Accuracy : {results["accuracy"]*100:.2f}%')
    print(f'  Random Baseline         : {100/NUM_DS_IDS:.2f}%')
    print(f'  Leakage Gap             : +{(results["accuracy"] - 1/NUM_DS_IDS)*100:.2f}%')
    print(f'{"─"*50}')
    print('\nPer-class accuracy:')
    for c, acc in results['per_class_acc'].items():
        bar = '█' * int(acc * 30)
        print(f'  dsId={c} {DS_LABELS[c]:<22} {acc*100:6.2f}%  {bar}')

    # Save attack model
    torch.save(attack_model.state_dict(), args.save_model)
    print(f'\nAttack model saved: {args.save_model}')

    # Plots
    plot_training_and_accuracy(train_history, results)
    plot_confusion_matrix(results)
    print_summary(results, args.checkpoint)


if __name__ == '__main__':
    main()
