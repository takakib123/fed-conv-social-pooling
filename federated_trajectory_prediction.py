# Federated trajectory prediction — standalone Python script.
#
# Converted from federated_trajectory_prediction_with_eval.ipynb
# with local NGSIM data paths and four additional CLI options:
#
#   --checkpoint      warm-start from a full highwayNet checkpoint
#   --fe-checkpoint   load TAPPFL FE-only weights (auto-freezes encoder)
#   --freeze-encoder  freeze encoder even when using --checkpoint
#   --eval-only       skip training; evaluate checkpoint on all 4 test splits
#
# Example commands:
#   # Train from scratch
#   python federated_trajectory_prediction.py
#
#   # Warm-start from existing full FL checkpoint
#   python federated_trajectory_prediction.py --checkpoint pretrained_models/fl_global_round_8.tar
#
#   # Load TAPPFL FE-only (auto-freezes FE, trains decoder only)
#   python federated_trajectory_prediction.py --fe-checkpoint pretrained_models\tappfl\tappfl_fe_round_5.tar
#
#   # Load full checkpoint, freeze encoder, train decoder only
#   python federated_trajectory_prediction.py --checkpoint pretrained_models/fl_global_round_8.tar --freeze-encoder
#
#   # Evaluate checkpoint on all 4 NGSIM test splits (no training)
#   python federated_trajectory_prediction.py --eval-only --checkpoint trained_models/fl_round_8.tar

# python federated_trajectory_prediction.py \
#   --fe-checkpoint pretrained_models/tappfl/tappfl_fe_best.tar \
#   --rounds 8 \
#   --pretrain-rounds 3 \
#   --local-epochs 2 \
#   --checkpoint-dir trained_models/fe_finetune \
#   --no-wandb


from __future__ import print_function, division

import argparse
import copy
import gc
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest, maskedMSETest

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Model architecture ────────────────────────────────────────────────────────

ARGS = {
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
    'train_flag':           True,
}

# ── FL hyperparameters (defaults; overridden by CLI args) ─────────────────────

NUM_CLIENTS    = 10
GLOBAL_ROUNDS  = 5
LOCAL_EPOCHS   = 2
PRETRAIN_ROUNDS = 3   # first N rounds use MSE loss, then switch to NLL
BATCH_SIZE     = 8192
LOG_INTERVAL   = 10   # log every N batches

# ── Data paths ────────────────────────────────────────────────────────────────

TRAIN_DATA = 'NGSIM/data/TrainSet.mat'
VAL_DATA   = 'NGSIM/data/ValSet.mat'
TEST_DIR   = 'NGSIM/data'

TEST_SPLITS = {
    'Keep':  'TestSet_Keep.mat',
    'Left':  'TestSet_Left.mat',
    'Right': 'TestSet_Right.mat',
    'Merge': 'TestSet_Merge.mat',
}

# ── FE module names (used for freezing) ──────────────────────────────────────
# These are the parameter-name prefixes that belong to the Feature Extractor.
# All other layers (dec_lstm, op, op_lat, op_lon) are the decoder.

_FE_MODULES = {'ip_emb', 'enc_lstm', 'dyn_emb', 'soc_conv', 'conv_3x1'}

# WandB is toggled on/off by --no-wandb; set at runtime in main()
_WANDB = False


# =============================================================================
# Checkpoint utilities
# =============================================================================

def load_full_checkpoint(path: str, model: highwayNet, device) -> None:
    """Load a full highwayNet state dict from path into model (in-place)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in model.parameters())
    logger.info('Loaded full checkpoint: %s  |  params: %d', path, n)


def load_fe_checkpoint(path: str, model: highwayNet, device) -> None:
    """
    Inject FE-only weights (e.g. from TAPPFL tappfl_fe_best.tar) into a full
    highwayNet model.  Only keys whose top-level prefix is in _FE_MODULES are
    updated; decoder keys keep their current (random) values.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'fe' in state:
        state = state['fe']

    current = model.state_dict()
    fe_keys_loaded = 0
    for k, v in state.items():
        prefix = k.split('.')[0]
        if prefix in _FE_MODULES:
            current[k] = v
            fe_keys_loaded += 1

    model.load_state_dict(current, strict=True)
    logger.info('Loaded FE-only checkpoint: %s  |  FE keys injected: %d', path, fe_keys_loaded)


def freeze_encoder(model: highwayNet) -> None:
    """Freeze all FE parameters in model (requires_grad = False)."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.split('.')[0] in _FE_MODULES:
            param.requires_grad_(False)
            frozen += param.numel()
    logger.debug('Encoder frozen: %d parameters with requires_grad=False', frozen)


def unfreeze_all(model: highwayNet) -> None:
    """Re-enable gradients for all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)


# =============================================================================
# Federated client
# =============================================================================

class FLClient:
    """
    One federated learning client holding a local copy of highwayNet and a
    partition of the NGSIM training data.

    If freeze_enc=True the encoder (FE) weights are frozen before every local
    training round, so only the decoder parameters receive gradient updates.
    """

    def __init__(self, client_id: int, dataset, device, args: dict,
                 freeze_enc: bool = False):
        self.client_id  = client_id
        self.device     = device
        self.args       = args
        self.freeze_enc = freeze_enc
        self.net        = highwayNet(args).to(device)
        self.crossEnt   = nn.BCELoss()

        self.dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.dataset.collate_fn,
        )

    def train(self, global_weights: dict, round_num: int,
              global_step: int, pretrain_rounds: int,
              local_epochs: int) -> tuple:
        """
        Run local_epochs of gradient descent using global_weights as the
        starting point.

        Returns:
            (state_dict, avg_loss, steps_taken)
        """
        # 1. Load latest global weights
        self.net.load_state_dict(global_weights)

        # 2. Freeze encoder if requested (must be re-applied after load_state_dict)
        if self.freeze_enc:
            freeze_encoder(self.net)

        # 3. Build optimizer over trainable parameters only
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable)

        # 4. Determine loss mode for this round
        use_mse = round_num < pretrain_rounds
        self.net.train()
        self.net.train_flag = True

        total_loss  = 0.0
        batch_count = 0

        for epoch in range(local_epochs):
            for i, data in enumerate(self.dataloader):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

                hist    = hist.to(self.device)
                nbrs    = nbrs.to(self.device)
                mask    = mask.to(self.device)
                lat_enc = lat_enc.to(self.device)
                lon_enc = lon_enc.to(self.device)
                fut     = fut.to(self.device)
                op_mask = op_mask.to(self.device)

                # Forward pass
                if self.args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = self.net(
                        hist, nbrs, mask, lat_enc, lon_enc
                    )
                else:
                    fut_pred = self.net(hist, nbrs, mask, lat_enc, lon_enc)

                # Loss
                if use_mse:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)
                    if self.args['use_maneuvers']:
                        l += self.crossEnt(lat_pred, lat_enc)
                        l += self.crossEnt(lon_pred, lon_enc)

                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
                optimizer.step()

                total_loss  += l.item()
                batch_count += 1
                current_step = global_step + batch_count

                if _WANDB and (batch_count % LOG_INTERVAL == 0):
                    import wandb
                    wandb.log({
                        f'client_{self.client_id}/train_loss': l.item(),
                        'round':       round_num + 1,
                        'client_id':   self.client_id,
                        'global_step': current_step,
                    })

        gc.collect()
        if self.args['use_cuda']:
            torch.cuda.empty_cache()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return self.net.state_dict(), avg_loss, batch_count


# =============================================================================
# FedAvg
# =============================================================================

def fed_avg(weights_list: list) -> dict:
    """Standard FedAvg: element-wise mean of all client state dicts."""
    w_avg = copy.deepcopy(weights_list[0])
    for k in w_avg.keys():
        for i in range(1, len(weights_list)):
            w_avg[k] = w_avg[k] + weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights_list))
    return w_avg


# =============================================================================
# Validation
# =============================================================================

def validate_global_model(model: highwayNet, val_loader: DataLoader,
                           round_num: int, pretrain_rounds: int,
                           device, args: dict) -> float:
    """
    Validate the global model on the validation set.

    Loss mode:
      round < pretrain_rounds  →  MSE (model in train_flag=True mode)
      round >= pretrain_rounds →  NLL test (model in train_flag=False mode)
                                  NOTE: maskedNLLTest requires CUDA.
    """
    model.eval()
    use_mse = round_num < pretrain_rounds
    avg_loss = 0.0
    count    = 0

    with torch.no_grad():
        for data in val_loader:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
            hist    = hist.to(device)
            nbrs    = nbrs.to(device)
            mask    = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut     = fut.to(device)
            op_mask = op_mask.to(device)

            if use_mse:
                # Use teacher-forced single-mode forward for MSE
                model.train_flag = True
                if args['use_maneuvers']:
                    fut_pred, _, _ = model(hist, nbrs, mask, lat_enc, lon_enc)
                else:
                    fut_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
                model.train_flag = False
            else:
                # NLL test mode: model returns 6-mode predictions
                model.train_flag = False
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = model(
                        hist, nbrs, mask, lat_enc, lon_enc
                    )
                    l = maskedNLLTest(
                        fut_pred, lat_pred, lon_pred, fut, op_mask,
                        avg_along_time=True,
                    )
                else:
                    fut_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedNLLTest(
                        fut_pred, 0, 0, fut, op_mask,
                        use_maneuvers=False, avg_along_time=True,
                    )

            avg_loss += l.item()
            count    += 1

    return avg_loss / count if count > 0 else 0.0


# =============================================================================
# Evaluation (RMSE / ADE / FDE)
# =============================================================================

def evaluate_model(model: highwayNet, test_loader: DataLoader,
                   device, args: dict) -> dict:
    """
    Evaluate model on a test split.

    Returns dict with keys 'rmse', 'ade', 'fde' — each a CPU tensor of length
    out_length (25 timesteps, 0.2 s each → 0.2 s … 5.0 s).
    Units: metres.
    """
    model.eval()
    model.train_flag = False

    out_len   = args['out_length']
    lossVals  = torch.zeros(out_len).to(device)
    counts    = torch.zeros(out_len).to(device)
    fdeVals   = torch.zeros(out_len).to(device)
    fdeCounts = torch.zeros(out_len).to(device)

    with torch.no_grad():
        for data in test_loader:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
            hist    = hist.to(device)
            nbrs    = nbrs.to(device)
            mask    = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut     = fut.to(device)
            op_mask = op_mask.to(device)

            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = model(
                    hist, nbrs, mask, lat_enc, lon_enc
                )
                # Argmax maneuver selection (best single mode per sample)
                fut_pred_max = torch.zeros_like(fut_pred[0])
                for k in range(lat_pred.shape[0]):
                    lat_man = torch.argmax(lat_pred[k, :]).detach()
                    lon_man = torch.argmax(lon_pred[k, :]).detach()
                    indx    = lon_man * args['num_lat_classes'] + lat_man
                    fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
            else:
                fut_pred_max = model(hist, nbrs, mask, lat_enc, lon_enc)

            # ── RMSE ─────────────────────────────────────────────────────────
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
            lossVals += l.detach()
            counts   += c.detach()

            # ── FDE (L2 displacement at each timestep) ────────────────────────
            pred_pos = fut_pred_max[:, :, 0:2]          # (T, B, 2)
            diff     = pred_pos - fut                    # (T, B, 2)
            dist     = torch.norm(diff, dim=2)           # (T, B)
            valid    = op_mask[:, :, 0]                  # (T, B)
            fdeVals   += (dist * valid).sum(dim=1).detach()
            fdeCounts += valid.sum(dim=1).detach()

    rmse = torch.pow(lossVals / counts.clamp(min=1), 0.5) * 0.3048
    fde  = (fdeVals / fdeCounts.clamp(min=1)) * 0.3048
    ade  = (torch.cumsum(fdeVals, dim=0) /
            torch.cumsum(fdeCounts, dim=0).clamp(min=1)) * 0.3048

    return {'rmse': rmse.cpu(), 'ade': ade.cpu(), 'fde': fde.cpu()}


def print_eval_table(metrics: dict, split_name: str) -> None:
    """
    Print RMSE / ADE / FDE at 1 s, 2 s, 3 s, 4 s, 5 s.
    Timesteps are at 0.2 s intervals (10 Hz data, downsampled ×2).
    Indices for 1-5 s: 4, 9, 14, 19, 24.
    """
    idx   = [4, 9, 14, 19, 24]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]

    header = '  '.join(f'{t:.1f}s' for t in times)
    sep    = '  '.join(['─────'] * 5)

    print(f'\n{"="*58}')
    print(f'  Evaluation — {split_name}')
    print(f'{"="*58}')
    print(f'  {"Metric":<8}  {header}')
    print(f'  {"─"*8}  {sep}')
    for key, label in [('rmse', 'RMSE'), ('ade', 'ADE'), ('fde', 'FDE')]:
        vals = metrics[key]
        row  = '  '.join(f'{vals[i]:>5.3f}' for i in idx)
        print(f'  {label:<8}  {row}')
    print()


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Federated trajectory prediction (highwayNet / NGSIM)'
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument('--train-data',   default=TRAIN_DATA,
                   help='Path to TrainSet.mat')
    p.add_argument('--val-data',     default=VAL_DATA,
                   help='Path to ValSet.mat')
    p.add_argument('--test-dir',     default=TEST_DIR,
                   help='Directory containing TestSet_Keep/Left/Right/Merge.mat')

    # ── Checkpoint options ────────────────────────────────────────────────────
    p.add_argument('--checkpoint',    default=None,
                   help='Full highwayNet checkpoint (.tar) to warm-start training')
    p.add_argument('--fe-checkpoint', default=None,
                   help='TAPPFL FE-only checkpoint (.tar). Injects FE weights and '
                        'automatically freezes the encoder — only the decoder trains.')
    p.add_argument('--freeze-encoder', action='store_true',
                   help='Freeze encoder even when using --checkpoint '
                        '(decoder-only fine-tuning)')
    p.add_argument('--eval-only',      action='store_true',
                   help='Skip training; load --checkpoint and evaluate on all 4 '
                        'NGSIM test splits (RMSE / ADE / FDE table)')
    p.add_argument('--checkpoint-dir', default='trained_models',
                   help='Directory to save per-round checkpoints')

    # ── FL hyperparameters ────────────────────────────────────────────────────
    p.add_argument('--rounds',          type=int,   default=GLOBAL_ROUNDS)
    p.add_argument('--pretrain-rounds', type=int,   default=PRETRAIN_ROUNDS,
                   help='First N rounds use MSE loss before switching to NLL')
    p.add_argument('--local-epochs',    type=int,   default=LOCAL_EPOCHS)
    p.add_argument('--batch-size',      type=int,   default=BATCH_SIZE)
    p.add_argument('--num-clients',     type=int,   default=NUM_CLIENTS)
    p.add_argument('--val-subset-ratio', type=float, default=1.0,
                   help='Fraction of ValSet to use (0.0–1.0)')

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument('--no-wandb', action='store_true',
                   help='Disable Weights & Biases logging')
    p.add_argument('--seed',     type=int, default=42)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global _WANDB, BATCH_SIZE

    args   = parse_args()
    device = torch.device('cuda' if ARGS['use_cuda'] else 'cpu')
    BATCH_SIZE = args.batch_size

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info('Device: %s', device)

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError('--eval-only requires --checkpoint <path>')

        eval_args = {**ARGS, 'train_flag': False}
        model = highwayNet(eval_args).to(device)
        load_full_checkpoint(args.checkpoint, model, device)
        model.eval()

        logger.info('Evaluating checkpoint: %s', args.checkpoint)
        for split_name, filename in TEST_SPLITS.items():
            test_path = os.path.join(args.test_dir, filename)
            if not os.path.exists(test_path):
                logger.warning('Test file not found, skipping: %s', test_path)
                continue
            ts_set    = ngsimDataset(test_path)
            ts_loader = DataLoader(
                ts_set, batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=ts_set.collate_fn,
            )
            logger.info('Evaluating split: %s  (%d samples)', split_name, len(ts_set))
            metrics = evaluate_model(model, ts_loader, device, eval_args)
            print_eval_table(metrics, split_name)
        return

    # ── WandB setup ───────────────────────────────────────────────────────────
    _WANDB = not args.no_wandb
    if _WANDB:
        try:
            import wandb
            wandb.init(
                project='conv-social-pooling-fl',
                reinit=True,
                config={
                    'num_clients':     args.num_clients,
                    'global_rounds':   args.rounds,
                    'pretrain_rounds': args.pretrain_rounds,
                    'local_epochs':    args.local_epochs,
                    'batch_size':      args.batch_size,
                    'freeze_encoder':  args.freeze_encoder or (args.fe_checkpoint is not None),
                    'checkpoint':      args.checkpoint,
                    'fe_checkpoint':   args.fe_checkpoint,
                },
            )
        except ImportError:
            logger.warning('wandb not installed — running without WandB logging.')
            _WANDB = False

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info('Loading training data: %s', args.train_data)
    train_dataset = ngsimDataset(args.train_data)

    logger.info('Loading validation data: %s', args.val_data)
    val_dataset_full = ngsimDataset(args.val_data)
    val_len = len(val_dataset_full)
    short_val_len = max(1, int(val_len * args.val_subset_ratio))
    val_subset = Subset(val_dataset_full, list(range(short_val_len)))
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=val_dataset_full.collate_fn,
    )
    logger.info('Val samples: %d / %d', short_val_len, val_len)

    # ── IID data partition across clients ─────────────────────────────────────
    total_samples = len(train_dataset)
    indices       = list(range(total_samples))
    split_size    = total_samples // args.num_clients

    # ── Global model init ─────────────────────────────────────────────────────
    global_model = highwayNet(ARGS).to(device)

    freeze_enc = False   # determined by checkpoint options below

    if args.fe_checkpoint:
        # Inject FE weights into full model, auto-freeze encoder
        load_fe_checkpoint(args.fe_checkpoint, global_model, device)
        freeze_enc = True
        logger.info('FE-only checkpoint loaded — encoder will be frozen, decoder trains only.')
    elif args.checkpoint:
        load_full_checkpoint(args.checkpoint, global_model, device)
        freeze_enc = args.freeze_encoder
        if freeze_enc:
            logger.info('Full checkpoint loaded — encoder frozen, decoder trains only.')
        else:
            logger.info('Full checkpoint loaded — full model will be fine-tuned.')
    else:
        logger.info('No checkpoint provided — training from random initialisation.')
        if args.freeze_encoder:
            logger.warning(
                '--freeze-encoder set but no checkpoint provided — '
                'encoder will be frozen from random init, which is unlikely to be useful.'
            )
            freeze_enc = True

    # ── Initialise clients ────────────────────────────────────────────────────
    clients = []
    for i in range(args.num_clients):
        idx_start  = i * split_size
        idx_end    = (i + 1) * split_size if i < args.num_clients - 1 else total_samples
        client_sub = Subset(train_dataset, indices[idx_start:idx_end])
        clients.append(FLClient(i, client_sub, device, ARGS, freeze_enc))
    logger.info(
        'Clients: %d  |  samples/client: ~%d  |  freeze_enc: %s',
        args.num_clients, split_size, freeze_enc,
    )

    # ── Federated training ────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_weights      = global_model.state_dict()
    global_step_tracker = 0
    best_val_loss       = float('inf')
    best_round          = 0

    logger.info(
        'Starting FL training: %d rounds | pretrain=%d | local_epochs=%d | '
        'batch=%d | clients=%d',
        args.rounds, args.pretrain_rounds, args.local_epochs,
        args.batch_size, args.num_clients,
    )

    for round_num in range(args.rounds):
        t0 = time.time()
        local_weights_list = []
        local_losses       = []
        max_steps          = 0

        loss_mode = 'MSE' if round_num < args.pretrain_rounds else 'NLL'
        logger.info('Round %d/%d  [%s]', round_num + 1, args.rounds, loss_mode)

        for client in clients:
            w, loss, steps = client.train(
                global_weights, round_num, global_step_tracker,
                args.pretrain_rounds, args.local_epochs,
            )
            local_weights_list.append(w)
            local_losses.append(loss)
            max_steps = max(max_steps, steps)
            logger.info(
                '  Client %d  avg_loss=%.4f  steps=%d',
                client.client_id, loss, steps,
            )

        global_step_tracker += max_steps

        # FedAvg
        global_weights = fed_avg(local_weights_list)
        global_model.load_state_dict(global_weights)

        # Validation
        val_loss = validate_global_model(
            global_model, val_loader, round_num,
            args.pretrain_rounds, device, ARGS,
        )
        avg_train_loss = sum(local_losses) / len(local_losses)
        elapsed        = time.time() - t0

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_round    = round_num + 1
            marker        = '  *best*'
            torch.save(
                global_weights,
                os.path.join(args.checkpoint_dir, 'fl_best.tar'),
            )

        logger.info(
            'Round %d | train=%.4f | val=%.4f | %.1fs%s',
            round_num + 1, avg_train_loss, val_loss, elapsed, marker,
        )

        if _WANDB:
            import wandb
            wandb.log({
                'global/val_loss':        val_loss,
                'global/avg_train_loss':  avg_train_loss,
                'round':                  round_num + 1,
                'global_step':            global_step_tracker,
            })

        # Per-round checkpoint
        torch.save(
            global_weights,
            os.path.join(args.checkpoint_dir, f'fl_round_{round_num + 1}.tar'),
        )

    logger.info(
        'Training complete.  Best val loss: %.4f at round %d.',
        best_val_loss, best_round,
    )

    if _WANDB:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
