from __future__ import division
import copy
import gc
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Laplace
from torch.utils.data import DataLoader, Subset

from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest, maskedMSETest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [FL-DP] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fl_dp_experiment.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Model & architecture configuration  (mirrors the notebook)
# ─────────────────────────────────────────────────────────────────────────────

ARGS = {
    "use_cuda":             True and torch.cuda.is_available(),
    "encoder_size":         64,
    "decoder_size":         128,
    "in_length":            16,
    "out_length":           25,
    "grid_size":            (13, 3),
    "soc_conv_depth":       64,
    "conv_3x1_depth":       16,
    "dyn_embedding_size":   32,
    "input_embedding_size": 32,
    "num_lat_classes":      3,
    "num_lon_classes":      2,
    "use_maneuvers":        True,
    "train_flag":           True,
}


# ─────────────────────────────────────────────────────────────────────────────
#  FL hyperparameters  (From ./federated_trajectory_prediction_with_eval.py)
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLIENTS     = 10
GLOBAL_ROUNDS   = 8
LOCAL_EPOCHS    = 2
PRETRAIN_ROUNDS = 5          # first N rounds use MSE; remaining use NLL
BATCH_SIZE      = 8192
LOG_INTERVAL    = 10
REUSE_WEIGHTS   = False      # set True + fill CHECKPOINT_PATH to warm-start
CHECKPOINT_PATH = ""         # e.g. "pretrained_models/fl_global_round_8.tar"
DATA_DIR        = "data"     # directory that contains TrainSet.mat / ValSet.mat / TestSet_Keep.mat

DEVICE = torch.device("cuda" if ARGS["use_cuda"] else "cpu")

torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
#  Differential Privacy configuration
#  ── All DP knobs live here; change these to run experiments ──────────────────
# ─────────────────────────────────────────────────────────────────────────────

DP_CONFIG = {
    # Toggle DP on / off without touching any other setting
    "enabled":       True,

    # Noise distribution:  'gaussian' | 'laplace' | 'none'
    # 'gaussian'  → (epsilon, delta)-DP  via the Gaussian mechanism
    # 'laplace'   → pure epsilon-DP      via the Laplace mechanism
    # 'none'      → no noise (equivalent to enabled=False)
    "noise_type":    "gaussian",

    # Privacy budget.  Smaller = more private = more noise.
    # Common research values: 0.1 (very strong) … 10 (mild).
    "epsilon":       1.0,

    # DP relaxation.  Used only for the Gaussian mechanism.
    # Rule-of-thumb: 1 / dataset_size, or a small constant like 1e-5.
    "delta":         1e-5,

    # L2 gradient clipping threshold (= L2 sensitivity of the gradient).
    # Noise scale is proportional to this value.
    # The notebook used 10.0 for pure FL; DP-SGD typically uses 1.0–5.0.
    "max_grad_norm": 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DP noise injectors  (adapted from UDP-FL  gradients/noise.py)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_sigma(epsilon: float, delta: float, max_grad_norm: float) -> float:
    """
    Calibrate Gaussian noise std for (epsilon, delta)-DP.
    Formula: sigma = sqrt(2 * ln(1.25 / delta)) * max_grad_norm / epsilon
    Reference: Dwork & Roth (2014), Theorem A.1.
    """
    return math.sqrt(2.0 * math.log(1.25 / delta)) * max_grad_norm / epsilon


class GaussianNoiseInjector:
    """
    Calibrated Gaussian noise for (epsilon, delta)-DP.
    Adapted from UDP-FL GaussianNoiseGenerator but with analytically derived sigma.
    """

    def __init__(self, epsilon: float, delta: float, max_grad_norm: float):
        self.epsilon       = epsilon
        self.delta         = delta
        self.max_grad_norm = max_grad_norm
        self.sigma         = _gaussian_sigma(epsilon, delta, max_grad_norm)
        logger.info(
            "[DP] Gaussian mechanism  |  epsilon=%.4f  delta=%.2e  "
            "max_grad_norm=%.4f  → sigma=%.4f",
            epsilon, delta, max_grad_norm, self.sigma,
        )

    def get_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return a noise tensor matching gradient shape, sampled from N(0, sigma^2)."""
        return torch.empty_like(gradient).normal_(0.0, self.sigma)


class LaplaceNoiseInjector:
    """
    Calibrated Laplace noise for pure epsilon-DP.
    scale = max_grad_norm / epsilon
    Adapted from UDP-FL LaplaceNoiseGenerator.
    """

    def __init__(self, epsilon: float, max_grad_norm: float):
        self.epsilon       = epsilon
        self.max_grad_norm = max_grad_norm
        self.scale         = max_grad_norm / epsilon
        logger.info(
            "[DP] Laplace mechanism  |  epsilon=%.4f  max_grad_norm=%.4f  → scale=%.4f",
            epsilon, max_grad_norm, self.scale,
        )

    def get_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        return Laplace(0.0, self.scale).sample(gradient.size())


def build_noise_injector(dp_cfg: dict):
    """
    Factory function – returns the appropriate noise injector or None.
    Mirrors the pattern in UDP-FL where BaseClient accepts a noise_generator.
    """
    if not dp_cfg["enabled"] or dp_cfg["noise_type"] == "none":
        return None

    noise_type = dp_cfg["noise_type"].lower()
    if noise_type == "gaussian":
        return GaussianNoiseInjector(
            epsilon=dp_cfg["epsilon"],
            delta=dp_cfg["delta"],
            max_grad_norm=dp_cfg["max_grad_norm"],
        )
    if noise_type == "laplace":
        return LaplaceNoiseInjector(
            epsilon=dp_cfg["epsilon"],
            max_grad_norm=dp_cfg["max_grad_norm"],
        )
    raise ValueError(
        f"Unknown noise_type '{noise_type}'. Choose 'gaussian', 'laplace', or 'none'."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FL Client with DP-SGD
# ─────────────────────────────────────────────────────────────────────────────

class DPFLClient:
    """
    Federated Learning client with DP-SGD gradient perturbation.

    Per-batch DP-SGD procedure (mirrors UDP-FL BaseModel.train_dpsgd):
      1. Forward pass + compute loss
      2. Backprop → raw gradients
      3. Clip each gradient tensor by L2 norm  (bounds sensitivity)
      4. Add noise from noise_injector          (privatises gradients)
      5. Optimizer step

    When noise_injector is None, steps 3–4 degrade to standard gradient
    clipping (same as the original notebook) for training stability.
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        device: torch.device,
        args: dict,
        noise_injector=None,
        max_grad_norm: float = 1.0,
    ):
        self.client_id      = client_id
        self.dataset        = dataset
        self.device         = device
        self.args           = args
        self.noise_injector = noise_injector
        self.max_grad_norm  = max_grad_norm

        self.net        = highwayNet(args).to(device)
        self.optimizer  = torch.optim.Adam(self.net.parameters())
        self.cross_ent  = nn.BCELoss()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.dataset.collate_fn,
        )

    # ── DP-SGD gradient step ───────────────────────────────────────────────

    def _dp_gradient_step(self):
        """
        Clip + optionally add noise to all parameter gradients.
        This is the core of DP-SGD (Algorithm 1, Abadi et al. 2016).
        Directly adapted from UDP-FL  models/base_model.py:train_dpsgd.
        """
        # Step 1: Clip gradients (bounds L2 sensitivity = max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        # Step 2: Inject noise (only when a noise injector is provided)
        if self.noise_injector is not None:
            for param in self.net.parameters():
                if param.grad is not None:
                    noise = self.noise_injector.get_noise(param.grad).to(self.device)
                    param.grad.add_(noise)

    # ── Local training ─────────────────────────────────────────────────────

    def train(self, global_weights: dict, round_num: int, global_step_offset: int):
        """
        Load global weights, run LOCAL_EPOCHS of DP-SGD training.

        Returns:
            state_dict  – updated local model weights
            avg_loss    – mean per-batch loss over all local epochs
            steps_taken – total number of gradient steps
        """
        self.net.load_state_dict(global_weights)
        self.net.train()
        self.net.train_flag = True

        epoch_loss  = 0.0
        batch_count = 0
        local_step  = 0
        use_mse     = round_num < PRETRAIN_ROUNDS
        loss_label  = "MSE" if use_mse else "NLL"

        for _epoch in range(LOCAL_EPOCHS):
            for i, data in enumerate(self.dataloader):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

                if self.args["use_cuda"]:
                    hist    = hist.to(self.device)
                    nbrs    = nbrs.to(self.device)
                    mask    = mask.to(self.device)
                    lat_enc = lat_enc.to(self.device)
                    lon_enc = lon_enc.to(self.device)
                    fut     = fut.to(self.device)
                    op_mask = op_mask.to(self.device)

                self.optimizer.zero_grad()

                if self.args["use_maneuvers"]:
                    fut_pred, lat_pred, lon_pred = self.net(
                        hist, nbrs, mask, lat_enc, lon_enc
                    )
                    if use_mse:
                        loss = maskedMSE(fut_pred, fut, op_mask)
                    else:
                        loss = (
                            maskedNLL(fut_pred, fut, op_mask)
                            + self.cross_ent(lat_pred, lat_enc)
                            + self.cross_ent(lon_pred, lon_enc)
                        )
                else:
                    fut_pred = self.net(hist, nbrs, mask, lat_enc, lon_enc)
                    loss = (
                        maskedMSE(fut_pred, fut, op_mask)
                        if use_mse
                        else maskedNLL(fut_pred, fut, op_mask)
                    )

                loss.backward()

                # DP-SGD: clip + noise (replaces plain clip_grad_norm_ in notebook)
                self._dp_gradient_step()

                self.optimizer.step()

                epoch_loss  += loss.item()
                batch_count += 1
                local_step  += 1

                if (i + 1) % LOG_INTERVAL == 0:
                    logger.info(
                        "Client %d | Round %d | Batch %d | Loss (%s): %.4f",
                        self.client_id, round_num + 1, i + 1,
                        loss_label, loss.item(),
                    )

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        gc.collect()
        torch.cuda.empty_cache()
        return self.net.state_dict(), avg_loss, local_step


# ─────────────────────────────────────────────────────────────────────────────
#  FedAvg aggregation  (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def fed_avg(weights_list: list) -> dict:
    w_avg = copy.deepcopy(weights_list[0])
    for k in w_avg.keys():
        for i in range(1, len(weights_list)):
            w_avg[k] += weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights_list))
    return w_avg


# ─────────────────────────────────────────────────────────────────────────────
#  Validation  (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def validate_global_model(model, val_loader, round_num: int) -> float:
    model.eval()
    model.train_flag = False
    total_loss  = 0.0
    batch_count = 0
    use_mse     = round_num < PRETRAIN_ROUNDS

    with torch.no_grad():
        for data in val_loader:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
            if ARGS["use_cuda"]:
                hist    = hist.to(DEVICE)
                nbrs    = nbrs.to(DEVICE)
                mask    = mask.to(DEVICE)
                lat_enc = lat_enc.to(DEVICE)
                lon_enc = lon_enc.to(DEVICE)
                fut     = fut.to(DEVICE)
                op_mask = op_mask.to(DEVICE)

            if ARGS["use_maneuvers"]:
                if use_mse:
                    model.train_flag = True
                    fut_pred, _, _ = model(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedMSE(fut_pred, fut, op_mask)
                    model.train_flag = False
                else:
                    fut_pred, lat_pred, lon_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedNLLTest(
                        fut_pred, lat_pred, lon_pred, fut, op_mask, avg_along_time=True
                    )
            else:
                fut_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
                l = (
                    maskedMSE(fut_pred, fut, op_mask)
                    if use_mse
                    else maskedNLL(fut_pred, fut, op_mask)
                )

            total_loss  += l.item()
            batch_count += 1

    return total_loss / batch_count if batch_count > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Final evaluation – RMSE / ADE / FDE  (adapted from notebook cell-5)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, out_length: int, device: torch.device) -> dict:
    """Compute per-time-step RMSE, ADE, FDE in metres."""
    model.eval()
    model.train_flag = False

    loss_vals   = torch.zeros(out_length).to(device)
    counts      = torch.zeros(out_length).to(device)
    fde_vals    = torch.zeros(out_length).to(device)
    fde_counts  = torch.zeros(out_length).to(device)

    with torch.no_grad():
        for data in test_loader:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
            if ARGS["use_cuda"]:
                hist    = hist.to(device)
                nbrs    = nbrs.to(device)
                mask    = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut     = fut.to(device)
                op_mask = op_mask.to(device)

            if ARGS["use_maneuvers"]:
                fut_pred, lat_pred, lon_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
                # Select the highest-probability maneuver per sample
                fut_pred_max = torch.zeros_like(fut_pred[0])
                for k in range(lat_pred.shape[0]):
                    lat_man = torch.argmax(lat_pred[k, :]).detach()
                    lon_man = torch.argmax(lon_pred[k, :]).detach()
                    idx = lon_man * 3 + lat_man
                    fut_pred_max[:, k, :] = fut_pred[idx][:, k, :]
                pred = fut_pred_max
            else:
                pred = model(hist, nbrs, mask, lat_enc, lon_enc)

            # RMSE accumulation
            l, c = maskedMSETest(pred, fut, op_mask)
            t = l.shape[0]
            loss_vals[:t] += l.detach()
            counts[:t]    += c.detach()

            # ADE / FDE accumulation (Euclidean distance in feet → metres)
            pred_pos    = pred[:, :, 0:2]          # [T, B, 2]
            dist_l2     = torch.norm(pred_pos - fut, dim=2)   # [T, B]
            valid_mask  = op_mask[:, :, 0]         # [T, B]
            masked_dist = dist_l2 * valid_mask
            fde_vals[:t]   += torch.sum(masked_dist, dim=1).detach()
            fde_counts[:t] += torch.sum(valid_mask,  dim=1).detach()

    # Convert foot-scale to metres (* 0.3048)
    rmse = torch.pow(loss_vals / counts, 0.5) * 0.3048
    fde  = (fde_vals / fde_counts) * 0.3048
    ade  = (torch.cumsum(fde_vals, dim=0) / torch.cumsum(fde_counts, dim=0)) * 0.3048

    return {
        "rmse": rmse.cpu().numpy(),
        "ade":  ade.cpu().numpy(),
        "fde":  fde.cpu().numpy(),
    }


def print_results_table(metrics: dict, out_length: int):
    horizons = [h for h in [5, 10, 15, 20, 25] if h <= out_length]
    headers  = [f"{h / 5:.1f}s" for h in horizons]
    rows = {
        "RMSE": [f"{metrics['rmse'][h - 1]:.3f}" for h in horizons],
        "ADE":  [f"{metrics['ade'][h - 1]:.3f}"  for h in horizons],
        "FDE":  [f"{metrics['fde'][h - 1]:.3f}"  for h in horizons],
    }
    col_w = 9
    sep   = "=" * (12 + (col_w + 3) * len(horizons))
    print(f"\n{sep}")
    print("EVALUATION RESULTS  (Metres)  –  DP config:", DP_CONFIG)
    print(sep)
    print(f"{'Metric':<12} | " + " | ".join(f"{h:<{col_w}}" for h in headers))
    print("-" * len(sep))
    for label, row in rows.items():
        print(f"{label:<12} | " + " | ".join(f"{v:<{col_w}}" for v in row))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 68)
    logger.info("  Federated Trajectory Prediction with Differential Privacy")
    logger.info("=" * 68)
    logger.info("DP Config  : %s", DP_CONFIG)
    logger.info("FL Config  : clients=%d  rounds=%d  local_epochs=%d  batch=%d",
                NUM_CLIENTS, GLOBAL_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE)
    logger.info("Device     : %s", DEVICE)

    # ── Build noise injector ─────────────────────────────────────────────
    noise_injector = build_noise_injector(DP_CONFIG)
    if noise_injector is None:
        logger.info("[DP] Differential privacy DISABLED – running vanilla FL.")
    else:
        logger.info("[DP] Noise injector ready: %s", type(noise_injector).__name__)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset    = ngsimDataset(os.path.join(DATA_DIR, "TrainSet.mat"))
    val_dataset_full = ngsimDataset(os.path.join(DATA_DIR, "ValSet.mat"))

    val_loader = DataLoader(
        Subset(val_dataset_full, list(range(len(val_dataset_full)))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset_full.collate_fn,
    )

    # ── Partition data across clients ────────────────────────────────────
    indices    = list(range(len(train_dataset)))
    split_size = len(train_dataset) // NUM_CLIENTS

    clients = []
    for i in range(NUM_CLIENTS):
        subset = Subset(train_dataset, indices[i * split_size:(i + 1) * split_size])
        clients.append(
            DPFLClient(
                client_id=i,
                dataset=subset,
                device=DEVICE,
                args=ARGS,
                noise_injector=noise_injector,
                max_grad_norm=DP_CONFIG["max_grad_norm"],
            )
        )
    logger.info("Created %d clients, each with ~%d samples.", NUM_CLIENTS, split_size)

    # ── Global model ─────────────────────────────────────────────────────
    global_model   = highwayNet(ARGS).to(DEVICE)
    global_weights = global_model.state_dict()

    if REUSE_WEIGHTS and CHECKPOINT_PATH:
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        global_model.load_state_dict(state.get("state_dict", state))
        global_weights = global_model.state_dict()
        logger.info("Loaded pretrained weights from  %s", CHECKPOINT_PATH)

    os.makedirs("trained_models_dp", exist_ok=True)

    # ── Federated rounds ─────────────────────────────────────────────────
    global_step_tracker = 0

    for round_num in range(GLOBAL_ROUNDS):
        logger.info("\n--- Global Round %d / %d ---", round_num + 1, GLOBAL_ROUNDS)

        local_weights_list = []
        local_losses       = []
        max_steps          = 0

        for client in clients:
            w_local, loss, steps = client.train(
                global_weights, round_num, global_step_tracker
            )
            local_weights_list.append(w_local)
            local_losses.append(loss)
            max_steps = max(max_steps, steps)

        global_step_tracker += max_steps

        # FedAvg
        global_weights = fed_avg(local_weights_list)
        global_model.load_state_dict(global_weights)

        # Validation
        val_loss   = validate_global_model(global_model, val_loader, round_num)
        avg_train  = sum(local_losses) / len(local_losses)
        loss_label = "MSE" if round_num < PRETRAIN_ROUNDS else "NLL"

        logger.info(
            "Round %d | Avg Train (%s): %.4f | Val Loss: %.4f",
            round_num + 1, loss_label, avg_train, val_loss,
        )

        save_path = f"trained_models_dp/fl_dp_round_{round_num + 1}.tar"
        torch.save(global_weights, save_path)
        logger.info("Checkpoint saved → %s", save_path)

    logger.info("\nFederated Learning (DP) Complete.")

    # ── Final evaluation ─────────────────────────────────────────────────
    test_path = os.path.join(DATA_DIR, "TestSet_Keep.mat")
    if os.path.exists(test_path):
        logger.info("Loading test set from  %s ...", test_path)
        test_dataset = ngsimDataset(test_path)
        test_loader  = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            collate_fn=test_dataset.collate_fn,
        )
        metrics = evaluate_model(global_model, test_loader, ARGS["out_length"], DEVICE)
        print_results_table(metrics, ARGS["out_length"])
    else:
        logger.warning("Test set not found at %s – skipping evaluation.", test_path)


if __name__ == "__main__":
    main()