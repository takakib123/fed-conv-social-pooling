"""
tappfl_trajectory_prediction.py
================================
Task-Agnostic Privacy-Preserving Federated Learning (TAPPFL) applied to
vehicle trajectory prediction on the NGSIM highway dataset.

Background
----------
In a standard federated learning (FL) setup, clients train a shared model on
their local data and periodically upload model weights to a central server.
Even though raw data never leaves the client, the server can still run
*attribute inference attacks* — training a classifier on the shared
representations to predict sensitive properties of each client's data. In
this project the sensitive attribute is **dsId**: which of the 2 NGSIM highway
segments a vehicle was driving on.

TAPPFL defends against this by adding a mutual information (MI) regularisation
term to the training objective. The FE is penalised whenever its output z
carries information about dsId, so it is forced to produce representations
that are useful for prediction but location-agnostic.

Model decomposition
-------------------
The original highwayNet (Convolutional Social Pooling) is split into three
independently-optimised parts:

    ┌─────────────────────────────────────────────────────────────┐
    │  HighwayNetFE  (Feature Extractor)  ← FEDERATED (FedAvg)   │
    │    ip_emb → enc_lstm → dyn_emb                              │
    │    soc_conv → conv_3x1 → soc_maxpool                        │
    │    Output: z ∈ R^112  (80 social + 32 dynamic)             │
    ├─────────────────────────────────────────────────────────────┤
    │  TrajectoryDecoder  (Task Classifier)  ← LOCAL only         │
    │    dec_lstm → op  +  maneuver heads (op_lat, op_lon)        │
    │    Predicts: 5-param Gaussian per future step               │
    ├─────────────────────────────────────────────────────────────┤
    │  TrajectoryMIEstimator  (MI Estimator)  ← LOCAL only        │
    │    Two-stream MLP: encodes (x, z) separately then fuses     │
    │    Estimates JSD MI between z and dsId                      │
    └─────────────────────────────────────────────────────────────┘

Training objective (per client, per round)
------------------------------------------
    L_task   = NLL (or MSE in early rounds) on trajectory prediction
    L_JSD    = JSD-based MI estimate between z and dsId
    L_total  = λ · L_task + (1-λ) · L_JSD

    FE        ← updated with L_total  (task-useful AND location-agnostic z)
    Decoder   ← updated with L_task   (task quality, independently of privacy)
    MI est.   ← updated with L_JSD    (adversarial: become a better discriminator)

Only FE weights are sent to the server after each round.
Decoder and MI estimator remain on the client permanently.

Usage
-----
    # From inside the fed-conv-social-pooling directory:
    python tappfl_trajectory_prediction.py

Dependencies
------------
    pip install wandb
    # model.py and utils.py must be in the working directory
    # data/TrainSet.mat, ValSet.mat, TestSet.mat must exist

Private attribute mapping
-------------------------
    dsId 1, 2, 3  →  US-101 highway  →  location_label 0
    dsId 4, 5, 6  →  I-80 highway    →  location_label 1
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import copy
import gc
import os
import math
import logging
import time

import numpy as np
import scipy.io as scp
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):   # type: ignore[misc]
        return iterable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

    class _WandbStub:
        """No-op wandb stub so the script runs without a wandb installation."""
        def init(self, **kwargs):   return self
        def log(self, *a, **kw):    pass
        def finish(self, **kwargs): pass
        def __enter__(self):        return self
        def __exit__(self, *a):     pass

    wandb = _WandbStub()

# Local modules — must be present in the working directory
from model import highwayNet  # noqa: F401  kept for weight-compatibility checks
from utils import (
    ngsimDataset,
    maskedNLL,       # noqa: F401  available for post-hoc trajectory eval
    maskedMSE,       # noqa: F401  available for post-hoc trajectory eval
    maskedNLLTest,   # noqa: F401  available for post-hoc trajectory eval
    outputActivation,
)

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('TAPPFL')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {DEVICE}')


# =============================================================================
# Configuration
# =============================================================================
# All hyperparameters and paths are centralised here.
# To reproduce the notebook exactly, do not change these values unless noted.

# ── Model architecture (must match the FL checkpoint being loaded) ────────────
ARGS = {
    'use_cuda':             torch.cuda.is_available(),
    'encoder_size':         64,    # LSTM hidden size for ego trajectory encoder
    'decoder_size':         128,   # LSTM hidden size for trajectory decoder
    'in_length':            16,    # number of history steps fed to the encoder
    'out_length':           25,    # number of future steps to predict (25 × 0.2 s = 5 s)
    'grid_size':            (13, 3),  # social pooling grid: 13 longitudinal × 3 lateral
    'soc_conv_depth':       64,    # channels after first convolutional social pooling layer
    'conv_3x1_depth':       16,    # channels after second (3×1) convolution
    'dyn_embedding_size':   32,    # dimension of the dynamic (ego) embedding
    'input_embedding_size': 32,    # dimension of the (x, y) → embedding projection
    'num_lat_classes':      3,     # lateral maneuver classes: left / keep / right
    'num_lon_classes':      2,     # longitudinal maneuver classes: brake / accelerate
    'use_maneuvers':        True,  # True = multi-modal maneuver-conditioned decoder
    'train_flag':           True,  # True = teacher-forced training mode
}

# ── FE output dimension (derived from ARGS, do not set manually) ─────────────
# soc_embedding = (((grid_w − 4) + 1) // 2) × conv_3x1_depth
#               = (((13 − 4) + 1) // 2) × 16 = (10 // 2) × 16 = 5 × 16 = 80
_soc_emb_size = (((ARGS['grid_size'][0] - 4) + 1) // 2) * ARGS['conv_3x1_depth']
FE_OUT_DIM    = _soc_emb_size + ARGS['dyn_embedding_size']  # 80 + 32 = 112

# ── TAPPFL hyperparameters ────────────────────────────────────────────────────
NUM_DS_IDS      = 2     # binary private attribute: 0 = US-101 (dsId 1-3), 1 = I-80 (dsId 4-6)
TRADEOFF_LAMBDA = 0.5   # λ: trade-off weight
                        #   λ → 1.0 : more task utility, less privacy protection
                        #   λ → 0.0 : more privacy protection, less task utility

# ── MI estimator input dimensions (derived from ARGS) ────────────────────────
HIST_STEPS = ARGS['in_length']  # 16 timesteps in the ego history window
HIST_DIM   = HIST_STEPS * 2    # flatten (x, y) per step → 32-dimensional vector

# ── Federated learning settings ───────────────────────────────────────────────
NUM_CLIENTS     = 10    # IID split across 10 clients
GLOBAL_ROUNDS   = 1    # total number of FL communication rounds
PRETRAIN_ROUNDS = 1     # use MSE loss for the first N rounds (federated pretraining)
BATCH_SIZE      = 128
LR_INIT         = 1e-3  # learning rate; reduced 10× after round 12 (fine-tuning)

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'NGSIM/data/TrainSet.mat'
VAL_DATA_PATH   = 'NGSIM/data/ValSet.mat'
TEST_DATA_PATH  = 'NGSIM/data/TestSet.mat'

# ── Checkpoint settings ───────────────────────────────────────────────────────
CHECKPOINT_DIR = 'pretrained_models/tappfl'
# Set REUSE_WEIGHTS = True and supply PRETRAIN_CKPT to skip centralised pretraining
REUSE_WEIGHTS  = False
PRETRAIN_CKPT  = None  # e.g. 'trained_models/tappfl/pretrained_fe.tar'


# =============================================================================
# Dataset: ngsimDatasetWithLocation
# =============================================================================

class ngsimDatasetWithLocation(Dataset):
    """
    NGSIM trajectory dataset extended to also return the highway segment ID
    (dsId) as a private attribute label.

    The original ngsimDataset returns per-sample:
        (hist, fut, neighbours, lat_enc, lon_enc)

    This subclass adds one extra value at the end of each sample:
        location_label  — int in [0, 5], equal to dsId − 1 (zero-indexed)

    Why dsId?
        dsId identifies which of the 6 NGSIM recording sessions the sample
        came from, and therefore which road segment. An honest-but-curious
        server could infer this from the learned representations z and thereby
        discover the physical location of the data.

    Segment mapping:
        dsId 1, 2, 3  →  US-101 freeway  →  location_label 0
        dsId 4, 5, 6  →  I-80  freeway   →  location_label 1
    """

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        """
        Args:
            mat_file   : path to .mat file (TrainSet / ValSet / TestSet)
            t_h        : history length in raw frames before downsampling
            t_f        : future length in raw frames before downsampling
            d_s        : downsampling stride (every d_s-th frame is kept)
            enc_size   : encoder hidden size — needed to size the social mask
            grid_size  : (longitudinal bins, lateral bins) of the surrounding grid
        """
        self.D         = scp.loadmat(mat_file)['traj']    # trajectory index table
        self.T         = scp.loadmat(mat_file)['tracks']  # per-vehicle track arrays
        self.t_h       = t_h
        self.t_f       = t_f
        self.d_s       = d_s
        self.enc_size  = enc_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        """
        Returns one training sample.

        Output tuple:
            hist           : (T_h, 2)       ego vehicle relative (x, y) history
            fut            : (T_f, 2)       ego vehicle relative (x, y) future
            neighbors      : list[ndarray]  one (T_h, 2) array per grid cell;
                                            empty array (0, 2) if cell is vacant
            lat_enc        : (3,)           one-hot lateral maneuver label
            lon_enc        : (2,)           one-hot longitudinal maneuver label
            location_label : int            dsId − 1  (the private attribute)
        """
        dsId  = self.D[idx, 0].astype(int)  # 1-indexed segment ID from the .mat file
        vehId = self.D[idx, 1].astype(int)
        t     = self.D[idx, 2]              # current frame timestamp
        grid  = self.D[idx, 8:]             # surrounding vehicle IDs in the social grid

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut  = self.getFuture(vehId, t, dsId)

        # Collect the history of every vehicle occupying a grid cell around the ego
        neighbors = [self.getHistory(i.astype(int), t, vehId, dsId) for i in grid]

        # One-hot encode the ground-truth maneuver labels
        lon_enc = np.zeros([2]); lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3]); lat_enc[int(self.D[idx, 6] - 1)] = 1

        # Binary location label: US-101 (dsId 1-3) → 0, I-80 (dsId 4-6) → 1
        location_label = 0 if dsId <= 3 else 1

        return hist, fut, neighbors, lat_enc, lon_enc, location_label

    def getHistory(self, vehId, t, refVehId, dsId):
        """
        Fetch the past trajectory of vehicle `vehId` expressed relative to
        `refVehId`'s position at time `t`.

        Relative coordinates mean the ego vehicle always starts at the origin,
        which makes the representations translation-invariant.

        Returns:
            ndarray of shape (T_h, 2), or (0, 2) if the vehicle is absent
        """
        if vehId == 0:
            return np.empty([0, 2])
        if self.T.shape[1] <= vehId - 1:
            return np.empty([0, 2])
        refTrack = self.T[dsId - 1][refVehId - 1].transpose()
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos   = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]
        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 2])
        tidx     = np.argwhere(vehTrack[:, 0] == t).flat[0]
        stpt     = np.maximum(0, tidx - self.t_h)
        enpt     = tidx + 1
        hist     = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 2])
        return hist

    def getFuture(self, vehId, t, dsId):
        """
        Fetch the future trajectory of vehicle `vehId` relative to its own
        position at time `t`.

        Returns:
            ndarray of shape (T_f, 2)
        """
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos   = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        tidx     = np.argwhere(vehTrack[:, 0] == t).flat[0]
        stpt     = tidx + self.d_s
        enpt     = np.minimum(len(vehTrack), tidx + self.t_f + 1)
        fut      = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def collate_fn(self, samples):
        """
        Custom collate function.

        Neighbour sequences have variable length (some grid cells are empty),
        so they are packed into a single dense tensor and an occupancy mask
        is built to tell the convolutional pooling layer which cells are active.

        Compared to the original ngsimDataset.collate_fn, this version
        additionally stacks the location labels into a LongTensor.

        Returns (all PyTorch tensors):
            hist_batch    : (T_h, B, 2)
            nbrs_batch    : (T_h, N_nbrs_total, 2)   packed neighbour sequences
            mask_batch    : (B, grid_h, grid_w, enc_size)   occupancy mask
            lat_enc_batch : (B, 3)
            lon_enc_batch : (B, 2)
            fut_batch     : (T_f, B, 2)
            op_mask_batch : (T_f, B, 2)   1 where ground-truth future exists
            loc_batch     : (B,)  LongTensor   zero-indexed location label
        """
        # Count total occupied grid cells across the whole batch —
        # this determines how wide nbrs_batch needs to be
        nbrs_batch_size = 0
        for _, _, nbrs, _, _, _ in samples:
            nbrs_batch_size += sum([1 for n in nbrs if len(n) > 0])

        maxlen        = self.t_h // self.d_s + 1   # history steps after downsampling
        nbrs_mat_size = (maxlen, nbrs_batch_size, 2)

        # Pre-allocate all output tensors
        nbrs_batch    = torch.zeros(nbrs_mat_size)
        mask_batch    = torch.zeros(len(samples), self.grid_size[1],
                                    self.grid_size[0], self.enc_size)
        hist_batch    = torch.zeros(maxlen, len(samples), 2)
        fut_batch     = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        loc_batch     = torch.zeros(len(samples), dtype=torch.long)

        count = 0  # running pointer into the packed nbrs_batch dimension
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, loc) in enumerate(samples):
            # History is right-aligned so the most recent step is at index -1.
            # Guard against len==0: -0 == 0 in Python, which would wrongly
            # select the entire tensor instead of a zero-length slice.
            if len(hist) > 0:
                hist_batch[-len(hist):, sampleId, :] = torch.from_numpy(hist).float()
            fut_batch[:len(fut), sampleId, :]    = torch.from_numpy(fut).float()
            op_mask_batch[:len(fut), sampleId, :] = 1  # mark valid future positions
            lat_enc_batch[sampleId, :]            = torch.from_numpy(lat_enc).float()
            lon_enc_batch[sampleId, :]            = torch.from_numpy(lon_enc).float()
            loc_batch[sampleId]                   = loc

            # Pack each occupied neighbour cell into the dense neighbour tensor
            # and set the corresponding position in the social occupancy mask
            for id, nbr in enumerate(nbrs):
                if len(nbr) > 0:
                    nbrs_batch[-len(nbr):, count, :] = torch.from_numpy(nbr).float()
                    pos = id % self.grid_size[0]
                    mask_batch[sampleId, id // self.grid_size[0], pos, :] = \
                        torch.ones(self.enc_size).byte()
                    count += 1

        return (hist_batch, nbrs_batch, mask_batch,
                lat_enc_batch, lon_enc_batch,
                fut_batch, op_mask_batch, loc_batch)


# =============================================================================
# Model Architecture
# =============================================================================
# highwayNet is split into three components at the boundary between
# the encoder (FE) and the decoder (Task Classifier).


class HighwayNetFE(nn.Module):
    """
    Feature Extractor — the shared, federated component.

    Replicates the encoder half of highwayNet exactly so that pretrained
    weights from cslstm_m.tar can be loaded by mapping matching parameter names.

    Architecture:
        1. ip_emb      Linear(2 → 32)         embed raw (x, y) coordinates
        2. enc_lstm    LSTM(32 → 64)           encode ego history; also reused
                                               for neighbour trajectories
        3. dyn_emb     Linear(64 → 32)         compress LSTM hidden → dynamic embedding
        4. soc_conv    Conv2d(64, 64, 3×3)     convolve over the social grid
        5. conv_3x1    Conv2d(64, 16, 3×1)     reduce spatial extent
        6. soc_maxpool MaxPool2d(2×1)          pool → flatten to social embedding
        7. concat      [soc_enc(48), dyn_enc(32)] → z(80)

    The FE weights are the only weights uploaded to the server.
    After FedAvg, the global FE weights are broadcast back to all clients.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder_size         = args['encoder_size']
        self.input_embedding_size = args['input_embedding_size']
        self.dyn_embedding_size   = args['dyn_embedding_size']
        self.soc_conv_depth       = args['soc_conv_depth']
        self.conv_3x1_depth       = args['conv_3x1_depth']
        self.grid_size            = args['grid_size']
        # Spatial pooling output width after conv + maxpool
        self.soc_embedding_size   = (
            (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth
        )

        self.ip_emb   = nn.Linear(2, self.input_embedding_size)
        self.enc_lstm = nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        self.dyn_emb  = nn.Linear(self.encoder_size, self.dyn_embedding_size)

        self.soc_conv    = nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1    = nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = nn.MaxPool2d((2, 1), padding=(1, 0))

        self.leaky_relu = nn.LeakyReLU(0.1)

    @property
    def out_dim(self):
        """Dimension of the output representation z."""
        return self.soc_embedding_size + self.dyn_embedding_size

    def forward(self, hist, nbrs, masks):
        """
        Args:
            hist  : (T_h, B, 2)                   ego trajectory history
            nbrs  : (T_h, N_nbrs_total, 2)         packed neighbour sequences
            masks : (B, grid_h, grid_w, enc_size)  social occupancy mask

        Returns:
            z : (B, FE_OUT_DIM)  the joint social-dynamic representation
        """
        # ── Ego history encoding ──────────────────────────────────────────────
        # Only the final hidden state is kept; it summarises the entire history
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(
            self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2]))
        )   # → (B, dyn_embedding_size=32)

        # ── Neighbour encoding (shares weights with ego encoder) ──────────────
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        # → (N_nbrs_total, encoder_size=64)

        # ── Scatter neighbour encodings into the spatial grid ─────────────────
        # masked_scatter_ places each packed neighbour encoding into the
        # (batch, row, col, channel) position indicated by the occupancy mask
        masks   = masks.bool()
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)   # → (B, encoder_size, grid_w, grid_h)

        # ── Convolutional social pooling ──────────────────────────────────────
        # soc_conv extracts local spatial interaction patterns from the grid
        # conv_3x1 + maxpool compresses the spatial dimensions into a flat vector
        soc_enc = self.soc_maxpool(
            self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc))))
        )
        soc_enc = soc_enc.view(-1, self.soc_embedding_size)
        # → (B, soc_embedding_size=48)

        # ── Concatenate social context + ego dynamics → z ─────────────────────
        z = torch.cat((soc_enc, hist_enc), dim=1)   # → (B, 112)
        return z


class TrajectoryDecoder(nn.Module):
    """
    Task Classifier — local component, never sent to the server.

    Mirrors the decoder half of highwayNet exactly:
        - op_lat  Linear(80 → 3)         predict lateral maneuver class
        - op_lon  Linear(80 → 2)         predict longitudinal maneuver class
        - dec_lstm  LSTM(80+3+2 → 128)   decode z + maneuver one-hot over time
        - op      Linear(128 → 5)        output 5 Gaussian parameters per step
                                          (μ_x, μ_y, σ_x, σ_y, ρ)

    train_flag controls two modes:
        True  (training): decoder receives teacher-forced ground-truth maneuver
                          labels → single-mode prediction, fast
        False (eval):     decoder enumerates all 6 maneuver combinations
                          → returns 6 trajectory predictions for marginalisation
    """

    def __init__(self, args):
        super().__init__()
        self.use_maneuvers   = args['use_maneuvers']
        self.train_flag      = args['train_flag']
        self.decoder_size    = args['decoder_size']
        self.out_length      = args['out_length']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']

        # FE output dimension (input to this decoder)
        _soc = (((args['grid_size'][0] - 4) + 1) // 2) * args['conv_3x1_depth']
        self.fe_dim = _soc + args['dyn_embedding_size']   # = 112

        dec_input = (self.fe_dim + self.num_lat_classes + self.num_lon_classes
                     if self.use_maneuvers else self.fe_dim)

        self.dec_lstm = nn.LSTM(dec_input, self.decoder_size)
        self.op       = nn.Linear(self.decoder_size, 5)    # → Gaussian params
        self.op_lat   = nn.Linear(self.fe_dim, self.num_lat_classes)
        self.op_lon   = nn.Linear(self.fe_dim, self.num_lon_classes)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax    = nn.Softmax(dim=1)

    def decode(self, enc):
        """
        Unroll the LSTM decoder over out_length steps.

        The same encoding vector `enc` is fed at every timestep as a
        constant input — the LSTM unrolls purely from its recurrent state.

        Args:
            enc : (B, dec_input_dim)

        Returns:
            fut_pred : (T_f, B, 5)  Gaussian trajectory distribution
        """
        enc      = enc.repeat(self.out_length, 1, 1)   # (T_f, B, dec_input)
        h_dec, _ = self.dec_lstm(enc)
        h_dec    = h_dec.permute(1, 0, 2)              # (B, T_f, decoder_size)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)            # (T_f, B, 5)
        fut_pred = outputActivation(fut_pred)            # softplus σ, tanh ρ
        return fut_pred

    def forward(self, z, lat_enc, lon_enc):
        """
        Args:
            z       : (B, fe_dim)  representation from HighwayNetFE
            lat_enc : (B, 3)       lateral maneuver one-hot (GT in train mode)
            lon_enc : (B, 2)       longitudinal maneuver one-hot

        Returns (train mode, train_flag=True):
            fut_pred : (T_f, B, 5)
            lat_pred : (B, 3)
            lon_pred : (B, 2)

        Returns (eval mode, train_flag=False):
            fut_pred : list of 6 × (T_f, B, 5)  one per maneuver combination
            lat_pred : (B, 3)
            lon_pred : (B, 2)
        """
        if self.use_maneuvers:
            lat_pred = self.softmax(self.op_lat(z))
            lon_pred = self.softmax(self.op_lon(z))

            if self.train_flag:
                # Teacher forcing: condition on ground-truth maneuver labels
                enc      = torch.cat((z, lat_enc, lon_enc), dim=1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                # Enumerate all 3 lat × 2 lon = 6 maneuver hypotheses
                fut_pred = []
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_tmp = torch.zeros_like(lat_enc); lat_tmp[:, l] = 1
                        lon_tmp = torch.zeros_like(lon_enc); lon_tmp[:, k] = 1
                        enc_tmp = torch.cat((z, lat_tmp, lon_tmp), dim=1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(z)
            return fut_pred


class TrajectoryMIEstimator(nn.Module):
    """
    Mutual Information Estimator — local component, never sent to the server.

    Estimates the JSD mutual information between:
        x : raw trajectory history (input to FE)
        z : learned representation (output of FE)
        u : private attribute label (dsId, scalar)

    Why estimate MI?
        Minimising I(z; dsId) forces the FE to produce representations that
        are statistically independent of which highway segment the data came
        from. The MI estimator plays the role of the adversary in the
        min-max game — it is trained to be a better location discriminator,
        which in turn forces the FE to hide more location information.

    Architecture:
        hist_encoder  3-layer MLP  x  → 256-dim embedding
        z_encoder     2-layer MLP  z  → 256-dim embedding
        classifier    3-layer MLP  [hist_enc | z_enc | u(scalar)] → logits

    The two separate encoder streams allow the estimator to independently
    process the raw input and the learned representation before fusing them.
    """

    def __init__(self, fe_out_dim: int, num_private_classes: int,
                 hist_dim: int = 32, hidden_dim: int = 256):
        """
        Args:
            fe_out_dim          : dimension of z (= FE_OUT_DIM = 80)
            num_private_classes : number of dsId classes (= NUM_DS_IDS = 6)
            hist_dim            : length of flattened history (T_h × 2 = 32)
            hidden_dim          : hidden size in both encoder streams (256)
        """
        super().__init__()

        # Stream 1: encode the raw ego trajectory history x
        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stream 2: encode the FE representation z
        self.z_encoder = nn.Sequential(
            nn.Linear(fe_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier: [hist_enc(256) | z_enc(256) | u(1)] → location logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_private_classes),
        )

    def forward(self, x, z, u):
        """
        Args:
            x : (B, hist_dim)    flattened ego history (from prepare_x_for_mi)
            z : (B, fe_out_dim)  learned representation from HighwayNetFE
            u : (B,)             private label (dsId as float scalar)

        Returns:
            (B, num_private_classes)  unnormalised location scores
        """
        h_x = self.hist_encoder(x)           # (B, hidden_dim)
        h_z = self.z_encoder(z)              # (B, hidden_dim)
        u_  = u.float().unsqueeze(1)         # (B, 1)
        out = self.classifier(torch.cat([h_x, h_z, u_], dim=1))
        return out


class AttributeClassifier(nn.Module):
    """
    Privacy network g_Ψ — local component, never sent to the server.

    A simple MLP that predicts the private attribute u (highway segment dsId)
    from the FE representation r. Plays the adversary role in the min-max game:

        g_Ψ  minimises  L_priv = CE(g_Ψ(r), u)   → becomes a better classifier
        f_Θ  minimises −λ · L_priv               → fools g_Ψ (location-agnostic r)

    Architecture:
        Linear(fe_out_dim → hidden_dim) → ReLU
        Linear(hidden_dim → hidden_dim) → ReLU
        Linear(hidden_dim → num_classes)           (logits, no softmax)
    """

    def __init__(self, fe_out_dim: int, num_classes: int, hidden_dim: int = 128):
        """
        Args:
            fe_out_dim  : dimension of r (= FE_OUT_DIM)
            num_classes : number of private attribute classes (= NUM_DS_IDS)
            hidden_dim  : width of the two hidden layers (default 128)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fe_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, r):
        """
        Args:
            r : (B, fe_out_dim)  representation from HighwayNetFE

        Returns:
            (B, num_classes)  unnormalised class logits
        """
        return self.net(r)


# =============================================================================
# JSD Mutual Information Loss
# =============================================================================

def jsd_MI(mi_estimator, x, z, u, x_prime):
    """
    Jensen-Shannon Divergence based MI estimation (TAPPFL paper, eq. 7).

    Approximates I(x; z | u) using the f-divergence variational lower bound:

        JSD ≈ E_{p(x,z)}   [ −softplus( −T(x,  z, u) ) ]
             − E_{p(x)p(z)} [  softplus(  T(x', z, u) ) ]

    where T is the MI estimator network and x' is an independent marginal
    sample (drawn from p(x) but paired with a different z).

    Game-theoretic interpretation:
        • The FE MINIMISES this quantity: drive z to be location-agnostic
        • The MI estimator MAXIMISES this: become a better location discriminator
        This adversarial dynamic is what enforces privacy.

    Args:
        mi_estimator : TrajectoryMIEstimator
        x            : (B, hist_dim)   joint sample (x paired with its own z)
        z            : (B, fe_dim)     learned representation
        u            : (B,)            private label as float
        x_prime      : (B, hist_dim)   marginal sample — same distribution as x
                                        but independent of z (cyclic shift)

    Returns:
        JSD scalar tensor
    """
    # Joint term: E_{p(x,z)}[ −softplus(−T(x,z,u)) ]
    # A perfect discriminator produces large positive T → −softplus(−large) ≈ 0
    Ej = (-F.softplus(-mi_estimator(x,       z, u))).mean()

    # Marginal term: E_{p(x')p(z)}[ softplus(T(x',z,u)) ]
    # For uncorrelated (x', z) the discriminator produces small T → softplus(small) ≈ small
    Em = ( F.softplus( mi_estimator(x_prime, z, u))).mean()

    return Ej - Em


def prepare_x_for_mi(hist):
    """
    Reshape the ego trajectory history for input to TrajectoryMIEstimator.

    The dataloader produces hist with shape (T, B, 2) — time-major to match
    PyTorch LSTM convention. The MI estimator expects a flat row vector per
    sample: (B, T*2).

    Args:
        hist : (T, B, 2)

    Returns:
        (B, T*2)  each sample's full history as a single flat vector
    """
    T, B, C = hist.shape
    return hist.permute(1, 0, 2).reshape(B, T * C)


def make_x_prime(x):
    """
    Create a marginal sample x' by cyclically shifting x by one position.

    x[i] and z[i] come from the same data point (joint distribution).
    x'[i] = x[(i+1) % B], so x' and z are drawn from the product of marginals —
    their pairing is broken while x' still follows the same distribution as x.

    This is the same trick used in the TAPPFL codebase.

    Args:
        x : (B, hist_dim)

    Returns:
        x_prime : (B, hist_dim)
    """
    return torch.cat([x[1:], x[0].unsqueeze(0)], dim=0)


# =============================================================================
# TAPPFL Client
# =============================================================================

class TAPPFLClient:
    """
    One federated client in the TAPPFL system.

    Each client holds an IID shard of the full training dataset and
    two local model components:

        self.attr_cls  AttributeClassifier   (g_Ψ) persists across rounds, never shared
        self.mi_est    TrajectoryMIEstimator  (h_Ω) persists across rounds, never shared

    The FE (f_Θ) is NOT stored on the client between rounds. Each call to train()
    creates a fresh local FE from the global weights, trains it for one epoch,
    then returns the updated weights to the server and discards the local copy.

    Per-batch training sequence (TAPPFL adversarial formulation)
    ------------------------------------------------------------
    1.  r       = f_Θ(x)                       forward pass through FE
        pred_u  = g_Ψ(r)                       attribute prediction

    2.  L_priv  = CE(pred_u, u)                cross-entropy on private attribute
        L_JSD   = −jsd_MI(h_Ω, x, r, u, x̃)   negative MI estimate  (≤ 0)
        L_total = −λ · L_priv + (1−λ) · L_JSD

    3.  Optimise f_Θ  with L_total   (FE fools g_Ψ AND minimises MI)
        Optimise g_Ψ  with L_priv    (classifier becomes a better adversary)
        Optimise h_Ω  with L_JSD     (MI estimator becomes a better discriminator)

    After all batches, return fe.state_dict() to the server.
    """

    def __init__(self, client_id, dataset_subset, device, args,
                 lam=TRADEOFF_LAMBDA):
        """
        Args:
            client_id      : integer segment index (0–5)
            dataset_subset : Subset of ngsimDatasetWithLocation
            device         : torch.device
            args           : model architecture dict (ARGS)
            lam            : λ trade-off parameter
        """
        self.client_id = client_id
        self.device    = device
        self.args      = args
        self.lam       = lam

        self.loader = DataLoader(
            dataset_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset_subset.dataset.collate_fn,
        )

        # Local models — created once, updated every round, never shared
        self.attr_cls = AttributeClassifier(
            fe_out_dim=FE_OUT_DIM,
            num_classes=NUM_DS_IDS,
        ).to(device)
        self.mi_est = TrajectoryMIEstimator(
            fe_out_dim=FE_OUT_DIM,
            num_private_classes=NUM_DS_IDS,
            hist_dim=HIST_DIM,
        ).to(device)

        self.ce_loss = nn.CrossEntropyLoss()

    def train(self, global_fe_weights: dict, round_num: int,
              global_step_offset: int = 0):
        """
        Execute one round of TAPPFL local training.

        Args:
            global_fe_weights  : state_dict broadcast from the server
            round_num          : 0-indexed round counter
            global_step_offset : WandB step offset for consistent x-axis alignment

        Returns:
            fe_weights  : updated FE state_dict (returned to server for FedAvg)
            avg_loss    : average L_total over all batches in this round
            steps_taken : number of batches processed
        """
        # Re-initialise local FE from the latest global weights.
        # This is the standard FedAvg pull step.
        fe = HighwayNetFE(self.args).to(self.device)
        fe.load_state_dict(global_fe_weights)
        fe.train()
        self.attr_cls.train()
        self.mi_est.train()

        # LR schedule: full LR for the first 12 rounds, then 10× smaller
        lr = LR_INIT * (0.1 if round_num >= 12 else 1.0)
        opt_fe  = torch.optim.Adam(fe.parameters(),             lr=lr, weight_decay=1e-4)
        opt_cls = torch.optim.Adam(self.attr_cls.parameters(),  lr=lr, weight_decay=1e-4)
        opt_mi  = torch.optim.Adam(self.mi_est.parameters(),    lr=lr, weight_decay=1e-4)

        total_loss_sum = 0.0
        priv_loss_sum  = 0.0
        jsd_loss_sum   = 0.0
        batch_count    = 0

        pbar = tqdm(self.loader,
                    desc=f'  Client {self.client_id} R{round_num+1}',
                    leave=False, unit='batch')
        for hist, nbrs, mask, _, _, _, _, loc in pbar:
            hist = hist.to(self.device)
            nbrs = nbrs.to(self.device)
            mask = mask.to(self.device)
            loc  = loc.to(self.device)   # (B,)  private location labels

            # Shared inputs computed once; reused across the three update steps.
            x_flat  = prepare_x_for_mi(hist)                        # (B, HIST_DIM)
            x_tilde = torch.cat([x_flat[1:], x_flat[0:1]], dim=0)  # cyclic marginal
            u_float = loc.float()

            # ── Step 1: Update f_Θ (FE) ──────────────────────────────────────
            # Full forward pass; gradients flow into FE through both losses.
            # Using retain_graph=True on l_total would fail here because
            # opt_fe.step() modifies FE params in-place, invalidating the saved
            # graph.  Instead each component gets its own independent pass below.
            r      = fe(hist, nbrs, mask)
            pred_u = self.attr_cls(r)
            l_priv = self.ce_loss(pred_u, loc)
            l_jsd  = -jsd_MI(self.mi_est, x_flat, r, u_float, x_tilde)
            l_total = -self.lam * l_priv + (1.0 - self.lam) * l_jsd

            opt_fe.zero_grad()
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(fe.parameters(), 10)
            opt_fe.step()

            # ── Step 2: Update g_Ψ (attribute classifier) ────────────────────
            # FE is frozen (torch.no_grad) → only attr_cls parameters get
            # gradients.  g_Ψ minimises L_priv → becomes a better classifier.
            opt_cls.zero_grad()
            with torch.no_grad():
                r_cls = fe(hist, nbrs, mask)
            l_priv_cls = self.ce_loss(self.attr_cls(r_cls), loc)
            l_priv_cls.backward()
            torch.nn.utils.clip_grad_norm_(self.attr_cls.parameters(), 10)
            opt_cls.step()

            # ── Step 3: Update h_Ω (MI estimator) ────────────────────────────
            # FE is frozen; h_Ω minimises −jsd_MI → maximises MI →
            # becomes a better location discriminator (adversarial to FE).
            opt_mi.zero_grad()
            with torch.no_grad():
                r_mi = fe(hist, nbrs, mask)
            l_jsd_mi = -jsd_MI(self.mi_est, x_flat, r_mi, u_float, x_tilde)
            l_jsd_mi.backward()
            torch.nn.utils.clip_grad_norm_(self.mi_est.parameters(), 10)
            opt_mi.step()

            p_val = l_priv.item()
            j_val = l_jsd.item()
            total_loss_sum += l_total.item()
            priv_loss_sum  += p_val
            jsd_loss_sum   += j_val
            batch_count    += 1

            pbar.set_postfix(priv=f'{p_val:.3f}', jsd=f'{j_val:.3f}',
                             total=f'{l_total.item():.3f}')

            current_step = global_step_offset + batch_count
            wandb.log({
                f'client_{self.client_id}/total_loss': l_total.item(),
                f'client_{self.client_id}/priv_loss':  p_val,
                f'client_{self.client_id}/jsd_loss':   j_val,
                'round':       round_num + 1,
                'global_step': current_step,
            })

        # Free GPU memory before the next client starts
        gc.collect()
        torch.cuda.empty_cache()

        n = batch_count if batch_count > 0 else 1
        avg_total = total_loss_sum / n
        avg_priv  = priv_loss_sum  / n
        avg_jsd   = jsd_loss_sum   / n
        return fe.state_dict(), avg_total, avg_priv, avg_jsd, batch_count


# =============================================================================
# FL Checkpoint Loading  (highwayNet → HighwayNetFE + TrajectoryDecoder)
# =============================================================================

# Parameter name prefixes that belong to each split half of highwayNet.
# These must stay in sync with HighwayNetFE and TrajectoryDecoder.__init__.
_FE_PREFIXES  = {'ip_emb', 'enc_lstm', 'dyn_emb', 'soc_conv', 'conv_3x1'}
_DEC_PREFIXES = {'dec_lstm', 'op', 'op_lat', 'op_lon'}


def load_fl_checkpoint(path: str, device: torch.device):
    """
    Load a flat highwayNet state_dict (e.g. fl_global_round_8.tar) and split it
    into separate state_dicts for HighwayNetFE and TrajectoryDecoder.

    The split is done by the first segment of each parameter key (before the
    first '.'), which is the sub-module name in highwayNet and matches the
    attribute names used in HighwayNetFE / TrajectoryDecoder exactly.

    Args:
        path   : path to a .tar file saved by federated_trajectory_prediction_dp.py
        device : map_location for torch.load

    Returns:
        (fe_state_dict, decoder_state_dict)  — ready to pass to load_state_dict()
    """
    state = torch.load(path, map_location=device)
    # Unwrap any outer wrapper keys that some checkpoints use
    if 'state_dict' in state:
        state = state['state_dict']

    fe_state  = {k: v for k, v in state.items()
                 if k.split('.')[0] in _FE_PREFIXES}
    dec_state = {k: v for k, v in state.items()
                 if k.split('.')[0] in _DEC_PREFIXES}

    missing_fe  = _FE_PREFIXES  - {k.split('.')[0] for k in fe_state}
    missing_dec = _DEC_PREFIXES - {k.split('.')[0] for k in dec_state}
    if missing_fe:
        logger.warning('[load_fl_checkpoint] FE prefixes not found in checkpoint: %s', missing_fe)
    if missing_dec:
        logger.warning('[load_fl_checkpoint] Decoder prefixes not found in checkpoint: %s', missing_dec)

    logger.info('[load_fl_checkpoint] Loaded %d FE params, %d decoder params from %s',
                len(fe_state), len(dec_state), path)
    return fe_state, dec_state


# =============================================================================
# FedAvg Aggregation & Validation
# =============================================================================

def fed_avg(weights_list: list) -> dict:
    """
    Federated Averaging (McMahan et al., 2017) over a list of FE state dicts.

    Computes the element-wise arithmetic mean of all parameter tensors.
    Equal client weighting is used (uniform sample counts assumed).

    NOTE: Only FE weights are aggregated. Decoder and MI estimator weights
    are local and never included in this call.

    Args:
        weights_list : list of state_dicts from each client's trained FE

    Returns:
        averaged state_dict to load into the global FE
    """
    w_avg = copy.deepcopy(weights_list[0])
    for k in w_avg:
        for i in range(1, len(weights_list)):
            w_avg[k] = w_avg[k] + weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights_list))
    return w_avg


def validate_attr_cls(fe_model, attr_cls, val_loader, device):
    """
    Validation pass: attribute classification accuracy on the val set.

    Uses the global FE + client 0's attribute classifier (g_Ψ) as a proxy.
    Lower accuracy means the FE is producing more location-agnostic representations
    (better privacy). The random-chance baseline is 1/NUM_DS_IDS = 50%.

    Args:
        fe_model   : current global HighwayNetFE
        attr_cls   : AttributeClassifier from client 0
        val_loader : DataLoader over ngsimDatasetWithLocation(ValSet.mat)
        device     : torch.device

    Returns:
        (avg_ce_loss, accuracy) — scalar float each
    """
    fe_model.eval()
    attr_cls.eval()

    ce_loss  = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for hist, nbrs, mask, _, _, _, _, loc in val_loader:
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            loc  = loc.to(device)

            r      = fe_model(hist, nbrs, mask)
            logits = attr_cls(r)

            total_loss += ce_loss(logits, loc).item()
            correct    += (logits.argmax(1) == loc).sum().item()
            total      += loc.size(0)

    attr_cls.train()
    avg_loss = total_loss / max(total, 1) * val_loader.batch_size   # re-scale by batch
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# =============================================================================
# Centralised MI Estimator Pretraining
# =============================================================================

def pretrain_mi_estimator(mi_est, fe_model, train_loader,
                          pretrain_epochs: int, device: torch.device):
    """
    Pretrain a single MI estimator on the full training set with FE frozen.

    The MI estimator is trained as a discriminator: it MAXIMISES JSD MI between
    the FE representation z and the binary highway label.  Maximising JSD =
    minimising −JSD, so we call backward() on −l_jsd.

    Only the MI estimator parameters are updated; the FE is held in eval() mode
    throughout and its gradients are not computed (torch.no_grad on z).

    After this function returns, the caller should broadcast the state_dict to
    all clients so every client starts FL with a warm MI estimator.

    Args:
        mi_est         : TrajectoryMIEstimator to pretrain (updated in-place)
        fe_model       : HighwayNetFE with fixed weights (put in eval mode here)
        train_loader   : DataLoader over the full ngsimDatasetWithLocation
        pretrain_epochs: number of pretraining epochs
        device         : torch.device

    Returns:
        mi_est  (same object, updated in-place)
    """
    fe_model.eval()
    mi_est.train()

    opt_mi = torch.optim.Adam(mi_est.parameters(), lr=LR_INIT, weight_decay=1e-4)

    logger.info('--- Pretraining MI estimator (centralised, FE frozen) ---')

    for epoch in range(pretrain_epochs):
        total_jsd, n = 0.0, 0

        pbar = tqdm(train_loader,
                    desc=f'  MI-pretrain epoch {epoch+1}/{pretrain_epochs}',
                    leave=False, unit='batch')
        for hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, loc in pbar:
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            loc  = loc.to(device)

            # FE is frozen: compute z without tracking gradients
            with torch.no_grad():
                z = fe_model(hist, nbrs, mask)

            x_flat  = prepare_x_for_mi(hist)
            x_prime = make_x_prime(x_flat)

            l_jsd = jsd_MI(mi_est, x_flat, z, loc.float(), x_prime)

            # MI estimator MAXIMISES JSD → minimise −l_jsd
            opt_mi.zero_grad()
            (-l_jsd).backward()
            torch.nn.utils.clip_grad_norm_(mi_est.parameters(), 10)
            opt_mi.step()

            total_jsd += l_jsd.item()
            n         += 1
            pbar.set_postfix(jsd=f'{l_jsd.item():.4f}')

        logger.info('MI-pretrain Epoch %d/%d | avg JSD: %.4f',
                    epoch + 1, pretrain_epochs, total_jsd / max(n, 1))
        wandb.log({
            'mi_pretrain/avg_jsd': total_jsd / max(n, 1),
            'mi_pretrain_epoch':   epoch + 1,
        })

    logger.info('MI estimator pretraining complete.')
    return mi_est


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_privacy_leakage(fe_model, test_loader, device,
                              num_classes=NUM_DS_IDS, num_epochs=10):
    """
    Measure privacy leakage via linear probing.

    A single linear layer is trained on top of frozen FE representations to
    predict dsId. This is the standard evaluation protocol for representation
    privacy: if location is linearly decodable from z, the FE has leaked it.

    Interpretation:
        accuracy ≈ 0.5 (50%)  →  z is location-agnostic (ideal post-TAPPFL)
        accuracy >> 50%       →  z carries location information (leakage)

    Run this function on the same checkpoint before and after TAPPFL training
    to measure how much the privacy regularisation reduced leakage.

    Args:
        fe_model    : trained HighwayNetFE (weights frozen during probing)
        test_loader : DataLoader over test set
        device      : torch.device
        num_classes : number of location classes (6)
        num_epochs  : epochs to train the linear probe

    Returns:
        accuracy : float  fraction of correctly classified dsId labels
    """
    fe_model.eval()

    # Collect all (z, dsId) pairs from the test set in one pass
    zs, labels = [], []
    with torch.no_grad():
        for hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, loc in test_loader:
            z = fe_model(hist.to(device), nbrs.to(device), mask.to(device))
            zs.append(z.cpu())
            labels.append(loc.cpu())

    Z = torch.cat(zs,     dim=0)  # (N, FE_OUT_DIM)
    L = torch.cat(labels, dim=0)  # (N,)

    # Train the linear probe on frozen representations
    probe  = nn.Linear(Z.shape[1], num_classes).to(device)
    opt    = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()
    ds     = torch.utils.data.TensorDataset(Z, L)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    for _ in tqdm(range(num_epochs), desc='  Privacy probe', leave=False):
        for z_b, l_b in loader:
            z_b, l_b = z_b.to(device), l_b.to(device)
            loss = crit(probe(z_b), l_b)
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate probe accuracy on the same data (linear probe standard protocol)
    correct = 0
    with torch.no_grad():
        for z_b, l_b in loader:
            z_b, l_b = z_b.to(device), l_b.to(device)
            correct += (probe(z_b).argmax(1) == l_b).sum().item()

    accuracy = correct / len(ds)
    logger.info(f'Privacy leakage (linear probe accuracy on highway label): {accuracy*100:.2f}%')
    logger.info(f'Random baseline: {100/num_classes:.2f}%  (ideal ≈ 50% for binary)')
    return accuracy


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='TAPPFL: privacy-preserving federated trajectory prediction'
    )
    parser.add_argument('--train-data',      default=TRAIN_DATA_PATH,
                        help='Path to TrainSet.mat')
    parser.add_argument('--val-data',        default=VAL_DATA_PATH,
                        help='Path to ValSet.mat')
    parser.add_argument('--test-data',       default=TEST_DATA_PATH,
                        help='Path to TestSet.mat')
    parser.add_argument('--checkpoint-dir',  default=CHECKPOINT_DIR,
                        help='Directory to save per-round and final checkpoints')
    parser.add_argument('--rounds',          type=int,   default=GLOBAL_ROUNDS,
                        help='Number of federated communication rounds')
    parser.add_argument('--pretrain-rounds', type=int,   default=PRETRAIN_ROUNDS,
                        help='Rounds to use MSE before switching to NLL')
    parser.add_argument('--lam',             type=float, default=TRADEOFF_LAMBDA,
                        help='λ trade-off: 1.0 = pure task, 0.0 = pure privacy')
    parser.add_argument('--batch-size',      type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',              type=float, default=LR_INIT)
    parser.add_argument('--reuse-weights',   action='store_true',
                        help='Load pretrained FE from --pretrain-ckpt instead of training from scratch')
    parser.add_argument('--pretrain-ckpt',   default=None,
                        help='Path to a TAPPFL FE-only checkpoint (.tar) used when --reuse-weights is set')
    # ── New options ───────────────────────────────────────────────────────────
    parser.add_argument('--fl-ckpt',         default=None,
                        help='Path to a flat highwayNet checkpoint (e.g. fl_global_round_8.tar). '
                             'FE and decoder weights are extracted and used to warm-start TAPPFL. '
                             'Takes priority over --reuse-weights/--pretrain-ckpt.')
    parser.add_argument('--mi-pretrain-epochs', type=int, default=0,
                        help='Epochs of centralised MI estimator pretraining before FL rounds. '
                             'FE is frozen; one shared MI estimator is trained then broadcast to all '
                             'clients. Set to 0 to skip (default).')
    return parser.parse_args()


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args: argparse.Namespace):
    """
    End-to-end TAPPFL training and evaluation.

    Steps
    -----
    1.  WandB initialisation
    2.  Load TrainSet and ValSet
    3.  Partition training samples IID across 10 clients
    4.  Initialise global FE + Decoder (random, or load from checkpoint)
    5.  Distribute initial decoder weights to all clients
    6.  Federated training rounds (first PRETRAIN_ROUNDS use MSE as federated warm-up):
            for each round:
                a. Each client runs one epoch of local TAPPFL training
                b. Server collects updated FE state_dicts
                c. FedAvg → new global FE
                d. Validate global FE with client 0's decoder
                e. Save per-round FE checkpoint
    7.  Load TestSet and evaluate RMSE
    8.  Evaluate privacy leakage via linear probe
    9.  Save final model checkpoints

    Returns:
        (global_fe, trained_clients)  for interactive inspection after training
    """
    logger.info(f'Device: {DEVICE}  |  FE output dim: {FE_OUT_DIM}')
    logger.info(f'Rounds: {args.rounds}  |  λ: {args.lam}  |  batch: {args.batch_size}')

    # ── WandB (no-op if not installed) ────────────────────────────────────────
    if _WANDB:
        wandb.init(
            project='tappfl-trajectory-prediction',
            reinit=True,
            config={
                'num_clients':     NUM_CLIENTS,
                'global_rounds':   args.rounds,
                'pretrain_rounds': args.pretrain_rounds,
                'batch_size':      args.batch_size,
                'lambda':          args.lam,
                'fe_out_dim':      FE_OUT_DIM,
                'num_ds_ids':      NUM_DS_IDS,
                'private_attr':    'dsId (highway segment)',
            }
        )

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info('Loading datasets...')
    full_train = ngsimDatasetWithLocation(args.train_data)
    full_val   = ngsimDatasetWithLocation(args.val_data)

    n_train = len(full_train)
    n_val   = len(full_val)
    val_sub = Subset(full_val, list(range(n_val)))

    val_loader = DataLoader(
        val_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=full_val.collate_fn,
    )

    # ── Partition training data across clients (IID) ──────────────────────────
    # Uniformly shuffle indices and split into NUM_CLIENTS equal shards.
    logger.info('Partitioning data IID across clients...')
    all_indices = list(range(n_train))
    np.random.shuffle(all_indices)
    split_size = n_train // NUM_CLIENTS
    client_indices = {
        i: all_indices[i * split_size:(i + 1) * split_size]
        for i in range(NUM_CLIENTS)
    }

    for i, idxs in client_indices.items():
        logger.info(f'  Client {i}: {len(idxs)} samples')

    # ── Initialise global FE ──────────────────────────────────────────────────
    global_fe = HighwayNetFE(ARGS).to(DEVICE)

    # Priority:  --fl-ckpt  >  --reuse-weights/--pretrain-ckpt  >  random init
    if getattr(args, 'fl_ckpt', None):
        # Load FE weights from a flat highwayNet checkpoint (decoder weights ignored)
        fe_state, _ = load_fl_checkpoint(args.fl_ckpt, DEVICE)
        global_fe.load_state_dict(fe_state, strict=True)
        logger.info('Warm-started FE from highwayNet checkpoint: %s', args.fl_ckpt)
    elif args.reuse_weights and args.pretrain_ckpt:
        # Load FE only from a TAPPFL FE-only checkpoint
        state = torch.load(args.pretrain_ckpt, map_location=DEVICE)
        global_fe.load_state_dict(state.get('fe', state))
        logger.info('Loaded pretrained FE weights from %s', args.pretrain_ckpt)
    else:
        logger.info('Starting from random FE initialisation.')

    # ── Build clients ─────────────────────────────────────────────────────────
    clients = [
        TAPPFLClient(
            client_id      = i,
            dataset_subset = Subset(full_train, client_indices[i]),
            device         = DEVICE,
            args           = ARGS,
            lam            = args.lam,
        )
        for i in range(NUM_CLIENTS)
    ]

    # ── MI estimator pretraining (optional) ───────────────────────────────────
    if getattr(args, 'mi_pretrain_epochs', 0) > 0:
        logger.info('Running centralised MI estimator pretraining (%d epochs)...',
                    args.mi_pretrain_epochs)
        # Build a central loader over the full training set
        central_loader = DataLoader(
            Subset(full_train, list(range(n_train))),
            batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=full_train.collate_fn,
        )
        # Train a single shared MI estimator with FE frozen
        shared_mi_est = TrajectoryMIEstimator(
            fe_out_dim=FE_OUT_DIM,
            num_private_classes=NUM_DS_IDS,
            hist_dim=HIST_DIM,
        ).to(DEVICE)
        pretrain_mi_estimator(shared_mi_est, global_fe, central_loader,
                               args.mi_pretrain_epochs, DEVICE)
        # Broadcast pretrained MI estimator weights to all clients
        for client in clients:
            client.mi_est.load_state_dict(shared_mi_est.state_dict())
        del shared_mi_est, central_loader
        gc.collect()
        torch.cuda.empty_cache()
        logger.info('Distributed pretrained MI estimator weights to all %d clients.', NUM_CLIENTS)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_fe_weights   = global_fe.state_dict()
    global_step_tracker = 0
    best_val_loss       = float('inf')
    best_round          = 0

    # ── Federated training rounds ─────────────────────────────────────────────
    for round_num in range(args.rounds):
        t_round_start = time.time()
        logger.info(f'=== Round {round_num+1}/{args.rounds} ===')

        local_fe_weights = []
        local_totals     = []
        local_privs      = []
        local_jsds       = []
        max_steps        = 0

        # Sequential client training (conceptually parallel in real FL)
        for client in clients:
            w_fe, avg_total, avg_priv, avg_jsd, steps = client.train(
                global_fe_weights, round_num, global_step_tracker
            )
            local_fe_weights.append(w_fe)
            local_totals.append(avg_total)
            local_privs.append(avg_priv)
            local_jsds.append(avg_jsd)
            max_steps = max(max_steps, steps)
            logger.info(f'  Client {client.client_id} | '
                        f'total={avg_total:.4f}  priv={avg_priv:.4f}  jsd={avg_jsd:.4f} '
                        f'({steps} batches)')

        # Advance the WandB step counter by the longest client's epoch length
        global_step_tracker += max_steps

        # FedAvg — only the FE weights are aggregated
        global_fe_weights = fed_avg(local_fe_weights)
        global_fe.load_state_dict(global_fe_weights)

        # Validate: attribute classification accuracy on val set
        # Lower accuracy = more location-agnostic representations (better privacy)
        val_ce, val_acc    = validate_attr_cls(global_fe, clients[0].attr_cls,
                                               val_loader, DEVICE)
        avg_train_loss = sum(local_totals) / len(local_totals)
        avg_priv_loss  = sum(local_privs)  / len(local_privs)
        avg_jsd_loss   = sum(local_jsds)   / len(local_jsds)
        elapsed        = time.time() - t_round_start

        # Best = lowest val accuracy (most location-agnostic FE)
        marker = ''
        if val_acc < best_val_loss:
            best_val_loss = val_acc
            best_round    = round_num + 1
            marker        = '  *best*'
            torch.save(global_fe_weights,
                       os.path.join(args.checkpoint_dir, 'tappfl_fe_best.tar'))

        logger.info(
            f'Round {round_num+1} | val_acc={val_acc:.3f} val_ce={val_ce:.4f} | '
            f'train={avg_train_loss:.4f} (priv={avg_priv_loss:.4f} jsd={avg_jsd_loss:.4f}) | '
            f'{elapsed:.1f}s{marker}'
        )

        wandb.log({
            'global/val_attr_acc':   val_acc,
            'global/val_ce':         val_ce,
            'global/avg_train_loss': avg_train_loss,
            'global/avg_priv_loss':  avg_priv_loss,
            'global/avg_jsd_loss':   avg_jsd_loss,
            'round':                 round_num + 1,
            'global_step':           global_step_tracker,
        })

        # Save per-round FE checkpoint for post-hoc analysis
        torch.save(
            global_fe_weights,
            os.path.join(args.checkpoint_dir, f'tappfl_fe_round_{round_num+1}.tar'),
        )

    logger.info(f'Training complete. Best val attr_acc: {best_val_loss:.4f} at round {best_round}.')
    if _WANDB:
        wandb.finish()

    return global_fe, clients, args.checkpoint_dir


# =============================================================================
# Evaluation Runner & Model Saving
# =============================================================================

def run_evaluation_and_save(global_fe, trained_clients, checkpoint_dir: str = CHECKPOINT_DIR):
    """
    Post-training evaluation and checkpoint saving.

    Separated from main() so it can be called independently after training,
    or after loading a checkpoint for analysis.

    Args:
        global_fe       : trained HighwayNetFE
        trained_clients : list of TAPPFLClient (one per segment)
    """
    # ── Test set evaluation ───────────────────────────────────────────────────
    test_dataset = ngsimDatasetWithLocation(TEST_DATA_PATH)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=test_dataset.collate_fn,
    )

    logger.info('\n── Privacy Leakage (Linear Probe on dsId) ──')
    evaluate_privacy_leakage(global_fe, test_loader, DEVICE)

    # ── Save final checkpoints ────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        global_fe.state_dict(),
        os.path.join(checkpoint_dir, 'tappfl_fe_final.tar'),
    )
    for i, client in enumerate(trained_clients):
        torch.save(
            client.attr_cls.state_dict(),
            os.path.join(checkpoint_dir, f'tappfl_attr_cls_client{i}_final.tar'),
        )

    logger.info(f'\nAll models saved to {checkpoint_dir}/')
    logger.info(f'  tappfl_fe_final.tar                    — Global Feature Extractor')
    logger.info(f'  tappfl_attr_cls_client{{i}}_final.tar   — Per-client Attribute Classifiers')


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    _args = parse_args()
    global_fe, trained_clients, ckpt_dir = main(_args)
    run_evaluation_and_save(global_fe, trained_clients, ckpt_dir)
