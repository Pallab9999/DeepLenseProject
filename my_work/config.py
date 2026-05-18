"""
config.py
=========
Central configuration for the DeepLense 4-class physics-informed classifier.
Every hyperparameter lives here — import this in every other module.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent   # DeepLenseProject/
DATA_DIR   = ROOT_DIR / "Data"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PLOTS_DIR       = RESULTS_DIR / "plots"
LOGS_DIR        = RESULTS_DIR / "logs"

# Create result dirs if missing
for d in [CHECKPOINTS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MODEL_I_TRAIN   = DATA_DIR / "Model_I"  / "train"
MODEL_I_TEST    = DATA_DIR / "Model_I"  / "test"
MODEL_II_TRAIN  = DATA_DIR / "Model_II" / "train"
MODEL_II_TEST   = DATA_DIR / "Model_II" / "test"
MODEL_III_TRAIN = DATA_DIR / "Model_III"/ "train"
MODEL_III_TEST  = DATA_DIR / "Model_III"/ "test"

# Class names for all four dark-matter types.
# Model I-III only contain the first three; vortex will be added when data
# becomes available (or via simulation with lenstronomy).
CLASS_NAMES = ["no_substructure", "cdm_subhalos", "axion", "vortex"]
NUM_CLASSES = 4         # Extension to 4 classes (Contribution 1)
IMAGE_SIZE  = 64        # All DeepLense images are 64×64 pixels
IN_CHANNELS = 1         # Single-channel greyscale

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
MIN_ANGLE     = -3.232  # arcsec — field-of-view boundary
MAX_ANGLE     =  3.232
PIXEL_SCALE   = 0.101   # arcsec / pixel (HSC-like)
THETA_E_MIN   =  0.0    # Einstein radius constraint (arcsec)
THETA_E_MAX   =  2.0

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# ViT backbone (timm model name).  vit_small_patch16_224 maps to 384-dim.
VIT_MODEL_NAME = "vit_small_patch16_224"
PATCH_SIZE     = 16
EMBED_DIM      = 384

# CNN decoder hidden width
CNN_HIDDEN     = 512

# ---------------------------------------------------------------------------
# Training — Phase 1 (supervised, simulated data)
# ---------------------------------------------------------------------------
BATCH_SIZE          = 32
NUM_WORKERS         = 4
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
EPOCHS              = 100
COSINE_T_MAX        = 100
LAMBDA_PHYSICS      = 0.1   # weight on physics consistency loss
DROPOUT             = 0.3

# ---------------------------------------------------------------------------
# Training — Phase 2 (ADDA domain adaptation)
# ---------------------------------------------------------------------------
ADDA_LR_TARGET      = 1e-5
ADDA_LR_DISCRIMINATOR = 1e-4
ADDA_EPOCHS         = 50
ADDA_PATIENCE       = 15

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED        = 42
DEVICE      = "cuda"   # will fall back to "cpu" automatically in train.py

WANDB_PROJECT = "deeplense-gsoc2026"
WANDB_ENABLED = False  # set True when you have a W&B account
