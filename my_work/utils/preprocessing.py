"""
utils/preprocessing.py
=======================
Physics-informed and domain-specific preprocessing utilities.

Includes:
  - PhysicsPreprocessing  (already in utils/physics.py — re-exported here
    for convenience so notebooks only need one import)
  - compute_distortion_map  (the tanh-log map used in LensPINN as a *third*
    input channel to the decoder)
  - normalise_batch  (zero-mean / unit-variance per sample)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Re-export for convenience
from utils.physics import PhysicsPreprocessing  # noqa: F401


# ---------------------------------------------------------------------------
# Distortion map  (additional input to decoder, per LensPINN paper)
# ---------------------------------------------------------------------------

def compute_distortion_map(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the distortion proxy map used as a third encoder input in LensPINN:

        D(x) = tanh( ∂/∂x ∂/∂y [ log(I_max / I) ]^2 )

    Parameters
    ----------
    image : Tensor (B, C, H, W)
    eps   : small constant for numerical stability

    Returns
    -------
    Tensor (B, C, H, W)  values in (-1, 1)
    """
    I_max = image.amax(dim=(-1, -2), keepdim=True)
    log_ratio = torch.log((I_max + eps) / (image + eps))
    squared = log_ratio ** 2

    dx = squared[:, :, :, 1:] - squared[:, :, :, :-1]
    dx = F.pad(dx, (0, 1))
    dy = squared[:, :, 1:, :] - squared[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))

    return torch.tanh(dx * dy)


# ---------------------------------------------------------------------------
# Per-sample normalisation
# ---------------------------------------------------------------------------

def normalise_batch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Zero-mean, unit-variance normalisation per sample (not per batch).
    Safe for single-channel images.

    Parameters
    ----------
    x : Tensor (B, C, H, W)

    Returns
    -------
    Tensor (B, C, H, W)
    """
    flat = x.view(x.size(0), -1)          # (B, C*H*W)
    mean = flat.mean(dim=1, keepdim=True)
    std  = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return ((flat - mean) / std).view_as(x)
