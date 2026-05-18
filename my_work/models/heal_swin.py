"""
models/heal_swin.py
===================
HEAL-Swin — Swin Transformer backbone with HEALPix-aware preprocessing.

Based on the HEAL-PINN paper (Srivastava et al., NeurIPS-ML4PS 2025).
Contribution 2 of this project: combine the physics encoder from LensPINN
with the more efficient Swin backbone for large-scale training.

Key ideas
─────────
  • Swin Transformer uses shifted-window attention → O(n) complexity vs O(n²)
    for plain ViT, allowing larger images and bigger batches.
  • HEALPix correction accounts for spherical sky projections.
    (In this simplified version we approximate it with a radial weight mask.)
  • The physics encoder still predicts θ_E and applies lensing inversion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils.physics import LensingInversionLayer, EinsteinRadiusHead


# ---------------------------------------------------------------------------
# Approximate HEALPix radial correction
# ---------------------------------------------------------------------------

class HEALPixCorrection(nn.Module):
    """
    Applies a learnable radial weight mask that approximates the area
    distortion introduced by projecting a spherical sky onto a flat pixel
    grid.  Pixels at larger angular radius are up-weighted to compensate for
    the cosine foreshortening effect.

    The weights are initialised to 1 (no correction) and learned end-to-end.
    """

    def __init__(self, image_size: int = 64):
        super().__init__()
        r = _radial_distance_grid(image_size)           # (1, 1, H, W), values in [0, 1]
        # Learnable scale and offset
        self.register_buffer("r_grid", r)
        self.alpha = nn.Parameter(torch.ones(1))        # amplitude of correction
        self.beta  = nn.Parameter(torch.zeros(1))       # bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Radial correction weight
        w = 1.0 + self.alpha * self.r_grid + self.beta  # (1, 1, H, W)
        return x * w


def _radial_distance_grid(size: int) -> torch.Tensor:
    """Returns a (1, 1, size, size) tensor of normalised radial distances."""
    coords = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2).clamp(max=1.0)
    return r.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


# ---------------------------------------------------------------------------
# HEAL-Swin model
# ---------------------------------------------------------------------------

class HEALSwin(nn.Module):
    """
    Physics-informed Swin Transformer for 4-class dark matter classification.

    Architecture
    ─────────────
    HEALPix correction → physics preprocessing → Swin encoder
                                                    ↓ predicts θ_E
                                               Lensing inversion
                                                    ↓
                                              [source, residual] → MLP head

    Parameters
    ----------
    num_classes   : number of output classes (4)
    swin_name     : timm Swin model name
    image_size    : spatial resolution (64 for Model I-III)
    dropout       : dropout before the final head
    theta_E_max   : physical upper bound on Einstein radius
    """

    def __init__(
        self,
        num_classes: int = 4,
        swin_name: str = "swin_tiny_patch4_window7_224",
        image_size: int = 64,
        dropout: float = 0.3,
        theta_E_max: float = 2.0,
    ):
        super().__init__()

        self.image_size = image_size

        # ── HEALPix correction ────────────────────────────────────────────
        self.healcorr = HEALPixCorrection(image_size)

        # ── Swin Transformer backbone ─────────────────────────────────────
        # Swin expects 3-channel input; we use a 1→3 stem conv
        self.stem = nn.Conv2d(1, 3, kernel_size=1)   # channel adapter

        self.swin = timm.create_model(
            swin_name,
            pretrained=True,
            num_classes=0,   # strip head
            in_chans=3,
        )
        swin_dim = self.swin.num_features

        # ── Einstein radius head ──────────────────────────────────────────
        self.er_head = EinsteinRadiusHead(
            in_features=swin_dim,
            theta_E_min=0.0,
            theta_E_max=theta_E_max,
        )

        # ── Lensing inversion ─────────────────────────────────────────────
        self.lensing_inv = LensingInversionLayer()

        # ── Classification head ───────────────────────────────────────────
        # Feature fusion: source features + residual features both come from
        # a small linear projection on each Swin feature vector.
        self.proj_source   = nn.Linear(swin_dim, 256)
        self.proj_residual = nn.Linear(swin_dim, 256)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    def _swin_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run 1-ch image through stem + Swin → (B, swin_dim) features."""
        x224 = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x3   = self.stem(x224)   # (B, 3, 224, 224)
        return self.swin(x3)     # (B, swin_dim)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, 1, H, W)  raw image

        Returns
        -------
        theta_E : (B, 1)
        source  : (B, 1, H, W)
        logits  : (B, num_classes)
        """
        # HEALPix spherical correction
        x_corr = self.healcorr(x)

        # Swin encode → predict θ_E
        swin_feats = self._swin_encode(x_corr)
        theta_E    = self.er_head(swin_feats)

        # Lensing inversion
        source   = self.lensing_inv(x, theta_E)         # (B, 1, H, W)
        residual = x - source

        # Encode both branches separately (re-use Swin — could also be a CNN)
        feat_source   = self.proj_source(self._swin_encode(source))    # (B, 256)
        feat_residual = self.proj_residual(self._swin_encode(residual)) # (B, 256)

        features = torch.cat([feat_source, feat_residual], dim=1)       # (B, 512)
        logits   = self.head(features)

        return theta_E, source, logits


# ---------------------------------------------------------------------------
# Quick architecture test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing HEALSwin on {device}")

    model = HEALSwin(num_classes=4).to(device)
    model.eval()

    dummy = torch.rand(2, 1, 64, 64, device=device)
    with torch.no_grad():
        theta_E, source, logits = model(dummy)

    print(f"  Input   : {dummy.shape}")
    print(f"  θ_E     : {theta_E.shape}")
    print(f"  Source  : {source.shape}")
    print(f"  Logits  : {logits.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params  : {n:,}")
    print("  HEALSwin architecture OK.")
