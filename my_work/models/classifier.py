"""
models/classifier.py
====================
Stand-alone 4-class classification head that wraps ANY encoder backbone.

Used for:
  1. Baseline-only training (pure CNN or ViT, no physics)
  2. Feature extraction evaluation
  3. The ADDA classifier that stays frozen during domain adaptation

The module is intentionally backend-agnostic: pass any encoder that returns
a flat (B, feature_dim) vector from forward().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Generic backbone + head
# ---------------------------------------------------------------------------

class DeepLenseClassifier(nn.Module):
    """
    Backbone encoder + classification head.

    Parameters
    ----------
    backbone_name : timm model name (used when no custom encoder is supplied)
    num_classes   : number of output classes
    feature_dim   : width of the hidden layer in the MLP head
    dropout       : dropout rate
    pretrained    : load ImageNet weights for timm backbone
    in_chans      : number of input image channels (1 for greyscale)
    custom_encoder : optional pre-built encoder module (overrides backbone_name)
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        num_classes: int = 4,
        feature_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = False,
        in_chans: int = 1,
        custom_encoder: nn.Module | None = None,
    ):
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
            # Probe feature dimension by a dry-run forward
            was_training = custom_encoder.training
            custom_encoder.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, in_chans, 64, 64)
                try:
                    out = custom_encoder(dummy)
                    if isinstance(out, (tuple, list)):
                        out = out[-1]
                    enc_dim = out.shape[-1]
                except Exception:
                    raise RuntimeError(
                        "Could not infer custom encoder output dimension. "
                        "Ensure forward(x) returns a (B, F) tensor or a tuple "
                        "whose last element is (B, F)."
                    )
            if was_training:
                custom_encoder.train()
        else:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,      # strip the classification head
                in_chans=in_chans,
            )
            enc_dim = self.encoder.num_features

        self.head = nn.Sequential(
            nn.Linear(enc_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        Returns: logits (B, num_classes)
        """
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return self.head(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the encoder features (B, enc_dim) — used by ADDA."""
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return features


# ---------------------------------------------------------------------------
# Lightweight ResNet-18 baseline (for the ablation study)
# ---------------------------------------------------------------------------

class ResNetBaseline(nn.Module):
    """
    Plain ResNet-18 baseline (no physics) for comparison in the ablation study.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
        )
        enc_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = DeepLenseClassifier(num_classes=4).to(device)
    dummy = torch.rand(4, 1, 64, 64, device=device)
    out = clf(dummy)
    print(f"DeepLenseClassifier output: {out.shape}")

    baseline = ResNetBaseline(num_classes=4).to(device)
    out2 = baseline(dummy)
    print(f"ResNetBaseline output      : {out2.shape}")
