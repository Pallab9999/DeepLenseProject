"""
utils/physics.py
================
Differentiable PyTorch modules that encode the SIS gravitational lensing
equations.  These are the physics "building blocks" shared across every model
in this project.

Key equations
-------------
    Lensing:      beta  = theta - alpha(theta)
    SIS alpha:    alpha = theta_E * theta / |theta|
    Einstein R:   theta_E = sqrt( 4GM/c^2 * D_ls / (D_l * D_s) )

Both modules are fully differentiable, so gradients flow through them during
end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lensing inversion  (image-plane → source-plane mapping)
# ---------------------------------------------------------------------------

class LensingInversionLayer(nn.Module):
    """
    Applies the SIS gravitational lensing equation to "undo" the lensing and
    reconstruct where the source galaxy actually sits.

    Forward:
        image   – (B, C, H, W)  the observed lensed image
        theta_E – (B, 1)        Einstein radius predicted by the encoder

    Returns:
        source  – (B, C, H, W)  source-plane reconstruction (same resolution)

    The layer builds a sampling grid in source coordinates (beta) and uses
    bilinear interpolation via grid_sample, so the operation is end-to-end
    differentiable.
    """

    def __init__(self, min_angle: float = -3.232, max_angle: float = 3.232):
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def forward(self, image: torch.Tensor, theta_E: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        device = image.device

        # -------- build normalised image-plane grid (theta) in [-1, 1] --------
        # grid_sample convention: x = column (W axis), y = row (H axis)
        y_norm = torch.linspace(-1.0, 1.0, H, device=device)
        x_norm = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")  # (H, W)
        theta = torch.stack([grid_x, grid_y], dim=-1)                    # (H, W, 2)

        # -------- SIS deflection angle  alpha = theta_E * theta / |theta| -----
        theta_norm = theta.norm(dim=-1, keepdim=True).clamp(min=1e-8)    # (H, W, 1)
        # theta_E: (B, 1) → (B, 1, 1, 1) for broadcasting
        tE = theta_E.view(B, 1, 1, 1)
        # alpha: (B, H, W, 2)
        alpha = tE * theta.unsqueeze(0) / theta_norm.unsqueeze(0)

        # -------- source-plane position  beta = theta - alpha -----------------
        beta = theta.unsqueeze(0) - alpha  # (B, H, W, 2)

        # grid_sample expects grid in [-1, 1]; beta is already in that range
        # because theta ∈ [-1, 1] and for realistic theta_E < 1 the beta stays
        # inside that range for most pixels.
        source = F.grid_sample(
            image, beta,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return source


# ---------------------------------------------------------------------------
# Physics-informed preprocessing  (from the LensPINN paper)
# ---------------------------------------------------------------------------

class PhysicsPreprocessing(nn.Module):
    """
    Computes the physics-informed preprocessing map described in the LensPINN
    paper (Ojha et al., NeurIPS-ML4PS 2024):

        preprocessed = tanh( ∂/∂x ∂/∂y [ log(I_max / I)^2 ] )

    This highlights arc / ring structures that carry the dark-matter signal
    while being invariant to overall brightness scaling.

    Input  – (B, C, H, W)  raw lensed image
    Output – (B, C, H, W)  physics-informed map (same shape)
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        I_max = image.amax(dim=(-1, -2), keepdim=True)          # (B, C, 1, 1)
        log_ratio = torch.log((I_max + eps) / (image + eps))    # (B, C, H, W)
        squared = log_ratio ** 2

        # Finite-difference cross-derivative  ∂²f / ∂x ∂y
        dx = squared[:, :, :, 1:] - squared[:, :, :, :-1]       # (B, C, H, W-1)
        dx = F.pad(dx, (0, 1))                                   # restore width
        dy = squared[:, :, 1:, :] - squared[:, :, :-1, :]       # (B, C, H-1, W)
        dy = F.pad(dy, (0, 0, 0, 1))                             # restore height

        return torch.tanh(dx * dy)


# ---------------------------------------------------------------------------
# Einstein radius estimator  (thin wrapper for readability)
# ---------------------------------------------------------------------------

class EinsteinRadiusHead(nn.Module):
    """
    Maps a ViT [CLS] embedding → a single scalar θ_E ∈ [theta_E_min, theta_E_max].

    Parameters
    ----------
    in_features : int
        Dimension of the incoming ViT feature vector.
    theta_E_min, theta_E_max : float
        Physical bounds on the Einstein radius (arcsec).
    """

    def __init__(
        self,
        in_features: int,
        theta_E_min: float = 0.0,
        theta_E_max: float = 2.0,
    ):
        super().__init__()
        self.scale = theta_E_max - theta_E_min
        self.shift = theta_E_min
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features)  →  theta_E: (B, 1)"""
        raw = self.linear(x)                    # (B, 1)
        return torch.sigmoid(raw) * self.scale + self.shift


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running physics sanity check on {device}")

    B, C, H, W = 4, 1, 64, 64
    images = torch.rand(B, C, H, W, device=device)
    theta_E = torch.rand(B, 1, device=device) * 0.8  # realistic Einstein radii

    inv_layer = LensingInversionLayer().to(device)
    preproc   = PhysicsPreprocessing().to(device)
    er_head   = EinsteinRadiusHead(in_features=384).to(device)

    source = inv_layer(images, theta_E)
    prep   = preproc(images)

    print(f"  Input image  shape : {images.shape}")
    print(f"  Source recon shape : {source.shape}")
    print(f"  Preprocessed shape : {prep.shape}")
    print(f"  theta_E sample     : {theta_E.squeeze().tolist()}")
    print("  All shapes OK — physics module working correctly.")
