"""
train_adda.py
=============
Phase 2 training script — Unsupervised Domain Adaptation (ADDA) to transfer 
pre-trained physics-informed LensPINN from clean simulations to noisy 
telescope-like target observations.

Usage
-----
    cd my_work
    python train_adda.py --checkpoint results/checkpoints/lens_pinn_model_i_best.pt --epochs 10 --workers 0
"""

import argparse
import sys
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE, NUM_WORKERS, DEVICE, SEED, CLASS_NAMES, IMAGE_SIZE,
    MODEL_I_TRAIN, MODEL_I_TEST, CHECKPOINTS_DIR, PLOTS_DIR
)
from utils.data_loader import build_dataloaders, DeepLenseDataset
from models.lens_pinn import LensPINN
from models.adda import DomainDiscriminator, ADDATrainer


# ---------------------------------------------------------------------------
# LensPINN Encoder Wrapper
# ---------------------------------------------------------------------------

class LensPINNEncoder(nn.Module):
    """
    Extracts the feature representation (penultimate layer) from LensPINN.
    This corresponds to the concatenated 1024-dimensional feature vector 
    from the source and residual branches, right before the classifier head.
    """

    def __init__(self, full_model: LensPINN):
        super().__init__()
        self.vit = full_model.vit
        self.er_head = full_model.er_head
        self.lensing_inv = full_model.lensing_inv
        self.cnn_source = full_model.cnn_source
        self.cnn_residual = full_model.cnn_residual

    def _resize_to_64(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.interpolate(x, size=224, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_224 = self._resize_to_64(x)
        vit_features = self.vit(x_224)
        theta_E = self.er_head(vit_features)
        
        source = self.lensing_inv(x, theta_E)
        feat_source = self.cnn_source(source)
        feat_residual = self.cnn_residual(x - source)
        
        return torch.cat([feat_source, feat_residual], dim=1)  # (B, 1024)


# ---------------------------------------------------------------------------
# Target Domain Distortion Transforms
# ---------------------------------------------------------------------------

def get_target_transforms(image_size: int):
    """
    Creates severe domain-shift transformations mimicking a real observatory:
      - Severe atmospheric turbulence (Gaussian Blur)
      - Optical distortion and sharpness loss
      - Sensor noise and intensity bias
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 3.0)),   # Atmospheric blur
        transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=1.0), # Optical distortion
        transforms.Normalize(mean=[0.1], std=[0.2]),                  # Systematic intensity bias
    ])


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepLense ADDA Domain Adaptation")
    p.add_argument(
        "--checkpoint", type=str, 
        default="results/checkpoints/lens_pinn_model_i_best.pt",
        help="Path to pre-trained LensPINN checkpoint"
    )
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr_target",    type=float, default=1e-5)
    p.add_argument("--lr_disc",      type=float, default=1e-4)
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--limit_batches", type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    device_str = DEVICE if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  Phase 2: ADDA Unsupervised Domain Adaptation")
    print(f"  Source Checkpoint : {args.checkpoint}")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── 1. Load Pre-trained Source Model ──────────────────────────────────
    print("[1/4] Loading pre-trained LensPINN source model...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        print("Please train your LensPINN model first using train.py.")
        sys.exit(1)

    full_source_model = LensPINN(num_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_source_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Extract source encoder and classifier head
    source_encoder = LensPINNEncoder(full_source_model).to(device)
    classifier = full_source_model.classifier.to(device)

    # Clone source encoder to become the target encoder
    target_encoder = copy.deepcopy(source_encoder).to(device)
    
    # Instantiate the domain discriminator
    discriminator = DomainDiscriminator(feature_dim=1024).to(device)

    # ── 2. Build Source and Target Dataloaders ───────────────────────────
    print("[2/4] Setting up domain-shifted dataloaders...")
    
    # Source Dataloader (Clean Simulations)
    source_loader, _, _ = build_dataloaders(
        train_root=MODEL_I_TRAIN,
        test_root=MODEL_I_TEST,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Target Dataloaders (Simulated Telescope Observations - severely distorted)
    target_ds_train = DeepLenseDataset(
        root=MODEL_I_TRAIN,
        class_names=CLASS_NAMES,
        transform=get_target_transforms(IMAGE_SIZE),
    )
    target_ds_val = DeepLenseDataset(
        root=MODEL_I_TEST,
        class_names=CLASS_NAMES,
        transform=get_target_transforms(IMAGE_SIZE),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_loader_train = DataLoader(target_ds_train, shuffle=True, **loader_kwargs)
    target_loader_val   = DataLoader(target_ds_val,   shuffle=False, **loader_kwargs)

    print(f"  Source (simulated) train samples : {len(source_loader.dataset)}")
    print(f"  Target (distorted) train samples : {len(target_ds_train)}")
    print(f"  Target (distorted) val samples   : {len(target_ds_val)}")

    # ── 3. Initialize ADDA Trainer ────────────────────────────────────────
    print("\n[3/4] Initializing Adversarial Trainer...")
    trainer = ADDATrainer(
        source_encoder=source_encoder,
        target_encoder=target_encoder,
        classifier=classifier,
        discriminator=discriminator,
        device=device,
    )

    # ── 4. Execute Adaptation ─────────────────────────────────────────────
    print("\n[4/4] Starting Adversarial Adaptation...")
    t0 = time.time()
    
    trainer.adapt(
        source_loader=source_loader,
        target_loader=target_loader_train,
        target_loader_val=target_loader_val,
        epochs=args.epochs,
        lr_target=args.lr_target,
        lr_disc=args.lr_disc,
        patience=5,
        save_path=CHECKPOINTS_DIR / "adda_adapted_best.pt",
        limit_batches=args.limit_batches,
    )
    
    elapsed = time.time() - t0
    print(f"\nAdaptation finished in {elapsed:.1f}s.")

    # ── 5. Plot training history ─────────────────────────────────────────
    history_plot_path = str(PLOTS_DIR / "adda_training_curves.png")
    trainer.plot_history(save_path=history_plot_path)
    print(f"[Success] Domain adaptation curves saved to {history_plot_path}\n")


if __name__ == "__main__":
    main()
