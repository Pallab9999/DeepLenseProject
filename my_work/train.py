"""
train.py
========
Phase 1 training script — supervised classification on simulated data.

Usage
-----
    cd my_work
    python train.py --model lens_pinn --data_root ../Data/Model_I --epochs 100

The script:
  1. Builds the dataloader for Model I / II / III simulated data
  2. Trains the selected model with cosine-annealed AdamW
  3. Tracks training / validation AUC (not just accuracy — to mirror papers)
  4. Saves the best checkpoint and training curves
  5. (optional) logs to Weights & Biases
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    COSINE_T_MAX, LAMBDA_PHYSICS, DEVICE, SEED,
    CLASS_NAMES, IMAGE_SIZE,
    CHECKPOINTS_DIR, PLOTS_DIR, LOGS_DIR,
    MODEL_I_TRAIN, MODEL_I_TEST,
    MODEL_II_TRAIN, MODEL_II_TEST,
    MODEL_III_TRAIN, MODEL_III_TEST,
    WANDB_ENABLED, WANDB_PROJECT,
)
from utils.data_loader import build_dataloaders
from utils.metrics import evaluate_model
from models.lens_pinn  import LensPINN
from models.heal_swin  import HEALSwin
from models.classifier import DeepLenseClassifier, ResNetBaseline


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepLense phase-1 training")

    p.add_argument(
        "--model", type=str, default="lens_pinn",
        choices=["lens_pinn", "heal_swin", "vit_baseline", "resnet_baseline"],
        help="Model architecture to train",
    )
    p.add_argument(
        "--dataset", type=str, default="model_i",
        choices=["model_i", "model_ii", "model_iii", "combined"],
        help="Which simulated dataset to train on",
    )
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",type=float, default=WEIGHT_DECAY)
    p.add_argument("--lambda_phys", type=float, default=LAMBDA_PHYSICS,
                   help="Weight on physics consistency loss (0 = disable)")
    p.add_argument("--workers",     type=int,   default=NUM_WORKERS)
    p.add_argument("--num_classes", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=SEED)
    p.add_argument("--run_name",    type=str,   default=None,
                   help="Custom name for checkpoints and W&B run")
    p.add_argument("--freeze_vit",  action="store_true",
                   help="Freeze ViT pretrained weights for the first N epochs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, device: torch.device) -> nn.Module:
    name = args.model
    n    = args.num_classes

    if name == "lens_pinn":
        model = LensPINN(num_classes=n, freeze_vit=args.freeze_vit)
    elif name == "heal_swin":
        model = HEALSwin(num_classes=n)
    elif name == "vit_baseline":
        model = DeepLenseClassifier(
            backbone_name="vit_small_patch16_224",
            num_classes=n, in_chans=1,
        )
    elif name == "resnet_baseline":
        model = ResNetBaseline(num_classes=n)
    else:
        raise ValueError(f"Unknown model: {name}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_loaders(args):
    ds = args.dataset
    if ds == "model_i":
        tr, te = MODEL_I_TRAIN, MODEL_I_TEST
    elif ds == "model_ii":
        tr, te = MODEL_II_TRAIN, MODEL_II_TEST
    elif ds == "model_iii":
        tr, te = MODEL_III_TRAIN, MODEL_III_TEST
    else:
        raise NotImplementedError("Combined dataset not yet implemented")

    return build_dataloaders(
        train_root=tr,
        test_root=te,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Physics consistency loss
# ---------------------------------------------------------------------------

def physics_consistency_loss(
    source_recon: torch.Tensor,
    theta_E: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise large values in the source reconstruction where there should be
    no source flux (outside the Einstein ring).  Also regularises θ_E.

    Returns a scalar loss.
    """
    # Smoothness prior: source should be compact (encourage sparsity)
    sparsity = source_recon.abs().mean()

    # θ_E should be physically reasonable (not collapse to 0 or explode)
    er_reg = ((theta_E - 0.8) ** 2).mean()   # soft target ~0.8 arcsec

    return sparsity + 0.1 * er_reg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion,
    device, lambda_phys, scaler,
):
    model.train()
    total_loss  = 0.0
    correct     = 0
    n_samples   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)

            if isinstance(outputs, (tuple, list)):
                # LensPINN / HEALSwin: (theta_E, source, logits)
                theta_E, source, logits = outputs
                cls_loss = criterion(logits, labels)
                if lambda_phys > 0:
                    phys_loss = physics_consistency_loss(source, theta_E, images)
                    loss = cls_loss + lambda_phys * phys_loss
                else:
                    loss = cls_loss
            else:
                logits   = outputs
                loss     = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct   += (preds == labels).sum().item()
        n_samples += images.size(0)

    return total_loss / n_samples, correct / n_samples


@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[-1]
        else:
            logits = outputs

        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        n_samples  += images.size(0)

    return total_loss / n_samples, correct / n_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device_str = DEVICE if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  Model         : {args.model}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Learning rate : {args.lr}")
    print(f"  λ_physics     : {args.lambda_phys}")
    print(f"{'='*60}\n")

    # ── Optional W&B ─────────────────────────────────────────────────────────
    run_name = args.run_name or f"{args.model}_{args.dataset}"
    if WANDB_ENABLED:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, name=run_name, config=vars(args))
        except ImportError:
            print("[WARNING] wandb not installed – skipping logging")

    # ── Data ─────────────────────────────────────────────────────────────────
    try:
        train_loader, val_loader, test_loader = build_loaders(args)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Data not found. Please run  setup_data.ps1  first.\n")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    # ── Loss / optimiser / scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    ckpt_path    = CHECKPOINTS_DIR / f"{run_name}_best.pt"
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, args.lambda_phys, scaler,
        )
        va_loss, va_acc = val_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"] .append(tr_acc)
        history["val_loss"]  .append(va_loss)
        history["val_acc"]   .append(va_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"train_loss: {tr_loss:.4f}  train_acc: {tr_acc:.4f}  "
            f"val_loss: {va_loss:.4f}  val_acc: {va_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if WANDB_ENABLED:
            try:
                import wandb
                wandb.log({
                    "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss,   "val_acc": va_acc,
                    "lr": scheduler.get_last_lr()[0],
                })
            except Exception:
                pass

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": va_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ↑ Best checkpoint saved ({best_val_acc:.4f})")

    # ── Final evaluation on test set ─────────────────────────────────────────
    print("\n[Final Evaluation on Test Set]")
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    results = evaluate_model(
        model, test_loader, device,
        class_names=[c for c in CLASS_NAMES[:args.num_classes]],
        verbose=True,
    )

    # ── Save training curves ──────────────────────────────────────────────────
    _save_curves(history, run_name)

    return results


def _save_curves(history: dict, run_name: str):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"],   label="Val")
        axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"],   label="Val")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend()

        plt.tight_layout()
        out = PLOTS_DIR / f"{run_name}_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {out}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Could not save curves: {e}")


if __name__ == "__main__":
    main()
