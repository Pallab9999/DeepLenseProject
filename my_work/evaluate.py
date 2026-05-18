"""
evaluate.py
===========
Standalone evaluation script.  Loads a checkpoint and runs full evaluation:
  - AUC (micro + macro)
  - F1 (macro)
  - Per-class AUC
  - Confusion matrix plot
  - GradCAM visualisation on a random sample (optional)

Usage
-----
    cd my_work
    python evaluate.py --checkpoint ../results/checkpoints/lens_pinn_model_i_best.pt \\
                       --model lens_pinn \\
                       --dataset model_i \\
                       [--gradcam]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE, NUM_WORKERS, DEVICE, IMAGE_SIZE,
    CLASS_NAMES, PLOTS_DIR,
    MODEL_I_TEST, MODEL_II_TEST, MODEL_III_TEST,
)
from utils.data_loader import DeepLenseDataset, get_val_transform
from utils.metrics import evaluate_model, per_class_auc, plot_confusion_matrix, gradcam_visualise
from models.lens_pinn  import LensPINN
from models.heal_swin  import HEALSwin
from models.classifier import DeepLenseClassifier, ResNetBaseline

from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepLense evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--model", type=str, default="lens_pinn",
        choices=["lens_pinn", "heal_swin", "vit_baseline", "resnet_baseline"],
    )
    p.add_argument(
        "--dataset", type=str, default="model_i",
        choices=["model_i", "model_ii", "model_iii"],
    )
    p.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--workers",     type=int, default=NUM_WORKERS)
    p.add_argument("--gradcam",     action="store_true",
                   help="Produce GradCAM visualisation")
    p.add_argument("--confusion",   action="store_true",
                   help="Plot confusion matrix")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_root = {
        "model_i":   MODEL_I_TEST,
        "model_ii":  MODEL_II_TEST,
        "model_iii": MODEL_III_TEST,
    }[args.dataset]

    try:
        test_ds = DeepLenseDataset(
            root=test_root,
            class_names=CLASS_NAMES[:args.num_classes],
            transform=get_val_transform(IMAGE_SIZE),
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\nHave you downloaded the data?\n")
        sys.exit(1)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    n = args.num_classes
    if args.model == "lens_pinn":
        model = LensPINN(num_classes=n)
    elif args.model == "heal_swin":
        model = HEALSwin(num_classes=n)
    elif args.model == "vit_baseline":
        model = DeepLenseClassifier(num_classes=n, in_chans=1)
    else:
        model = ResNetBaseline(num_classes=n)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    print(f"\nLoaded checkpoint: {args.checkpoint}")

    # ── Full evaluation ───────────────────────────────────────────────────────
    results = evaluate_model(
        model, test_loader, device,
        class_names=CLASS_NAMES[:n],
        verbose=True,
    )

    print("\n[Per-class AUC]")
    per_class_auc(model, test_loader, device, CLASS_NAMES[:n])

    # ── Optional plots ────────────────────────────────────────────────────────
    if args.confusion:
        cm_path = str(PLOTS_DIR / f"{args.model}_{args.dataset}_cm.png")
        plot_confusion_matrix(model, test_loader, device, CLASS_NAMES[:n], save_path=cm_path)

    if args.gradcam and args.model in {"lens_pinn", "heal_swin"}:
        # Pick one image from the test set
        sample_img, sample_label = test_ds[0]
        sample_img = sample_img.unsqueeze(0).to(device)

        # GradCAM target layer differs by model
        if args.model == "lens_pinn":
            target_layer = model.cnn_source.net[-3]
        else:
            target_layer = list(model.swin.children())[-1]

        cam_path = str(PLOTS_DIR / f"{args.model}_{args.dataset}_gradcam.png")
        gradcam_visualise(
            model=model,
            image=sample_img,
            target_layer=target_layer,
            save_path=cam_path,
        )

    return results


if __name__ == "__main__":
    main()
