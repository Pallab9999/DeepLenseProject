"""
utils/metrics.py
================
Evaluation metrics and GradCAM visualisation utilities.

Metrics
-------
  evaluate_model()  – AUC (micro + macro), F1, confusion matrix
  per_class_auc()   – per-class one-vs-rest AUC

GradCAM
-------
  gradcam_visualise()  – overlay activation map on a single image
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Runs inference on `dataloader` and returns a dict with:
        auc_micro, auc_macro, f1_macro, accuracy

    Also prints a classification report if `verbose=True`.

    Parameters
    ----------
    model       : trained PyTorch model.  forward() must return *logits*
                  of shape (B, num_classes).  If the model returns a tuple,
                  the last element is assumed to be the logits.
    dataloader  : validation or test DataLoader
    device      : cpu / cuda
    class_names : optional list of string labels for the printed report
    verbose     : print results to stdout

    Returns
    -------
    dict with float metrics
    """
    model.eval()
    all_probs:  List[np.ndarray] = []
    all_preds:  List[int]        = []
    all_labels: List[int]        = []

    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1]

        outputs = model(images)
        # Handle tuple outputs (e.g. LensPINN returns (theta_E, source, logits))
        if isinstance(outputs, (tuple, list)):
            logits = outputs[-1]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = all_probs.shape[1]

    # Only compute multi-class AUC if all classes are represented in the labels
    present = np.unique(all_labels)
    if len(present) == num_classes:
        auc_micro = float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="micro"))
        auc_macro = float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"))
    else:
        auc_micro = auc_macro = float("nan")
        print(f"[metrics] WARNING: only {len(present)}/{num_classes} classes found — AUC skipped.")

    f1   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    acc  = float((all_preds == all_labels).mean())

    results = {
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1_macro":  f1,
        "accuracy":  acc,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  AUC (micro): {auc_micro:.4f}")
        print(f"  AUC (macro): {auc_macro:.4f}")
        print(f"  F1  (macro): {f1:.4f}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"{'='*50}\n")
        if class_names is not None:
            print(classification_report(
                all_labels, all_preds,
                labels=list(range(len(class_names))),
                target_names=class_names,
                zero_division=0
            ))

    return results


def per_class_auc(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, float]:
    """One-vs-rest AUC for each class individually."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    n = all_probs.shape[1]

    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=list(range(n)))

    results: Dict[str, float] = {}
    for i, name in enumerate(class_names[:n]):
        try:
            auc = roc_auc_score(y_bin[:, i], all_probs[:, i])
        except Exception:
            auc = float("nan")
        results[name] = float(auc)
        print(f"  AUC [{name:20s}]: {auc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Compute and plot the confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set(
        xticks=ticks, yticks=ticks,
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label",
        title="Normalised Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] Confusion matrix saved to {save_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# GradCAM visualisation
# ---------------------------------------------------------------------------

def gradcam_visualise(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute and overlay a GradCAM activation map on `image`.

    Parameters
    ----------
    model        : model (must be in eval mode)
    image        : (1, C, H, W) or (C, H, W) tensor
    target_layer : the layer whose gradients are used (e.g. last conv or ViT norm)
    class_idx    : index of the target class; None = argmax of the prediction
    device       : computation device
    save_path    : if given, save the figure to this path

    Returns
    -------
    grayscale_cam : np.ndarray (H, W), values in [0, 1]
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        raise ImportError("Install grad-cam:  pip install grad-cam")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if class_idx is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=image, targets=targets)[0]  # (H, W)

    # Prepare RGB image for overlay (normalise to [0, 1])
    img_np = image.squeeze().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_rgb = np.stack([img_np] * 3, axis=-1)  # (H, W, 3)

    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np, cmap="gray"); axes[0].set_title("Input Image"); axes[0].axis("off")
    axes[1].imshow(grayscale_cam, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
    axes[2].imshow(cam_image); axes[2].set_title("Overlay"); axes[2].axis("off")

    plt.suptitle("GradCAM Interpretability", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] GradCAM saved to {save_path}")
        plt.close()
    else:
        plt.show()

    return grayscale_cam
