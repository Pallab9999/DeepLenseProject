"""
utils/data_loader.py
====================
Dataset and DataLoader utilities for the DeepLense project.

Supports:
  - Model I, II, III  (simulated, 3-class today but 4-class ready)
  - Lazy label-from-folder parsing (standard ImageFolder layout)
  - Optional physics-informed preprocessing toggle
  - Pin-memory + persistent workers for fast GPU training

Expected on-disk layout
-----------------------
Data/
  Model_I/
    train/
      no_substructure/   *.npy  or  *.pt  or  *.png / *.jpg
      cdm_subhalos/
      axion/
    test/
      ...
  Model_II/ ...
  Model_III/ ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> torch.Tensor:
    """
    Load a single lensing image from .npy, .pt, .png, or .jpg.
    Always returns a float32 tensor of shape (1, H, W) normalised to [0, 1].
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    elif suffix == ".pt":
        arr = torch.load(str(path), weights_only=True).float().numpy()
    else:
        from PIL import Image
        img = Image.open(str(path)).convert("L")  # greyscale
        arr = np.array(img, dtype=np.float32) / 255.0

    # Ensure shape is (H, W)
    if arr.ndim == 3:
        arr = arr.squeeze()

    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    # Normalise to [0, 1] if not already
    mn, mx = tensor.min(), tensor.max()
    if mx > mn:
        tensor = (tensor - mn) / (mx - mn)

    return tensor


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

CLASS_ORDER = ["no_substructure", "cdm_subhalos", "axion", "vortex"]


class DeepLenseDataset(Dataset):
    """
    Filesystem dataset for DeepLense simulated images.

    Parameters
    ----------
    root : Path
        Directory containing one sub-folder per class.
    class_names : list[str]
        Names of the classes in label order.  Folders not in this list are
        silently ignored.  Classes present in class_names but absent from disk
        produce a warning and contribute 0 samples.
    transform : callable, optional
        Image-level transform applied after loading.
    """

    def __init__(
        self,
        root: Path | str,
        class_names: List[str] = CLASS_ORDER,
        transform: Optional[Callable] = None,
    ):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.root = root
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []

        for cls_name in class_names:
            cls_dir = root / cls_name
            if not cls_dir.is_dir():
                print(f"[DeepLenseDataset] WARNING: class folder not found: {cls_dir}")
                continue
            valid_exts = {".npy", ".pt", ".png", ".jpg", ".jpeg"}
            for fname in sorted(os.listdir(cls_dir)):
                _, ext = os.path.splitext(fname)
                if ext.lower() in valid_exts:
                    self.samples.append((cls_dir / fname, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image files found under {root}. "
                "Have you run setup_data.ps1 to download the datasets?"
            )

        print(
            f"[DeepLenseDataset] Loaded {len(self.samples)} samples from {root} "
            f"(classes: {class_names})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = _load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Keeps the physical content intact (no colour jitter, no drastic crops).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),  # lensing is rotationally symmetric
        # Normalise to zero-mean, unit-std (single channel)
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_root: Path | str,
    test_root: Path | str,
    class_names: List[str] = CLASS_ORDER,
    image_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    A random validation split is carved out of the training set so we can do
    early stopping without touching the held-out test set.
    """
    train_ds = DeepLenseDataset(
        root=train_root,
        class_names=class_names,
        transform=get_train_transform(image_size),
    )
    test_ds = DeepLenseDataset(
        root=test_root,
        class_names=class_names,
        transform=get_val_transform(image_size),
    )

    # split train → train + val
    n_val = int(len(train_ds) * val_split)
    n_train = len(train_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_ds = random_split(train_ds, [n_train, n_val], generator=generator)

    # Override transform on val split (no augmentation)
    val_split_ds.dataset = DeepLenseDataset(
        root=train_root,
        class_names=class_names,
        transform=get_val_transform(image_size),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_split, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_split_ds, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,      shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import MODEL_I_TRAIN, MODEL_I_TEST, IMAGE_SIZE, BATCH_SIZE

    try:
        train_loader, val_loader, test_loader = build_dataloaders(
            train_root=MODEL_I_TRAIN,
            test_root=MODEL_I_TEST,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
        batch, labels = next(iter(train_loader))
        print(f"Batch shape  : {batch.shape}")
        print(f"Labels sample: {labels[:8].tolist()}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches  : {len(val_loader)}")
        print(f"Test batches : {len(test_loader)}")
    except FileNotFoundError as e:
        print(f"\n[NOTE] {e}")
        print("Run setup_data.ps1 first to download the datasets.")
