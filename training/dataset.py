#!/usr/bin/env python3
"""
training/dataset.py
--------------------
Dataset loaders for two distinct data sources:

  (a) CASIA-MS-PalmprintV1 — Public NIR palmprint dataset for Phase 1 pretraining.
      Expected layout AFTER download (user provides data):
          training/data_raw/casia/
            <subject_id>/           (e.g. "001", "002", ...)
              <session>/            (e.g. "01", "02", ...)
                <image>.jpg         (or .bmp, .png)

      Alternatively, if CASIA is pre-organized as:
          training/data_raw/casia/<class_name>/<image>.*
      it will also work — the loader accepts any ImageFolder-compatible structure.

  (b) Own captured ROI images — Phase 2 fine-tuning and held-out validation.
      Expected layout (user copies from Pi's roi_clahe/ directory):
          training/data_raw/own/
            <username>/
              *.png               (224x224 grayscale CLAHE ROI images)

IMPORTANT DESIGN DECISION:
  CASIA and own-data subjects are kept as TWO SEPARATE loaders, not merged.
  This is intentional:
    - CASIA (many subjects) -> Phase 1 pretraining for generalizable vascular features.
    - Own data (few subjects, known to deployment) -> Phase 2 fine-tuning & validation.
  Merging them randomly would produce a misleading accuracy metric and could leak
  deployment-domain subjects into pretraining, invalidating the fine-tuning phase.

ROI convention:
  All images are loaded as single-channel grayscale (mode="L"), converted to
  3-channel RGB by replication, then resized/padded to 224x224.
  This matches the β=1.6 scalable ROI extraction in app/mediapipe_img.py.
"""

import os
import re
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_image_paths(root_dir: str) -> List[Tuple[str, int]]:
    """
    Recursively collects (image_path, class_index) tuples from an ImageFolder-
    compatible directory. Each immediate subdirectory is one identity class.

    Subject directories are sorted alphabetically for reproducible class indices.

    Args:
        root_dir: Path to folder containing one subdirectory per subject.

    Returns:
        List of (absolute_image_path, class_index) tuples.
    Raises:
        FileNotFoundError if root_dir does not exist.
        ValueError if no subjects or no valid images are found.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root_dir}\n"
            "Please create the directory and place your data there."
        )

    _VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Collect all immediate subdirectories (one per subject)
    subject_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not subject_dirs:
        raise ValueError(
            f"No subject subdirectories found in {root_dir}.\n"
            "Expected: {root_dir}/<subject_name>/<image_file>.*"
        )

    samples = []
    for class_idx, subject_dir in enumerate(subject_dirs):
        # Recursively find all images within this subject's directory
        for img_path in subject_dir.rglob("*"):
            if img_path.suffix.lower() in _VALID_EXTENSIONS:
                samples.append((str(img_path), class_idx))

    if not samples:
        raise ValueError(
            f"No valid images found in {root_dir}. "
            f"Accepted extensions: {_VALID_EXTENSIONS}"
        )

    return samples


def _build_class_map(root_dir: str) -> Dict[str, int]:
    """
    Returns {subject_name -> class_index} mapping, alphabetically sorted.
    Subject name is the immediate subdirectory name.
    """
    root = Path(root_dir)
    subject_dirs = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return {name: idx for idx, name in enumerate(subject_dirs)}


def _load_as_3channel_pil(image_path: str, target_size: int = 224) -> Image.Image:
    """
    Loads any image (grayscale NIR or color) as 3-channel RGB PIL Image at
    target_size x target_size, preserving aspect ratio with zero-padding.

    Processing steps:
      1. Open image file.
      2. Convert to single-channel grayscale ("L") — discards color info
         from RGB palmprints, retaining vascular structure.
      3. Pad to square (letterbox style) to avoid aspect ratio distortion.
      4. Resize to target_size x target_size.
      5. Convert to "RGB" by channel replication (3-channel grayscale).

    Returns:
        PIL Image in "RGB" mode at (target_size, target_size).
    """
    img = Image.open(image_path).convert("L")   # Load as grayscale

    # --- Aspect-ratio-preserving letterbox padding ---
    w, h = img.size
    if w != h:
        max_side = max(w, h)
        new_img = Image.new("L", (max_side, max_side), color=0)  # zero-pad
        paste_x = (max_side - w) // 2
        paste_y = (max_side - h) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img

    # --- Resize to target square ---
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)

    # --- Replicate to 3 channels: [R, G, B] = [I, I, I] ---
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# PalmVeinDataset
# ---------------------------------------------------------------------------

class PalmVeinDataset(Dataset):
    """
    Generic dataset for palm vein image classification.

    Suitable for both CASIA and own-captured ROI images.
    Subject directories are alphabetically sorted for deterministic class indices.

    Args:
        root_dir:    Directory containing subject subdirectories.
        transform:   torchvision transform pipeline to apply to each PIL image.
        image_size:  Target image size. Must match AMPVNet input (224 by default).
        split_ratio: If not None, (train_fraction, val_fraction) tuple that splits
                     subjects by image: first split_ratio[0] fraction for train,
                     remaining for val. Only used if split='train' or 'val'.
        split:       'train', 'val', or 'all'. Only relevant when split_ratio is set.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[T.Compose] = None,
        image_size: int = 224,
        split_ratio: Optional[Tuple[float, float]] = None,
        split: str = "all",
    ):
        self.root_dir   = root_dir
        self.transform  = transform
        self.image_size = image_size
        self.split      = split

        # Collect all (path, class_idx) samples
        all_samples = _collect_image_paths(root_dir)
        self.class_map = _build_class_map(root_dir)
        self.num_classes = len(self.class_map)

        # Apply train/val split at the image level if requested
        if split_ratio is not None and split != "all":
            train_frac = split_ratio[0]
            n_total = len(all_samples)
            n_train = int(n_total * train_frac)
            if split == "train":
                all_samples = all_samples[:n_train]
            elif split == "val":
                all_samples = all_samples[n_train:]
            else:
                raise ValueError(f"split must be 'train', 'val', or 'all'. Got: {split}")

        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]

        pil_img = _load_as_3channel_pil(img_path, self.image_size)

        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = T.ToTensor()(pil_img)

        return img_tensor, class_idx

    def describe(self) -> str:
        return (
            f"PalmVeinDataset(root='{self.root_dir}', "
            f"split='{self.split}', "
            f"subjects={self.num_classes}, "
            f"images={len(self.samples)})"
        )


# ---------------------------------------------------------------------------
# Factory functions for Phase 1 (CASIA) and Phase 2 (own data)
# ---------------------------------------------------------------------------

def build_casia_loaders(
    casia_root: str,
    train_transform: T.Compose,
    val_transform: T.Compose,
    batch_size: int = 16,
    train_split: float = 0.85,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Builds train/val DataLoaders for Phase 1 pretraining on CASIA-MS-PalmprintV1.

    CASIA subjects are split 85% train / 15% val AT THE IMAGE LEVEL per subject.
    This is a simple random split — for robust evaluation, a subject-disjoint
    split (train on subjects 1-400, val on subjects 401-500) is preferred but
    requires knowing the exact dataset size in advance.

    Args:
        casia_root:      Path to training/data_raw/casia/
        train_transform: Training augmentation pipeline from augmentations.py.
        val_transform:   Inference-time transform (no augmentation).
        batch_size:      Training batch size. Paper uses 16.
        train_split:     Fraction of images for training (default 0.85).
        num_workers:     DataLoader worker processes.
        seed:            Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, num_classes)
    """
    train_ds = PalmVeinDataset(
        casia_root, transform=train_transform,
        split_ratio=(train_split, 1.0 - train_split), split="train"
    )
    val_ds = PalmVeinDataset(
        casia_root, transform=val_transform,
        split_ratio=(train_split, 1.0 - train_split), split="val"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True,        # avoid partial batches with AdaFace EMA
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset] CASIA pretraining: {train_ds.describe()}")
    print(f"[Dataset] CASIA validation : {val_ds.describe()}")

    return train_loader, val_loader, train_ds.num_classes


def build_own_loaders(
    own_root: str,
    train_transform: T.Compose,
    val_transform: T.Compose,
    batch_size: int = 8,
    train_split: float = 0.80,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Builds train/val DataLoaders for Phase 2 fine-tuning on own captured data.

    Own-data subjects are the deployment identities. They are ALWAYS kept
    completely separate from CASIA subjects — never merged into Phase 1 training.

    Expected data layout (copy from Pi's roi_clahe/ directory):
        training/data_raw/own/<username>/<roi_image>.png

    Args:
        own_root:        Path to training/data_raw/own/
        train_transform: Training augmentation pipeline from augmentations.py.
        val_transform:   Inference-time transform (no augmentation).
        batch_size:      Fine-tuning batch size. Smaller than pretraining.
        train_split:     Fraction of images for training (default 0.80).
        num_workers:     DataLoader worker processes.

    Returns:
        (train_loader, val_loader, num_classes)
    """
    train_ds = PalmVeinDataset(
        own_root, transform=train_transform,
        split_ratio=(train_split, 1.0 - train_split), split="train"
    )
    val_ds = PalmVeinDataset(
        own_root, transform=val_transform,
        split_ratio=(train_split, 1.0 - train_split), split="val"
    )

    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)),
        shuffle=True, num_workers=num_workers, pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, len(val_ds)),
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset] Own train : {train_ds.describe()}")
    print(f"[Dataset] Own val   : {val_ds.describe()}")

    return train_loader, val_loader, train_ds.num_classes


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("--- dataset.py sanity test (synthetic data) ---")

    # Create a synthetic dataset structure in temp dir
    with tempfile.TemporaryDirectory() as tmp:
        for subj in ["alice", "bob", "charlie"]:
            subj_dir = os.path.join(tmp, subj)
            os.makedirs(subj_dir)
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L"
                )
                img.save(os.path.join(subj_dir, f"sample_{i:02d}.png"))

        ds = PalmVeinDataset(tmp, transform=None)
        print(f"  {ds.describe()}")
        assert len(ds) == 12, f"Expected 12 samples, got {len(ds)}"
        x, y = ds[0]
        assert isinstance(x, torch.Tensor) and x.shape == (3, 224, 224), \
            f"Unexpected sample shape: {x.shape}"
        print(f"  Sample shape: {tuple(x.shape)}, class: {y}")
        print("[dataset.py] Sanity check PASSED ✓")
