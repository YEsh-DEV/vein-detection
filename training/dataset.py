#!/usr/bin/env python3
"""
training/dataset.py
--------------------
PyTorch Dataset and DataLoader for Palm Vein Recognition (AMPVNet + AdaFace).

Directory layout expected:
    <data_dir>/
        <subject_id_1>/
            image_01.png (or .jpg, .bmp)
            image_02.png
        <subject_id_2>/
            ...

Preprocessing Pipeline (Step 4 item 2 / exact match to backend cnn_extractor.py):
    1. Load image, convert to grayscale if not already.
    2. Resize to 224x224 with cv2.INTER_CUBIC (or PIL BICUBIC).
    3. Duplicate to 3 channels: [R, G, B] = [I, I, I].
    4. Convert to float32, scale to [0, 1] (divide by 255.0).
    5. Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]:
       formula: (pixel_val - 0.5) / 0.5  -> maps [0, 1] to [-1.0, 1.0].
       NOTE: We use 0.5/0.5 mean/std instead of ImageNet stats because palm vein
       NIR imagery is not natural RGB imagery; 0.5/0.5 is the defensible symmetric
       default for grayscale-replicated input. Backend cnn_extractor.py must
       copy this exact normalization.
    6. Training split only: apply RPT (Random Perspective Transform) and
       RGA (Random Gamma Adjustment). Validation split gets NO augmentation.

Subject-Independent Protocol (Step 4 item 3 / paper Section IV):
    Splits subjects (identities), NOT individual images, into train/val
    using the paper's 5:5 (or configurable 8:2) split ratio.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from augmentations import (
    build_train_transform,
    build_inference_transform,
    RandomPerspectiveTransform,
    RandomGammaAdjustment,
    NORM_MEAN,
    NORM_STD,
)


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def preprocess_image_to_tensor(
    image_input,
    target_size: Tuple[int, int] = (224, 224),
    transform: Optional[T.Compose] = None,
) -> torch.Tensor:
    """
    Standard preprocessing pipeline per Step 4 item 2:
      a. Load image, convert to grayscale if not already.
      b. Resize/verify 224x224 (bicubic / cv2.INTER_CUBIC equivalent).
      c. Duplicate to 3 channels.
      d. Convert to float32, scale to [0, 1].
      e. Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
      f. Optional transform (e.g. RPT/RGA in training).
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            # Convert RGB to grayscale
            pil_img = Image.fromarray(image_input).convert("L")
        elif image_input.ndim == 3 and image_input.shape[2] == 1:
            pil_img = Image.fromarray(image_input.squeeze(2), mode="L")
        else:
            pil_img = Image.fromarray(image_input.astype(np.uint8), mode="L")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize to target 224x224 with BICUBIC
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.BICUBIC)

    # Replicate single grayscale channel to 3 channels: [I, I, I]
    pil_rgb = pil_img.convert("RGB")

    # Apply transform if provided (contains RPT, RGA, ToTensor, Normalize)
    if transform is not None:
        tensor = transform(pil_rgb)
    else:
        # Default deterministic preprocessing
        infer_transform = build_inference_transform(image_size=target_size[0])
        tensor = infer_transform(pil_rgb)

    return tensor


class PalmVeinDataset(Dataset):
    """
    PyTorch Dataset for Palm Vein Recognition.
    Loads images from <data_dir>/<subject_id>/<image_file>.
    Splits at the subject level to guarantee zero identity leakage between train and val.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_ratio: float = 0.5,
        seed: int = 42,
        r_rpt: float = 0.4,
        p_rpt: float = 0.5,
        gamma_rga: float = 0.6,
        p_rga: float = 0.3,
        transform: Optional[T.Compose] = None,
        is_train: Optional[bool] = None,
        max_classes: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing subdirectories named by subject ID.
            split: 'train', 'val', or 'all'.
            split_ratio: Fraction of subjects to allocate to train (e.g. 0.5 or 0.8).
            seed: Random seed for deterministic subject splitting.
            r_rpt: RPT distortion scale.
            p_rpt: RPT probability.
            gamma_rga: RGA gamma parameter.
            p_rga: RGA probability.
            transform: Custom transform override (if None, standard transforms are built).
            is_train: Explicit override for train vs inference transform (default: split == 'train').
            max_classes: Optional maximum number of classes to load.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_ratio = float(split_ratio)
        self.seed = int(seed)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Collect subject directories
        subject_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not subject_dirs:
            raise ValueError(
                f"No subject subdirectories found in {self.data_dir}. "
                f"Expected: {self.data_dir}/<subject_id>/<images>"
            )

        # Subject-independent split (Step 4 item 3)
        # Deterministically permute subject list with seed
        rng = random.Random(self.seed)
        shuffled_subjects = list(subject_dirs)
        rng.shuffle(shuffled_subjects)

        n_train = max(1, int(len(shuffled_subjects) * self.split_ratio))
        if self.split == "train":
            active_subjects = shuffled_subjects[:n_train]
        elif self.split == "val":
            active_subjects = shuffled_subjects[n_train:]
            if not active_subjects:
                # Fallback if too few subjects: use all for val testing
                active_subjects = shuffled_subjects
        elif self.split == "all":
            active_subjects = shuffled_subjects
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train', 'val', or 'all'.")

        if max_classes is not None and max_classes > 0:
            active_subjects = active_subjects[:max_classes]

        # Map active subjects to consecutive integer labels [0, num_classes - 1]
        active_subjects = sorted(active_subjects, key=lambda d: d.name)
        self.subject_to_label: Dict[str, int] = {
            d.name: idx for idx, d in enumerate(active_subjects)
        }
        self.num_classes = len(active_subjects)

        # Collect all samples: (file_path, class_label, subject_id)
        self.samples: List[Tuple[str, int, str]] = []
        for subj_dir in active_subjects:
            subj_name = subj_dir.name
            class_label = self.subject_to_label[subj_name]
            for file_path in subj_dir.rglob("*"):
                if file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    self.samples.append((str(file_path), class_label, subj_name))

        if not self.samples:
            raise ValueError(f"No valid image files found for split '{self.split}' in {self.data_dir}")

        # Setup transforms
        training_mode = is_train if is_train is not None else (self.split == "train")
        if transform is not None:
            self.transform = transform
        else:
            if training_mode:
                self.transform = build_train_transform(
                    image_size=224,
                    r_rpt=r_rpt,
                    p_rpt=p_rpt,
                    gamma_rga=gamma_rga,
                    p_rga=p_rga,
                )
            else:
                self.transform = build_inference_transform(image_size=224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image_tensor: (3, 224, 224) float32 normalized tensor.
            class_label: integer class index.
        """
        img_path, class_label, _ = self.samples[idx]
        image_tensor = preprocess_image_to_tensor(img_path, transform=self.transform)
        return image_tensor, class_label

if __name__ == "__main__":
    print("--- PalmVeinDataset Module Loaded ---")
    data_path = Path("training/data")
    if data_path.exists() and any(d.is_dir() for d in data_path.iterdir()):
        train_ds = PalmVeinDataset(str(data_path), split="train", split_ratio=0.5)
        val_ds = PalmVeinDataset(str(data_path), split="val", split_ratio=0.5)
        print(f"Train dataset size: {len(train_ds)}, classes: {train_ds.num_classes}")
        print(f"Val dataset size  : {len(val_ds)}, classes: {val_ds.num_classes}")
    else:
        print("Dataset directory 'training/data' not yet populated. Ready for data ingestion.")

