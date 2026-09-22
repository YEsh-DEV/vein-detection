#!/usr/bin/env python3
"""
training/augmentations.py
--------------------------
Training-only data augmentation pipeline for AMPVNet palm vein recognition.

Augmentation Strategy — based on Luo et al. (IEEE TIFS 2024), Section IV-B &
system_architecture.md Section 7:
  1. RPT: Random Perspective Transformation — simulates 3D hand tilt / out-of-plane
     rotation (pitch, yaw, roll). r=0.4, p_RPT=0.5.
  2. RGA: Random Gamma Adjustment — simulates non-linear NIR illumination variation
     due to variable palm height above NIR LEDs (5cm-15cm). V_out = V_in^gamma,
     gamma in [1 - gamma_param, 1 + gamma_param] (e.g. [0.4, 1.6] for gamma=0.6), p_RGA=0.3.

WARNING — INFERENCE RESTRICTION:
  These transforms are strictly TRAINING-ONLY. At inference time, NEVER apply RPT,
  RGA, or horizontal flip. The inference pipeline receives a normalized 224x224x3
  grayscale-replicated crop and passes it directly to the model.
"""

import random
from typing import Tuple, Optional, Union
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomPerspectiveTransform:
    """
    Random Perspective Transformation (RPT) per Luo et al. (2024) Section IV-B1.

    Given a 224x224 ROI image and scale parameter r (default 0.4), randomly perturbs
    the four corner points within [0, r * side_length / 2] and applies perspective warp.
    Simulates out-of-plane hand tilt during unconstrained presentation.

    Args:
        r: Maximum distortion scale parameter (default: 0.4).
        p: Probability of applying the transform (default: 0.5).
    """

    def __init__(self, r: float = 0.4, p: float = 0.5):
        self.r = float(r)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        width, height = img.size
        max_dx = width * self.r * 0.5
        max_dy = height * self.r * 0.5

        startpoints = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        endpoints = [
            (random.uniform(0, max_dx), random.uniform(0, max_dy)),
            (width - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)),
            (width - 1 - random.uniform(0, max_dx), height - 1 - random.uniform(0, max_dy)),
            (random.uniform(0, max_dx), height - 1 - random.uniform(0, max_dy)),
        ]
        return TF.perspective(
            img,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )

    def __repr__(self) -> str:
        return f"RandomPerspectiveTransform(r={self.r}, p={self.p})"


class RandomGammaAdjustment:
    """
    Random Gamma Adjustment (RGA) per Luo et al. (2024) Section IV-B1.

    V_out = V_in^gamma, where gamma is drawn uniformly from [1 - gamma_param, 1 + gamma_param].
    Simulates variable palm distance from NIR illumination source.

    Args:
        gamma: Half-width of gamma variation range (default: 0.6 -> gamma in [0.4, 1.6]).
        p:     Probability of applying the transform (default: 0.3).
    """

    def __init__(
        self,
        gamma: float = 0.6,
        p: float = 0.3,
        gamma_range: Optional[Tuple[float, float]] = None,
    ):
        if gamma_range is not None:
            self.gamma_range = gamma_range
        else:
            self.gamma_range = (max(0.01, 1.0 - float(gamma)), 1.0 + float(gamma))
        self.gamma_param = float(gamma)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(img, gamma=gamma)

    def __repr__(self) -> str:
        return f"RandomGammaAdjustment(gamma_range={self.gamma_range}, p={self.p})"


# ---------------------------------------------------------------------------
# Normalization Statistics
# ---------------------------------------------------------------------------
# Normalization with simple 0.5/0.5 mean/std per channel (mean=[0.5,0.5,0.5],
# std=[0.5,0.5,0.5]) per Step 4 item 2e and system_architecture.md.
# Input is grayscale-derived 3-channel NIR, NOT natural ImageNet RGB.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_train_transform(
    image_size: int = 224,
    r_rpt: float = 0.4,
    p_rpt: float = 0.5,
    gamma_rga: float = 0.6,
    p_rga: float = 0.3,
    hflip_p: float = 0.0,
) -> T.Compose:
    """
    Builds the training-time augmentation and preprocessing pipeline.

    Args:
        image_size: Target square size (default: 224).
        r_rpt:      RPT corner shift scale (default: 0.4).
        p_rpt:      Probability of applying RPT (default: 0.5).
        gamma_rga:  RGA gamma variation half-range (default: 0.6 -> [0.4, 1.6]).
        p_rga:      Probability of applying RGA (default: 0.3).
        hflip_p:    Horizontal flip probability (default: 0.0 to preserve chirality).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transforms = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
    ]

    if hflip_p > 0.0:
        transforms.append(T.RandomHorizontalFlip(p=hflip_p))

    if p_rpt > 0.0:
        transforms.append(RandomPerspectiveTransform(r=r_rpt, p=p_rpt))

    if p_rga > 0.0:
        transforms.append(RandomGammaAdjustment(gamma=gamma_rga, p=p_rga))

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return T.Compose(transforms)


def build_inference_transform(image_size: int = 224) -> T.Compose:
    """
    Deterministic inference-time preprocessing pipeline.
    NO augmentation. Matches backend cnn_extractor.py exactly:
      1. Resize to (224, 224) via BICUBIC interpolation.
      2. Convert to float32 tensor [0, 1].
      3. Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def gray_to_3channel_tensor(
    gray_img: np.ndarray,
    transform: Optional[T.Compose] = None,
) -> torch.Tensor:
    """
    Converts 2D grayscale NumPy array (H x W) in [0, 255] to 3-channel tensor (3, 224, 224).
    """
    if transform is None:
        transform = build_inference_transform()

    if gray_img.ndim == 3 and gray_img.shape[2] == 1:
        gray_img = gray_img.squeeze(2)
    if gray_img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {gray_img.shape}")

    pil_gray = Image.fromarray(gray_img.astype(np.uint8), mode="L")
    pil_rgb = pil_gray.convert("RGB")
    return transform(pil_rgb)


if __name__ == "__main__":
    print("--- Testing training augmentations (RPT + RGA) ---")
    train_tf = build_train_transform(r_rpt=0.4, p_rpt=1.0, gamma_rga=0.6, p_rga=1.0)
    infer_tf = build_inference_transform()

    dummy_gray = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    pil_rgb = Image.fromarray(dummy_gray, mode="L").convert("RGB")

    train_tensor = train_tf(pil_rgb)
    infer_tensor = infer_tf(pil_rgb)

    assert train_tensor.shape == (3, 224, 224)
    assert infer_tensor.shape == (3, 224, 224)
    print(f"Train tensor shape: {train_tensor.shape}, min: {train_tensor.min():.2f}, max: {train_tensor.max():.2f}")
    print(f"Infer tensor shape: {infer_tensor.shape}, min: {infer_tensor.min():.2f}, max: {infer_tensor.max():.2f}")
    print("Augmentations unit test PASSED ✓")
