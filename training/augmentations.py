#!/usr/bin/env python3
"""
training/augmentations.py
--------------------------
Training-only data augmentation pipeline for AMPVNet palm vein recognition.

Augmentation Strategy — based on Luo et al. (IEEE TIFS 2024), Section IV-B:
  1. RPT: Random Perspective Transformation — simulates 3D hand tilt / out-of-plane
           rotation (pitch, yaw). distortion_scale=0.4, p=0.5.
  2. RGA: Random Gamma Adjustment — simulates NIR illumination variation due to
           hand height changes (5cm-15cm). gamma range=[1-0.6, 1+0.6] = [0.4, 1.6], p=0.3.

⚠ INFERENCE NOTE:
  These transforms are TRAINING-ONLY. At inference time (app/server.py and cam_test.py),
  NEVER apply RPT, RGA, or horizontal flip. The production pipeline receives a live
  224x224x3 grayscale-replicated crop and passes it directly to the ONNX model.
  Applying augmentations at inference would systematically degrade matching accuracy.
"""

import random
from typing import Tuple, Optional
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Custom RGA: Random Gamma Adjustment
# ---------------------------------------------------------------------------

class RandomGammaAdjustment:
    """
    RGA — Random Gamma Adjustment.
    Simulates the non-linear photometric variation caused by variable palm height
    above the NIR illuminator (V_out = V_in ^ gamma).

    From Luo et al. (2024) Section IV-B:
      gamma range: [1 - 0.6, 1 + 0.6] = [0.4, 1.6]
      probability: p = 0.3

    Args:
        gamma_range: (min_gamma, max_gamma). Values < 1.0 brighten, > 1.0 darken.
        p:           Probability of applying the transform.
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.4, 1.6), p: float = 0.3):
        assert 0.0 < gamma_range[0] < gamma_range[1], \
            f"Invalid gamma_range: {gamma_range}. Must be 0 < min < max."
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img: PIL Image (any mode).
        Returns:
            Gamma-adjusted PIL Image (same mode).
        """
        if random.random() > self.p:
            return img

        gamma = random.uniform(*self.gamma_range)
        # torchvision adjust_gamma expects PIL image and returns PIL image
        return TF.adjust_gamma(img, gamma=gamma)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"gamma_range={self.gamma_range}, p={self.p})")


# ---------------------------------------------------------------------------
# Normalization Statistics
# ---------------------------------------------------------------------------
# We replicate the grayscale channel into 3 channels. ImageNet statistics
# are NOT appropriate for NIR palm vein images. We use channel-neutral stats
# (mean=0.5, std=0.5 per channel) for initial training.
# After collecting real data, run tools/real_data_analysis.py and replace
# these with empirically measured statistics.
#
# PLACEHOLDER — replace with real dataset statistics after first full epoch.
NORM_MEAN = (0.5, 0.5, 0.5)   # PLACEHOLDER: measure on CASIA dataset
NORM_STD  = (0.5, 0.5, 0.5)   # PLACEHOLDER: measure on CASIA dataset


# ---------------------------------------------------------------------------
# Training Transform
# ---------------------------------------------------------------------------

def build_train_transform(
    image_size: int = 224,
    rpt_distortion_scale: float = 0.4,
    rpt_p: float = 0.5,
    rga_gamma_range: Tuple[float, float] = (0.4, 1.6),
    rga_p: float = 0.3,
    hflip_p: float = 0.5,
) -> T.Compose:
    """
    Builds the training-time augmentation and preprocessing pipeline.

    Operations applied in order:
      1. Resize to (image_size x image_size) — maintains aspect ratio via
         resize + center crop to avoid distortion if source isn't square.
      2. RandomHorizontalFlip — handles left/right hand chirality ambiguity.
         (p=0.5 by default; harmless since chirality is resolved by MediaPipe
         at inference, but fine to expose both orientations during training)
      3. RPT — Random Perspective Transformation (distortion_scale=0.4, p=0.5)
         Simulates 3D out-of-plane hand rotation for unconstrained recognition.
      4. RGA — Random Gamma Adjustment (gamma=[0.4, 1.6], p=0.3)
         Simulates NIR illumination variation due to palm-to-sensor distance.
      5. ToTensor — converts PIL Image [0,255] to float32 tensor [0.0, 1.0]
      6. Normalize — subtract mean, divide by std per channel.

    ⚠ TRAINING-ONLY. Never use this transform at inference time.

    Args:
        image_size:              Target square size (default 224 for AMPVNet).
        rpt_distortion_scale:    RPT warp severity [0,1]. Paper uses 0.4.
        rpt_p:                   Probability of applying RPT. Paper uses 0.5.
        rga_gamma_range:         (min_gamma, max_gamma). Paper uses [0.4, 1.6].
        rga_p:                   Probability of applying RGA. Paper uses 0.3.
        hflip_p:                 Probability of horizontal flip. Default 0.5.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        # ---- Spatial normalization ----
        T.Resize((image_size, image_size)),              # Resize to 224x224
        T.RandomHorizontalFlip(p=hflip_p),              # Chirality augmentation

        # ---- RPT: 3D tilt simulation ----
        # torchvision RandomPerspective distorts the image by randomly shifting
        # its four corners. distortion_scale=0.4 matches the paper.
        T.RandomPerspective(
            distortion_scale=rpt_distortion_scale,
            p=rpt_p,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,     # zero-fill boundary (matches zero-padding in ROI extraction)
        ),

        # ---- RGA: Illumination simulation ----
        RandomGammaAdjustment(gamma_range=rga_gamma_range, p=rga_p),

        # ---- Tensor conversion & channel normalization ----
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Inference Transform (no augmentation)
# ---------------------------------------------------------------------------

def build_inference_transform(image_size: int = 224) -> T.Compose:
    """
    Inference-time transform — NO augmentation, deterministic preprocessing only.

    Used when:
      - Evaluating the model during validation.
      - Extracting embeddings for database enrollment via ONNX Runtime (server.py).
      - Running the accuracy calibration harness (tools/real_data_analysis.py).

    ⚠ This must match EXACTLY what app/server.py does when preparing the
      224x224x3 tensor before ONNX inference. Any mismatch creates a
      train/inference domain gap that silently degrades accuracy.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Utility: Grayscale image -> 3-channel tensor for AMPVNet
# ---------------------------------------------------------------------------

def gray_to_3channel_tensor(
    gray_img: np.ndarray,
    transform: Optional[T.Compose] = None,
) -> torch.Tensor:
    """
    Converts a single-channel 8-bit grayscale NumPy array (H x W) or
    (H x W x 1) to a 3-channel RGB PIL Image, then applies the given transform.

    This replicates the inference preprocessing chain used in app/server.py:
      1. 224x224 grayscale ROI (numpy uint8)
      2. Replicate to 3 channels: [R, G, B] = [I, I, I]
      3. Apply inference transform -> (1, 3, 224, 224) float32 tensor

    Args:
        gray_img:  Grayscale 2D or 3D numpy array, values in [0, 255].
        transform: A Compose transform to apply (default: build_inference_transform()).

    Returns:
        Tensor of shape (3, 224, 224).
    """
    if transform is None:
        transform = build_inference_transform()

    if gray_img.ndim == 3 and gray_img.shape[2] == 1:
        gray_img = gray_img.squeeze(2)
    if gray_img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {gray_img.shape}")

    # Convert uint8 to PIL, then to RGB (3-channel replication)
    pil_gray = Image.fromarray(gray_img.astype(np.uint8), mode="L")
    pil_rgb  = pil_gray.convert("RGB")

    return transform(pil_rgb)     # (3, 224, 224)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Testing training augmentations ---")
    train_tf = build_train_transform()
    infer_tf = build_inference_transform()

    # Simulate a 224x224 grayscale NIR palm image
    dummy_gray = np.random.randint(0, 200, (224, 224), dtype=np.uint8)
    pil_gray   = Image.fromarray(dummy_gray, mode="L").convert("RGB")

    train_tensor = train_tf(pil_gray)
    infer_tensor = infer_tf(pil_gray)

    assert train_tensor.shape == (3, 224, 224), f"Bad training tensor shape: {train_tensor.shape}"
    assert infer_tensor.shape == (3, 224, 224), f"Bad inference tensor shape: {infer_tensor.shape}"

    gray_tensor = gray_to_3channel_tensor(dummy_gray, infer_tf)
    assert gray_tensor.shape == (3, 224, 224), f"Bad gray->3ch tensor shape: {gray_tensor.shape}"

    print(f"  Training tensor shape  : {tuple(train_tensor.shape)} ✓")
    print(f"  Inference tensor shape : {tuple(infer_tensor.shape)} ✓")
    print(f"  gray_to_3ch tensor     : {tuple(gray_tensor.shape)} ✓")
    print(f"  Train transforms:\n    {train_tf}")
    print("[augmentations.py] Smoke test PASSED ✓")
