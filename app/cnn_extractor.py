#!/usr/bin/env python3
"""
cnn_extractor.py
----------------
AMPVNet CNN embedding extraction and cosine similarity matching module for
the Palm Vein Biometrics System.

Replaces the legacy Gabor-wavelet feature extractor (app/gabor.py).
Uses ONNX Runtime CPU execution provider (ARM NEON optimized on Raspberry Pi 5).
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from app.constants import AMPVNET_ONNX_PATH, EMBEDDING_DIM
except ImportError:
    from constants import AMPVNET_ONNX_PATH, EMBEDDING_DIM

# Module-level ONNX Runtime session and readiness flag
_session = None
_input_name = None
MODEL_LOADED = False

try:
    import onnxruntime as ort

    if os.path.exists(AMPVNET_ONNX_PATH):
        # Explicit CPUExecutionProvider avoids slow auto-discovery on Raspberry Pi 5
        _session = ort.InferenceSession(
            AMPVNET_ONNX_PATH,
            providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
        MODEL_LOADED = True
        print(f"[+] AMPVNet ONNX model loaded successfully from {AMPVNET_ONNX_PATH}")
    else:
        print(f"[!] AMPVNet model not found at {AMPVNET_ONNX_PATH} — CNN matching unavailable")
except Exception as e:
    print(f"[!] Failed to initialize ONNX Runtime session: {e} — CNN matching unavailable")
    _session = None
    _input_name = None
    MODEL_LOADED = False


def extract_embedding(roi: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-dimensional L2-normalized embedding from a 224x224 uint8
    grayscale palm ROI.

    Preprocessing order MUST strictly match the training pipeline:
      a. Duplicate grayscale to 3 channels: np.stack([roi, roi, roi], axis=-1)
      b. Scale uint8 to float32 [0.0, 1.0] by dividing by 255.0
      c. Apply identical channel normalization (mean=0.5, std=0.5 per channel):
         # Normalized using mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5) from training/augmentations.py lines 84-85
      d. Transpose to CHW (3, 224, 224) and add batch dim (1, 3, 224, 224)

    Args:
        roi: 224x224 uint8 grayscale numpy array.

    Returns:
        embedding: L2-normalized (512,) float32 numpy vector.
    """
    if not MODEL_LOADED or _session is None:
        raise RuntimeError(f"AMPVNet model is not loaded (path: {AMPVNET_ONNX_PATH}). Cannot extract embedding.")

    if roi is None or roi.size == 0:
        raise ValueError("Invalid input ROI: empty array provided.")

    if roi.ndim == 3 and roi.shape[2] == 1:
        roi = roi.squeeze(2)

    if roi.shape != (224, 224):
        # Resize defensively if input slightly differs from 224x224
        import cv2
        roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Step a: Duplicate single grayscale channel to 3 channels
    ch3 = np.stack([roi, roi, roi], axis=-1)  # shape (224, 224, 3)

    # Step b: Convert to float32 and scale to [0, 1]
    norm_img = ch3.astype(np.float32) / 255.0

    # Step c: Apply identical channel normalization
    # Copied from training/augmentations.py lines 84-85:
    #   NORM_MEAN = (0.5, 0.5, 0.5)
    #   NORM_STD  = (0.5, 0.5, 0.5)
    norm_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    norm_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    norm_img = (norm_img - norm_mean) / norm_std

    # Step d: Transpose HWC -> CHW format and add batch dimension
    tensor_input = np.transpose(norm_img, (2, 0, 1))  # (3, 224, 224)
    tensor_input = np.expand_dims(tensor_input, axis=0)  # (1, 3, 224, 224)

    # Run ONNX Runtime inference
    outputs = _session.run(None, {_input_name: tensor_input})
    raw_embedding = outputs[0]  # shape (1, 512)

    # Model internally L2-normalizes (Linear(512) -> L2 norm per training/model.py),
    # but defensively re-normalize to ensure strict unit length
    raw_embedding = raw_embedding.reshape(-1).astype(np.float32)  # shape (512,)
    norm = float(np.linalg.norm(raw_embedding))

    if norm > 1e-8:
        embedding = raw_embedding / norm
    else:
        logger.warning("[!] Warning: Near-zero norm (<1e-8) detected in extracted embedding. Possible corrupt ROI or model bug.")
        embedding = raw_embedding

    return embedding.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 512-dimensional vectors.
    Since both vectors are L2-normalized unit vectors, this is simply the dot product.
    Result is clipped to [-1.0, 1.0] to guard against floating-point precision drift.
    """
    dot = float(np.dot(a, b))
    return float(np.clip(dot, -1.0, 1.0))
