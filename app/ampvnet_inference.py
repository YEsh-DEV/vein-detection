#!/usr/bin/env python3
"""
app/ampvnet_inference.py
------------------------
Authoritative inference module for AMPVNet palm vein embeddings.

Loads models/ampvnet_finetuned.onnx (with automatic fallback to models/ampvnet.onnx)
using ONNX Runtime CPUExecutionProvider (ARM NEON optimized for Raspberry Pi 5).

Applies exact same preprocessing as PyTorch training pipeline:
  1. Grayscale verification & 224x224 bicubic resize
  2. 3-channel replication [I, I, I]
  3. Scale uint8 to float32 [0.0, 1.0]
  4. Symmetric channel normalization (mean=0.5, std=0.5) -> [-1.0, 1.0]
  5. Layout: NCHW (1, 3, 224, 224)
  6. Return L2-normalized 512-D float32 vector with norm validation (~1.0)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Determine project paths
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

FINETUNED_ONNX_PATH = _MODELS_DIR / "ampvnet_finetuned.onnx"
DEFAULT_ONNX_PATH = _MODELS_DIR / "ampvnet.onnx"
EMBEDDING_DIM = 512


class AMPVNetInference:
    """
    Dedicated ONNX Runtime inference engine for AMPVNet 512-D embedding extraction.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_loaded = False
        self.error_detail = ""

        # Priority 1: explicitly passed path
        # Priority 2: finetuned ONNX model
        # Priority 3: default ONNX model
        if model_path is not None:
            candidate_paths = [Path(model_path)]
        else:
            candidate_paths = [FINETUNED_ONNX_PATH, DEFAULT_ONNX_PATH]

        for path in candidate_paths:
            if path.exists():
                self.model_path = path
                break

        if self.model_path is None:
            self.error_detail = (
                f"No ONNX model found at {FINETUNED_ONNX_PATH} or {DEFAULT_ONNX_PATH}."
            )
            logger.warning(f"[ampvnet_inference] {self.error_detail}")
            return

        self._init_session()

    def _init_session(self):
        try:
            import onnxruntime as ort
        except (ImportError, ModuleNotFoundError) as e:
            self.model_loaded = False
            self.error_detail = (
                f"ONNX Runtime package is not installed ({e}). "
                "Please run on Raspberry Pi: pip install onnxruntime"
            )
            logger.error(f"[ampvnet_inference] {self.error_detail}")
            return

        try:
            # Configure session options for CPU execution (optimized for Pi 5 4-core Cortex-A76)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            available_providers = ort.get_available_providers()
            target_providers = (
                ["CPUExecutionProvider"]
                if "CPUExecutionProvider" in available_providers
                else available_providers
            )

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=target_providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model_loaded = True
            logger.info(
                f"[ampvnet_inference] Loaded AMPVNet ONNX model from {self.model_path} "
                f"(provider={target_providers}, input='{self.input_name}', output='{self.output_name}')"
            )
        except Exception as e:
            self.model_loaded = False
            self.error_detail = (
                f"Failed to initialize ONNX session from {self.model_path}: {e}"
            )
            logger.error(f"[ampvnet_inference] {self.error_detail}")

    @staticmethod
    def preprocess_roi(
        roi: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Authoritative preprocessing function for AMPVNet input.
        Strictly replicates PyTorch training/dataset.py:preprocess_image_to_tensor.

        Args:
            roi: uint8 grayscale or BGR numpy array.
            target_size: Target (width, height) tuple, default (224, 224).

        Returns:
            tensor: (1, 3, 224, 224) float32 numpy array in [-1.0, 1.0].
        """
        if roi is None or roi.size == 0:
            raise ValueError("Input ROI is None or empty.")

        # Ensure single-channel 2D grayscale
        if roi.ndim == 3 and roi.shape[2] == 1:
            roi = roi.squeeze(2)
        elif roi.ndim == 3 and roi.shape[2] == 3:
            import cv2
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elif roi.ndim != 2:
            raise ValueError(f"Expected 2D grayscale or 3D image, got shape {roi.shape}")

        # Bicubic resize to target size
        if roi.shape != target_size:
            import cv2
            roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)

        # 3-channel replication [I, I, I]
        ch3 = np.stack([roi, roi, roi], axis=-1)  # (224, 224, 3)

        # Convert uint8 to float32 and scale to [0.0, 1.0]
        norm_img = ch3.astype(np.float32) / 255.0

        # Channel normalization: mean=0.5, std=0.5 -> (x - 0.5) / 0.5 -> maps to [-1.0, 1.0]
        norm_img = (norm_img - 0.5) / 0.5

        # Transpose HWC (224, 224, 3) -> CHW (3, 224, 224)
        tensor = np.transpose(norm_img, (2, 0, 1))

        # Add batch dimension -> (1, 3, 224, 224)
        tensor = np.expand_dims(tensor, axis=0)
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def extract_embedding(self, roi: np.ndarray) -> np.ndarray:
        """
        Runs inference on an input ROI and returns a 512-D L2-normalized float32 vector.

        Args:
            roi: uint8 palm ROI patch (typically 224x224).

        Returns:
            embedding: (512,) float32 unit vector with ||v||_2 ~= 1.0.
        """
        if not self.model_loaded or self.session is None:
            raise RuntimeError(
                f"AMPVNet model not loaded ({self.error_detail}). Cannot extract embedding."
            )

        tensor_input = self.preprocess_roi(roi)

        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor_input})
            raw_emb = outputs[0].reshape(-1).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"ONNX Runtime inference failed: {e}")

        if len(raw_emb) != EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dimension: expected {EMBEDDING_DIM}, got {len(raw_emb)}"
            )

        # Verify and ensure strict unit L2 norm
        norm = float(np.linalg.norm(raw_emb))
        if norm < 1e-8:
            logger.warning("[ampvnet_inference] Near-zero embedding norm detected.")
            return raw_emb

        normalized_emb = (raw_emb / norm).astype(np.float32)
        return normalized_emb


# ---------------------------------------------------------------------------
# Module-Level Singleton & Convenience Wrappers
# ---------------------------------------------------------------------------
_default_engine: Optional[AMPVNetInference] = None


def get_inference_engine() -> AMPVNetInference:
    """Returns the process-wide AMPVNetInference singleton."""
    global _default_engine
    if _default_engine is None:
        _default_engine = AMPVNetInference()
    return _default_engine


def extract_embedding(roi: np.ndarray) -> np.ndarray:
    """Convenience functional wrapper using the process-wide inference engine."""
    engine = get_inference_engine()
    return engine.extract_embedding(roi)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two unit vectors: dot product clipped to [-1.0, 1.0].
    """
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# Export flags matching cnn_extractor.py contract
_init_eng = get_inference_engine()
MODEL_LOADED = _init_eng.model_loaded
MODEL_ERROR_DETAIL = _init_eng.error_detail
