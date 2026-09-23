#!/usr/bin/env python3
"""
constants.py
------------
Consolidated biometric thresholds, algorithm hyperparameters, and canonical
filesystem paths for the Palm Vein Biometric System.
"""

import os

# ---------------------------------------------------------------------------
# Canonical Project Directories & File Paths
# ---------------------------------------------------------------------------
# app/ is one level below project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "palm_vein.db")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
AMPVNET_FINETUNED_ONNX_PATH = os.path.join(MODELS_DIR, "ampvnet_finetuned.onnx")
AMPVNET_DEFAULT_ONNX_PATH = os.path.join(MODELS_DIR, "ampvnet.onnx")
AMPVNET_ONNX_PATH = AMPVNET_FINETUNED_ONNX_PATH if os.path.exists(AMPVNET_FINETUNED_ONNX_PATH) else AMPVNET_DEFAULT_ONNX_PATH
EMBEDDING_DIM = 512

STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")
CAPTURE_DIR = os.path.join(PROJECT_ROOT, "captures")
ROI_DIR = os.path.join(PROJECT_ROOT, "roi_clahe")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SCAN_DIAGNOSTICS_LOG = os.path.join(LOGS_DIR, "scan_diagnostics.jsonl")
CAPTURE_DIAGNOSTICS_LOG = os.path.join(LOGS_DIR, "capture_diagnostics.jsonl")

# ---------------------------------------------------------------------------
# Canonical Structured Capture Failure Codes (Demo Hardening)
# ---------------------------------------------------------------------------
CODE_HAND_TOO_CLOSE = "HAND_TOO_CLOSE"
CODE_HAND_TOO_FAR = "HAND_TOO_FAR"
CODE_HAND_OUTSIDE_FRAME = "HAND_OUTSIDE_FRAME"
CODE_MEDIAPIPE_NO_LANDMARKS = "MEDIAPIPE_NO_LANDMARKS"
CODE_INVALID_LANDMARKS = "INVALID_LANDMARKS"
CODE_VALLEY_EXTRACTION_FAILED = "VALLEY_EXTRACTION_FAILED"
CODE_ROI_EXTRACTION_FAILED = "ROI_EXTRACTION_FAILED"
CODE_QUALITY_LOW_CONTRAST = "QUALITY_LOW_CONTRAST"
CODE_QUALITY_EXCESSIVE_PADDING = "QUALITY_EXCESSIVE_PADDING"
CODE_MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
CODE_CAMERA_ERROR = "CAMERA_ERROR"
CODE_UNKNOWN_PIPELINE_ERROR = "UNKNOWN_PIPELINE_ERROR"

# Temporary Diagnostic & Debug Mode (Stage 4 Phase A4)
# Disabled by default. When enabled, writes raw frames, landmark overlays, and ROIs to debug_frames/
DEBUG_DIAGNOSTICS_MODE = os.environ.get("DEBUG_DIAGNOSTICS_MODE", "false").lower() in ("true", "1", "yes")
DEBUG_FRAMES_DIR = os.path.join(PROJECT_ROOT, "debug_frames")

# Ensure runtime directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Biometric Engine & Threshold Configuration (Phase 8 & 13)
# ---------------------------------------------------------------------------
# Feature flag for engine selection: 'v2' (AMPVNet CNN) or 'legacy' (Gabor+MNHD)
BIOMETRIC_ENGINE = os.environ.get("BIOMETRIC_ENGINE", "v2").lower()

# EXPERIMENTAL MATCH THRESHOLD (Phase 8):
# Derived from Stage-2 validation set EER (0.2226).
# Marked EXPERIMENTAL: can be overridden via MATCH_THRESHOLD environment variable.
EXPERIMENTAL_MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.2226"))
MATCH_THRESHOLD = EXPERIMENTAL_MATCH_THRESHOLD

# Enrollment Validation Bounds & Consistency
ENROLL_CONSISTENCY_THRESHOLD = 0.35
ENROLL_SAMPLE_MIN = 3
ENROLL_SAMPLE_MAX = 6

# Audit & Diagnostics Warning Levels
CROSS_MATCH_WARN_THRESHOLD = 0.45
SELF_MATCH_WARN_THRESHOLD = 0.35

# In-Memory Session Management
ENROLLMENT_CACHE_TTL = 600  # 10 minutes in seconds


# ─── LEGACY (Gabor+MNHD era) — unused by v2 CNN pipeline, kept only because
# app/gabor.py still references them and we're preserving that file for reference ───
L1_THRESHOLD = 1.1500
L1_BYPASS_MAX_TEMPLATES = 80
TOP_K = 80
MAX_DISPLACEMENT = 8
ANGLE_BRACKET = (-4, -2, 0, 2, 4)
ANGLE_EARLY_EXIT = 0.35
BLOCK_SIZE = 32
ROI_SIZE = 256
GABOR_KSIZE = 15
ORIENTATIONS_DEG = (0, 30, 60, 90, 120, 150)

