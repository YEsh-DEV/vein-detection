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

# ---------------------------------------------------------------------------
# Camera & Optical Pipeline Tuning Constants (NoIR Sensor Calibration)
# ---------------------------------------------------------------------------
# Candidate exposure and gain settings for Raspberry Pi OV5647 NoIR with 850nm IR LEDs.
DEFAULT_EXPOSURE_US = int(os.environ.get("CAMERA_EXPOSURE_US", "2500"))
DEFAULT_ANALOGUE_GAIN = float(os.environ.get("CAMERA_GAIN", "1.0"))

MIN_EXPOSURE_US = 1000
MAX_EXPOSURE_US = 12000
MIN_ANALOGUE_GAIN = 1.0
MAX_ANALOGUE_GAIN = 1.8

# Bounded calibration search ranges for Picamera2
EXPOSURE_SEARCH_BOUNDS_US = (MIN_EXPOSURE_US, MAX_EXPOSURE_US)
GAIN_SEARCH_BOUNDS = (MIN_ANALOGUE_GAIN, MAX_ANALOGUE_GAIN)
EXPOSURE_SEARCH_STEPS_US = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000]
GAIN_SEARCH_STEPS = [1.0, 1.2, 1.4, 1.6, 1.8]

# Candidate exposure sweep pairs for empirical hardware validation
EXPOSURE_SWEEP_PRESETS = [
    (1000, 1.0),
    (2000, 1.0),
    (3000, 1.0),
    (4000, 1.0),
    (5000, 1.0),
    (6000, 1.0),
    (8000, 1.0),
    (10000, 1.0),
    (12000, 1.0),
]
CANDIDATE_EXPOSURE_SWEEPS = EXPOSURE_SWEEP_PRESETS

NIR_WEIGHT_RED = 0.50
NIR_WEIGHT_GREEN = 0.25
NIR_WEIGHT_BLUE = 0.25

# Configurable NIR channel extraction method:
# 'weighted_nir' (0.50R + 0.25G + 0.25B), 'r_channel', 'g_channel', 'b_channel', 'rec601_gray', 'equal_nir'
NIR_EXTRACTION_METHOD = os.environ.get("NIR_EXTRACTION_METHOD", "weighted_nir").lower()

# Display-Only Percentile-Clipped Normalization & Gentle CLAHE
DISPLAY_PERCENTILE_LOW = 2.0
DISPLAY_PERCENTILE_HIGH = 88.0
DISPLAY_CLAHE_CLIP = 2.5
DISPLAY_CLAHE_GRID = (8, 8)

# Quality gate bounds for palm illumination
TARGET_PALM_MEAN_MIN = 80.0
TARGET_PALM_MEAN_MAX = 160.0
MIN_CONTRAST_STD = 12.0
MAX_IR_SATURATION_PCT = 3.0
MIN_SHARPNESS_LAPLACIAN = 10.0

# Best-frame burst selection parameters
BURST_CAPTURE_FRAMES = 5
BURST_FRAME_INTERVAL_S = 0.05

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

# CALIBRATED DEMO MATCH THRESHOLD (Phase 13 — Stage-5 Threshold Recalibration):
# Previous value: 0.2226 (EER-derived from Stage-2 synthetic validation split).
# Problem at 0.2226: FAR = 41.32% (2159/5225 impostor pairs accepted) on real NIR hardware.
#
# Stage-5 recalibration used the global ROC curve from 131 genuine + 5225 impostor pairs
# (38 palm identities, Raspberry Pi 850nm NIR sensor).
#
# Threshold sweep (global):
#   0.2226 → FAR=41.32%, TAR=92.37%
#   0.40   → FAR=17.05%, TAR=83.97%   (near global EER @ 0.41)
#   0.45   → FAR=11.98%, TAR=80.15%   ← SELECTED
#   0.50   → FAR= 7.94%, TAR=74.81%
#   0.63   → FAR= 1.00%, TAR=63.36%
#
# Selection rationale:
#   Validation-split EER is at threshold≈0.437 (N=18 genuine, N=135 impostor).
#   0.45 is just above that, providing substantial FAR reduction (41.32%→11.98%)
#   while retaining demo-viable TAR (80.15%).
#   Higher thresholds (0.50+) would reject too many genuine users in live demo.
#
# Limitation: Dataset is DATA-LIMITED (131 genuine pairs). Numbers are indicative only.
# System remains CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED).
#
# Can be overridden at runtime via MATCH_THRESHOLD environment variable.
EXPERIMENTAL_MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.45"))
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

