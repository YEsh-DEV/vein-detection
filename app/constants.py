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
AMPVNET_ONNX_PATH = os.path.join(MODELS_DIR, "ampvnet.onnx")
EMBEDDING_DIM = 512

STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")
CAPTURE_DIR = os.path.join(PROJECT_ROOT, "captures")
ROI_DIR = os.path.join(PROJECT_ROOT, "roi_clahe")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SCAN_DIAGNOSTICS_LOG = os.path.join(LOGS_DIR, "scan_diagnostics.jsonl")

# Ensure runtime directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Biometric Matching Thresholds & Tolerances (v2 CNN Embedding Pipeline)
# ---------------------------------------------------------------------------
# PLACEHOLDER — cosine similarity threshold, NOT calibrated on real data.
# Must be set via a real FAR/FRR/EER sweep (see tools/real_data_analysis.py, which
# needs updating for embeddings — see Step 4 below) using real enrolled users'
# embeddings before this system is used for anything beyond internal testing.
MATCH_THRESHOLD = 0.5

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

