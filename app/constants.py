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
# Biometric Matching Thresholds & Tolerances
# ---------------------------------------------------------------------------
# MATCH_THRESHOLD: Threshold for authentic identification.
# CALIBRATION PENDING — calibrated via empirical EER harness (0.3650 achieves FAR=0.0%, FRR=0.0%).
MATCH_THRESHOLD = 0.3650

# Layer 1 signature pre-filter threshold (Euclidean distance on 64-float (VR+VI)/2 signature)
# Empirical calibration: genuine max=1.1004, impostor min=1.0016, mean=1.4205.
# 1.1500 guarantees 100.0% genuine pass while filtering 93.07% of impostors in RAM.
L1_THRESHOLD = 1.1500

# Edge database bypass: if total enrolled templates <= L1_BYPASS_MAX_TEMPLATES,
# all templates are evaluated directly in Layer 2 to avoid any risk of premature filtering.
L1_BYPASS_MAX_TEMPLATES = 80
TOP_K = 80

# MNHD Translation and Rotation Search Parameters
MAX_DISPLACEMENT = 8
ANGLE_BRACKET = (-4, -2, 0, 2, 4)
ANGLE_EARLY_EXIT = 0.35  # If 0-deg baseline score <= 0.35, bypass remaining angle evaluations

# Enrollment Validation Bounds & Consistency
ENROLL_CONSISTENCY_THRESHOLD = 0.35
ENROLL_SAMPLE_MIN = 3
ENROLL_SAMPLE_MAX = 6

# Audit & Diagnostics Warning Levels
CROSS_MATCH_WARN_THRESHOLD = 0.45
SELF_MATCH_WARN_THRESHOLD = 0.35

# In-Memory Session Management
ENROLLMENT_CACHE_TTL = 600  # 10 minutes in seconds

# Gabor Feature Extraction Hyperparameters
BLOCK_SIZE = 32
ROI_SIZE = 256
GABOR_KSIZE = 15
ORIENTATIONS_DEG = (0, 30, 60, 90, 120, 150)
