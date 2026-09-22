#!/usr/bin/env python3
"""
training/scripts/prepare_own.py
-------------------------------
Phase 6, 7 & 8: Clean, Reproducible Real-Data Preparation Pipeline.

Reads raw NIR camera captures and preprocessed full-frame images from:
  sample dataset/SASH-VPV(Sample)/
(ORIGINAL DATA IS READ-ONLY AND NEVER MODIFIED, RENAMED, OR DELETED).

Executes the exact Raspberry Pi inference pipeline (Section 5 / app/server.py):
  1. Contrast stretching: cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
  2. MediaPipe Hand Landmark Detection (21 skeletal joints)
  3. Knuckle Valley Extraction: Pv1 (radial) and Pv2 (ulnar)
  4. Scalable Dual-Signal ROI extraction (beta = 1.6, Ma et al. 2017):
     - Affine knuckle-axis rotation alignment
     - Distance scaling: d_ROI = 1.6 * dist(Pv1, Pv2)
     - Symmetric edge-safe padding (1:1 aspect ratio guarantee)
     - Resampling to canonical 224x224
  5. Vein enhancement: bilateralFilter + CLAHE (app/mediapipe_img.py: enhance_roi_vessels)
  6. Saves 224x224 uint8 single-channel PNG to:
     - training/data_processed/own/<subject_id>/<filename>.png (for standard PalmVeinDataset)
     - training/data_processed/own_splits/<split>/<subject_id>/<filename>.png (for split isolation)
  7. Outputs machine-readable manifest: training/data_processed/dataset_index.csv.
"""

import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mediapipe_img import (
    build_landmarker,
    detect_hand_landmarks,
    extract_valleys_from_landmarks,
    extract_ma2017_scaled_roi,
    enhance_roi_vessels,
    segment_hand,
)

RAW_DATA_DIR = PROJECT_ROOT / "sample dataset" / "SASH-VPV(Sample)"
PROCESSED_DIR = PROJECT_ROOT / "training" / "data_processed"
PROCESSED_OWN_DIR = PROCESSED_DIR / "own"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "own_splits"
INDEX_CSV = PROCESSED_DIR / "dataset_index.csv"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset(seed: int = 42):
    print("==================================================")
    print("STARTING DATASET PREPARATION PIPELINE")
    print("==================================================")
    print(f"Source Directory (Read-Only): {RAW_DATA_DIR}")
    print(f"Target Processed Directory:  {PROCESSED_OWN_DIR}")

    # Verify input prerequisites
    splits_file = PROCESSED_DIR / "splits.json"
    quality_file = PROCESSED_DIR / "quality_report.json"
    audit_file = PROCESSED_DIR / "audit_metadata.json"

    if not splits_file.exists() or not quality_file.exists() or not audit_file.exists():
        print("[!] Generating prerequisites...")
        from audit_dataset import audit_dataset
        from visualize_dataset import run_visual_audit
        from verify_splits import generate_and_verify_splits
        audit_dataset()
        run_visual_audit()
        generate_and_verify_splits(seed=seed)

    with open(splits_file) as f:
        splits_data = json.load(f)
    with open(quality_file) as f:
        quality_data = json.load(f)
    with open(audit_file) as f:
        audit_data = json.load(f)

    # Build lookup map: path -> split ('train', 'validation', 'test')
    path_to_split = {}
    for split_name, paths in splits_data["split_mapping"].items():
        for p in paths:
            path_to_split[p] = split_name

    # Build lookup map: path -> quality record
    path_to_quality = {r["path"]: r for r in quality_data["records"]}

    landmarker = build_landmarker(str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    clahe_assist = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    PROCESSED_OWN_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    processed_count = 0
    skipped_count = 0

    # Process all usable items in path_to_split
    total_to_process = len(path_to_split)
    print(f"[+] Total usable samples to process into 224x224 ROIs: {total_to_process}")

    for rel_path, split_name in sorted(path_to_split.items()):
        source_path = PROJECT_ROOT / rel_path
        if not source_path.exists():
            print(f"[!] Warning: source file not found: {source_path}")
            skipped_count += 1
            continue

        q_rec = path_to_quality.get(rel_path, {})
        sub_id = q_rec.get("subject_id", source_path.parent.parent.name)
        hand = q_rec.get("hand", source_path.parent.name)
        session = q_rec.get("session", "S1")
        status = q_rec.get("status", "GOOD")
        metrics = q_rec.get("metrics", {})

        # Load raw grayscale image
        gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[!] Warning: cv2.imread failed on {source_path}")
            skipped_count += 1
            continue

        # Progressive landmark detection cascade: stretched -> raw gray -> CLAHE assist
        stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        landmarks = None
        for candidate_img in [stretched, gray, clahe_assist.apply(gray)]:
            try:
                landmarks = detect_hand_landmarks(candidate_img, landmarker)
                if landmarks and len(landmarks) == 21:
                    break
            except Exception:
                continue

        if landmarks is None or len(landmarks) < 21:
            print(f"[!] Failed landmark detection on {rel_path}")
            skipped_count += 1
            continue

        # Inference Pipeline Step 3: Knuckle valley anchors Pv1 & Pv2
        try:
            pv1, pv2 = extract_valleys_from_landmarks(landmarks)
            hand_mask = segment_hand(stretched)
        except Exception as e:
            print(f"[!] Failed valley extraction on {rel_path}: {e}")
            skipped_count += 1
            continue

        # Inference Pipeline Step 4: Scalable Dual-Signal ROI extraction (beta = 1.6, target 224x224)
        try:
            roi_224, bbox, _ = extract_ma2017_scaled_roi(
                stretched, pv1, pv2, hand_mask,
                target_size=224, scale_factor=1.6, offset_factor=0.35,
                landmarks_px=landmarks
            )
        except Exception as e:
            print(f"[!] Failed ROI extraction on {rel_path}: {e}")
            skipped_count += 1
            continue

        # Inference Pipeline Step 5: Bilateral filter + CLAHE vein enhancement
        enhanced_roi = enhance_roi_vessels(roi_224)

        # File naming & storage
        # Canonical filename: <session>_<subject>_<hand>_<stem>.png
        out_stem = f"{session}_{sub_id}_{hand}_{source_path.stem}"
        out_fname = f"{out_stem}.png"

        # 1. Store in training/data_processed/own/<subject_id>/
        sub_dir = PROCESSED_OWN_DIR / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path_own = sub_dir / out_fname
        cv2.imwrite(str(out_path_own), enhanced_roi)

        # 2. Store in training/data_processed/own_splits/<split>/<subject_id>/
        split_sub_dir = PROCESSED_SPLITS_DIR / split_name / sub_id
        split_sub_dir.mkdir(parents=True, exist_ok=True)
        out_path_split = split_sub_dir / out_fname
        cv2.imwrite(str(out_path_split), enhanced_roi)

        # Hash of output ROI
        roi_sha256 = compute_sha256(out_path_own)

        manifest_rows.append({
            "subject_id": sub_id,
            "hand": hand,
            "session": session,
            "split": split_name,
            "quality_status": status,
            "pad_pct": metrics.get("pad_pct", 0.0),
            "knuckle_dist_px": metrics.get("knuckle_dist", 0.0),
            "source_raw_path": rel_path,
            "processed_own_path": str(out_path_own.relative_to(PROJECT_ROOT)),
            "processed_split_path": str(out_path_split.relative_to(PROJECT_ROOT)),
            "roi_sha256": roi_sha256,
            "roi_shape": "224x224x1",
            "model_input_spec": "224x224x3 replicated float32 normalized [-1, 1]"
        })
        processed_count += 1

    # Write Manifest CSV
    fieldnames = [
        "subject_id", "hand", "session", "split", "quality_status",
        "pad_pct", "knuckle_dist_px", "source_raw_path",
        "processed_own_path", "processed_split_path", "roi_sha256",
        "roi_shape", "model_input_spec"
    ]
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n================ PREPARATION SUMMARY ================")
    print(f"Total processed 224x224 ROIs: {processed_count} / {total_to_process}")
    print(f"Skipped: {skipped_count}")
    print(f"Output directories created:")
    print(f"  - {PROCESSED_OWN_DIR} (Flat subject layout for PalmVeinDataset)")
    print(f"  - {PROCESSED_SPLITS_DIR} (Partitioned train/val/test layout)")
    print(f"Manifest CSV saved: {INDEX_CSV}")
    print(f"Source dataset remained untouched: {RAW_DATA_DIR} ✓")
    print(f"=====================================================\n")

    return manifest_rows


if __name__ == "__main__":
    prepare_dataset()
