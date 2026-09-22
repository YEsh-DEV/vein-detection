#!/usr/bin/env python3
"""
training/scripts/audit_dataset.py
---------------------------------
Phase 1 & Phase 2: Comprehensive automated dataset inventory and data-type audit
for real palm-vein samples.

Performs:
1. File integrity & corrupt image detection.
2. Complete dataset inventory (counts, identities, sessions, hands, resolutions, channels, sizes).
3. Exact duplicate detection (SHA-256 hash).
4. Near-duplicate detection (Normalized cross-correlation / pixel MSE).
5. Exposure & contrast analysis (extreme dark, overexposed, dynamic range).
6. Cross-folder relationship analysis (Raw vs Preprocessed / CLAHE analysis).
7. Machine-readable output to training/data_processed/audit_metadata.json.
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import cv2

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "sample dataset" / "SASH-VPV(Sample)"
OUTPUT_DIR = PROJECT_ROOT / "training" / "data_processed"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_filename_metadata(filepath: Path):
    """
    Parses metadata from filename and directory structure.
    Standard pattern: S<session>_<subject_id>_<hand>_<sample_idx>.png
    e.g. S1_001_L_1.png or S2_005_R_3.png
    """
    stem = filepath.stem
    parts = stem.split("_")
    meta = {
        "raw_stem": stem,
        "session": None,
        "subject_id": None,
        "hand": None,
        "sample_idx": None,
        "parse_success": False
    }

    # Extract subject ID and hand from parent directories if possible
    parent_hand = filepath.parent.name  # "Left" or "Right"
    parent_subject = filepath.parent.parent.name # "001", etc.
    top_folder = filepath.parent.parent.parent.name # "Raw-Session-1", etc.

    if len(parts) >= 4 and parts[0].startswith("S"):
        meta["session"] = parts[0]  # "S1" or "S2"
        meta["subject_id"] = parts[1]
        meta["hand"] = "Left" if parts[2] == "L" else ("Right" if parts[2] == "R" else parts[2])
        meta["sample_idx"] = parts[3]
        meta["parse_success"] = True
    else:
        # Fallback to directory naming
        meta["subject_id"] = parent_subject
        meta["hand"] = parent_hand
        if "Session-1" in top_folder:
            meta["session"] = "S1"
        elif "Session-2" in top_folder:
            meta["session"] = "S2"
        meta["parse_success"] = True

    meta["top_folder"] = top_folder
    meta["parent_subject"] = parent_subject
    meta["parent_hand"] = parent_hand
    return meta


def audit_dataset(data_dir: Path = DEFAULT_DATA_DIR):
    print(f"==================================================")
    print(f"Starting Dataset Audit on: {data_dir}")
    print(f"==================================================")

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    all_files = sorted(list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.bmp")))
    print(f"[+] Total image files discovered: {len(all_files)}")

    hash_map = defaultdict(list)
    images_metadata = []
    corrupt_files = []
    
    unique_subjects = set()
    samples_per_subject = Counter()
    hand_dist = Counter()
    session_dist = Counter()
    res_dist = Counter()
    channels_dist = Counter()
    format_dist = Counter()
    folder_dist = Counter()

    dark_images = []
    bright_images = []
    low_contrast_images = []
    abnormal_dim_images = []

    file_sizes_bytes = []

    # Store image arrays for near-duplicate check per subject
    subject_arrays = defaultdict(list)

    for idx, fpath in enumerate(all_files):
        rel_path = str(fpath.relative_to(PROJECT_ROOT))
        file_size = fpath.stat().st_size
        file_sizes_bytes.append(file_size)

        # 1. Check readability & format
        try:
            with Image.open(fpath) as pil_img:
                pil_img.verify()
            # Reopen for actual data read (verify closes file/marks it unusable)
            with Image.open(fpath) as pil_img:
                width, height = pil_img.size
                pil_mode = pil_img.mode
                img_format = pil_img.format
        except Exception as e:
            corrupt_files.append({"path": rel_path, "error": str(e)})
            continue

        # OpenCV read for numeric statistics
        cv_img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            corrupt_files.append({"path": rel_path, "error": "cv2.imread failed"})
            continue

        channels = 1 if cv_img.ndim == 2 else cv_img.shape[2]

        # Hash check
        file_hash = compute_sha256(fpath)
        hash_map[file_hash].append(rel_path)

        # Metadata parsing
        meta = parse_filename_metadata(fpath)
        sub_id = meta["subject_id"]
        hand = meta["hand"]
        session = meta["session"]
        top_folder = meta["top_folder"]

        unique_subjects.add(sub_id)
        samples_per_subject[sub_id] += 1
        hand_dist[hand] += 1
        session_dist[session] += 1
        res_dist[f"{width}x{height}"] += 1
        channels_dist[f"{channels}ch ({pil_mode})"] += 1
        format_dist[img_format] += 1
        folder_dist[top_folder] += 1

        # Pixel intensity statistics
        gray = cv_img if channels == 1 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))
        p_dark = float(np.mean(gray < 15))
        p_bright = float(np.mean(gray > 240))

        if mean_val < 30 or p_dark > 0.60:
            dark_images.append({"path": rel_path, "mean": mean_val, "dark_pct": p_dark})
        if mean_val > 220 or p_bright > 0.40:
            bright_images.append({"path": rel_path, "mean": mean_val, "bright_pct": p_bright})
        if std_val < 15:
            low_contrast_images.append({"path": rel_path, "std": std_val})

        if (width, height) != (480, 640):
            abnormal_dim_images.append({"path": rel_path, "dimensions": f"{width}x{height}"})

        # Save for near-duplicate and cross-folder inspection
        subject_arrays[(sub_id, hand, session, top_folder)].append({
            "rel_path": rel_path,
            "filename": fpath.name,
            "array": gray,
            "mean": mean_val,
            "std": std_val
        })

        images_metadata.append({
            "path": rel_path,
            "filename": fpath.name,
            "top_folder": top_folder,
            "subject_id": sub_id,
            "hand": hand,
            "session": session,
            "width": width,
            "height": height,
            "channels": channels,
            "pil_mode": pil_mode,
            "file_size": file_size,
            "sha256": file_hash,
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2),
            "min_val": min_val,
            "max_val": max_val,
            "dark_ratio": round(p_dark, 4),
            "bright_ratio": round(p_bright, 4)
        })

    # Exact duplicates
    exact_duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    # Near-duplicates check (MSE < 1.0 or identical captures with trivial noise)
    near_duplicates = []
    for key, items in subject_arrays.items():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.abs(items[i]["array"].astype(float) - items[j]["array"].astype(float))
                mae = float(np.mean(diff))
                mse = float(np.mean(diff ** 2))
                if mse < 2.0:  # extremely high similarity threshold
                    near_duplicates.append({
                        "file1": items[i]["rel_path"],
                        "file2": items[j]["rel_path"],
                        "mae": round(mae, 4),
                        "mse": round(mse, 4)
                    })

    # Cross-Folder Relationship Analysis: Compare Preprocessed vs Raw
    cross_folder_matches = []
    raw_dict = {}
    for item in images_metadata:
        if item["top_folder"] in ["Raw-Session-1", "Raw-Session-2"]:
            # Key by (subject_id, hand, filename)
            raw_dict[(item["subject_id"], item["hand"], item["filename"])] = item

    for item in images_metadata:
        if item["top_folder"] == "Preprocessed Data":
            key = (item["subject_id"], item["hand"], item["filename"])
            if key in raw_dict:
                raw_item = raw_dict[key]
                cross_folder_matches.append({
                    "raw_path": raw_item["path"],
                    "prep_path": item["path"],
                    "subject": item["subject_id"],
                    "hand": item["hand"],
                    "filename": item["filename"]
                })

    # Summary Statistics
    total_imgs = len(all_files)
    total_valid = len(images_metadata)
    total_corrupt = len(corrupt_files)
    avg_size_kb = (sum(file_sizes_bytes) / len(file_sizes_bytes) / 1024) if file_sizes_bytes else 0.0
    total_size_mb = sum(file_sizes_bytes) / (1024 * 1024) if file_sizes_bytes else 0.0

    report_dict = {
        "inventory": {
            "total_images_found": total_imgs,
            "valid_images": total_valid,
            "corrupt_images": total_corrupt,
            "corrupt_details": corrupt_files,
            "unique_subjects_count": len(unique_subjects),
            "unique_subjects": sorted(list(unique_subjects)),
            "samples_per_subject": dict(sorted(samples_per_subject.items())),
            "hand_distribution": dict(hand_dist),
            "session_distribution": dict(session_dist),
            "folder_distribution": dict(folder_dist),
            "resolution_distribution": dict(res_dist),
            "channels_distribution": dict(channels_dist),
            "format_distribution": dict(format_dist),
            "file_sizes": {
                "min_bytes": min(file_sizes_bytes) if file_sizes_bytes else 0,
                "max_bytes": max(file_sizes_bytes) if file_sizes_bytes else 0,
                "mean_kb": round(avg_size_kb, 2),
                "total_mb": round(total_size_mb, 2)
            },
            "abnormal_dimensions_count": len(abnormal_dim_images),
            "abnormal_dimensions": abnormal_dim_images,
            "exact_duplicate_groups_count": len(exact_duplicates),
            "exact_duplicates": exact_duplicates,
            "near_duplicates_count": len(near_duplicates),
            "near_duplicates": near_duplicates,
            "dark_images_count": len(dark_images),
            "bright_images_count": len(bright_images),
            "low_contrast_images_count": len(low_contrast_images),
            "cross_folder_matches_count": len(cross_folder_matches),
            "cross_folder_matches": cross_folder_matches
        },
        "all_images": images_metadata
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "audit_metadata.json"
    with open(out_file, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n================ AUDIT SUMMARY ================")
    print(f"Total images audited: {total_valid} / {total_imgs}")
    print(f"Corrupt images: {total_corrupt}")
    print(f"Unique subjects: {len(unique_subjects)}")
    print(f"Hand distribution: {dict(hand_dist)}")
    print(f"Session distribution: {dict(session_dist)}")
    print(f"Folder distribution: {dict(folder_dist)}")
    print(f"Resolution distribution: {dict(res_dist)}")
    print(f"Channels distribution: {dict(channels_dist)}")
    print(f"Exact duplicates: {len(exact_duplicates)}")
    print(f"Near duplicates (MSE < 2.0): {len(near_duplicates)}")
    print(f"Dark images: {len(dark_images)}")
    print(f"Bright images: {len(bright_images)}")
    print(f"Low contrast images: {len(low_contrast_images)}")
    print(f"Preprocessed <-> Raw matching captures: {len(cross_folder_matches)}")
    print(f"Report saved to: {out_file}")
    print(f"===============================================\n")
    return report_dict


if __name__ == "__main__":
    audit_dataset()
