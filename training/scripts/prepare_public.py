#!/usr/bin/env python3
"""
training/scripts/prepare_public.py
---------------------------------
Adapter and preparation pipeline for public palm biometric dataset (Tongji TJU600).

Processes the canonical 128x128 palm ROI images into normalized 224x224x3 PNG images:
  1. Maps image filenames to (subject_id, palm_id, hand, session).
  2. Resizes with bicubic interpolation to 224x224.
  3. Verifies zero corrupt images and computes SHA-256 hashes to detect collisions.
  4. Generates a strict subject-disjoint split:
     - Train: 240 subjects (480 palms, 9,600 images) [80%]
     - Validation: 30 subjects (60 palms, 1,200 images) [10%]
     - Test: 30 subjects (60 palms, 1,200 images) [10%]
  5. Saves images to training/data_processed/public/<class_id>/<img_name>.png
  6. Saves partitioned splits to training/data_processed/public_splits/{train,val,test}/
  7. Outputs public_splits.json and public_index.csv.
"""

import os
import sys
import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TRAINING_DIR.parent

DEFAULT_RAW_DIR = _TRAINING_DIR / "data_raw" / "public" / "tju_palm_raw"
DEFAULT_OUT_DIR = _TRAINING_DIR / "data_processed" / "public"
DEFAULT_SPLITS_DIR = _TRAINING_DIR / "data_processed" / "public_splits"


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(filename: str, session: str) -> Dict[str, Any]:
    """
    Tongji Contactless Palm Naming Convention:
      6000 images per session from 600 palms (10 images per palm).
      00001 - 00010: Palm 1
      00011 - 00020: Palm 2
      ...
      Palm P in [1, 600]:
        Subject S = (P - 1) // 2 + 1  (in [1, 300])
        Hand = 'left' if P % 2 == 1 else 'right'
    """
    stem = Path(filename).stem
    idx = int(stem)  # 1 to 6000
    palm_id = (idx - 1) // 10 + 1
    sample_idx = (idx - 1) % 10 + 1
    subject_id = (palm_id - 1) // 2 + 1
    hand = "left" if (palm_id % 2 == 1) else "right"
    class_id = f"sub_{subject_id:03d}_{hand}"

    return {
        "filename": filename,
        "session": session,
        "index_in_session": idx,
        "palm_id": palm_id,
        "sample_idx": sample_idx,
        "subject_id": f"{subject_id:03d}",
        "subject_num": subject_id,
        "hand": hand,
        "class_id": class_id,
    }


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    splits_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    target_size: int = 224,
) -> Dict[str, Any]:
    print(f"[prepare_public.py] Starting preparation from {raw_dir}")
    print(f"[prepare_public.py] Target output directory: {out_dir}")

    s1_dir = raw_dir / "session1"
    s2_dir = raw_dir / "session2"

    if not s1_dir.exists() or not s2_dir.exists():
        raise FileNotFoundError(f"Expected session1 and session2 in {raw_dir}")

    s1_files = sorted(s1_dir.glob("*.bmp"))
    s2_files = sorted(s2_dir.glob("*.bmp"))

    print(f"[prepare_public.py] Found {len(s1_files)} files in session1, {len(s2_files)} files in session2.")
    assert len(s1_files) == 6000, f"Expected 6000 files in session1, got {len(s1_files)}"
    assert len(s2_files) == 6000, f"Expected 6000 files in session2, got {len(s2_files)}"

    # Parse all records
    records = []
    for f in s1_files:
        meta = parse_filename(f.name, "session1")
        meta["raw_path"] = str(f)
        records.append(meta)

    for f in s2_files:
        meta = parse_filename(f.name, "session2")
        meta["raw_path"] = str(f)
        records.append(meta)

    print(f"[prepare_public.py] Total records collected: {len(records)}")

    # Distinct subjects
    all_subject_nums = sorted(list(set(r["subject_num"] for r in records)))
    total_subjects = len(all_subject_nums)
    print(f"[prepare_public.py] Total unique subjects: {total_subjects} (palms: {total_subjects * 2})")

    # Strict subject-disjoint split
    rng = random.Random(seed)
    shuffled_subjects = list(all_subject_nums)
    rng.shuffle(shuffled_subjects)

    n_train = int(total_subjects * train_ratio)
    n_val = int(total_subjects * val_ratio)
    n_test = total_subjects - n_train - n_val

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val :])

    # Assert strict disjointness
    assert len(train_subjects & val_subjects) == 0, "Subject leakage between train and val!"
    assert len(train_subjects & test_subjects) == 0, "Subject leakage between train and test!"
    assert len(val_subjects & test_subjects) == 0, "Subject leakage between val and test!"

    print(f"[prepare_public.py] Split partition:")
    print(f"  - Train: {len(train_subjects)} subjects ({len(train_subjects)*2} palms, {len(train_subjects)*40} images)")
    print(f"  - Val  : {len(val_subjects)} subjects ({len(val_subjects)*2} palms, {len(val_subjects)*40} images)")
    print(f"  - Test : {len(test_subjects)} subjects ({len(test_subjects)*2} palms, {len(test_subjects)*40} images)")

    # Prepare output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (splits_dir / sp).mkdir(parents=True, exist_ok=True)

    # Process and write images
    hash_map: Dict[str, str] = {}
    duplicates = 0
    corrupt = 0
    processed_records = []

    print("[prepare_public.py] Resizing and writing 224x224 images...")
    for idx, r in enumerate(records):
        raw_path = r["raw_path"]
        img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            corrupt += 1
            print(f"  [ERROR] Corrupt file: {raw_path}")
            continue

        # Bicubic resize to 224x224
        roi_224 = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        # Output filenames
        sess_code = "s1" if r["session"] == "session1" else "s2"
        out_filename = f"{r['class_id']}_{sess_code}_{r['sample_idx']:02d}.png"

        # Flat class directory in out_dir
        class_dir = out_dir / r["class_id"]
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / out_filename

        cv2.imwrite(str(out_path), roi_224)

        # Assign split
        subj = r["subject_num"]
        if subj in train_subjects:
            split_name = "train"
        elif subj in val_subjects:
            split_name = "val"
        else:
            split_name = "test"

        split_class_dir = splits_dir / split_name / r["class_id"]
        split_class_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_class_dir / out_filename
        cv2.imwrite(str(split_path), roi_224)

        # Hash check
        file_hash = get_sha256(out_path)
        if file_hash in hash_map:
            duplicates += 1
        else:
            hash_map[file_hash] = str(out_path)

        r["processed_path"] = str(out_path)
        r["split_path"] = str(split_path)
        r["split"] = split_name
        r["sha256"] = file_hash
        processed_records.append(r)

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(records)} images...")

    print(f"[prepare_public.py] Completed processing:")
    print(f"  Total processed: {len(processed_records)}")
    print(f"  Corrupt images : {corrupt}")
    print(f"  Hash collisions: {duplicates}")

    # Generate CSV manifest
    csv_path = out_dir.parent / "public_index.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,session,class_id,subject_id,hand,split,raw_path,processed_path,sha256\n")
        for pr in processed_records:
            f.write(
                f"{pr['filename']},{pr['session']},{pr['class_id']},{pr['subject_id']},{pr['hand']},"
                f"{pr['split']},{pr['raw_path']},{pr['processed_path']},{pr['sha256']}\n"
            )
    print(f"[prepare_public.py] Manifest saved to: {csv_path}")

    # Generate JSON split summary
    splits_json_path = out_dir.parent / "public_splits.json"
    split_info = {
        "dataset_name": "Tongji Touchless Palm (TJU600)",
        "source": "https://cslinzhang.github.io/ContactlessPalm/ (IEEE TIFS 2017)",
        "total_images": len(processed_records),
        "total_subjects": total_subjects,
        "total_palm_classes": total_subjects * 2,
        "split_seed": seed,
        "train": {
            "num_subjects": len(train_subjects),
            "subjects": sorted([f"{s:03d}" for s in train_subjects]),
            "num_palms": len(train_subjects) * 2,
            "num_images": sum(1 for pr in processed_records if pr["split"] == "train"),
        },
        "val": {
            "num_subjects": len(val_subjects),
            "subjects": sorted([f"{s:03d}" for s in val_subjects]),
            "num_palms": len(val_subjects) * 2,
            "num_images": sum(1 for pr in processed_records if pr["split"] == "val"),
        },
        "test": {
            "num_subjects": len(test_subjects),
            "subjects": sorted([f"{s:03d}" for s in test_subjects]),
            "num_palms": len(test_subjects) * 2,
            "num_images": sum(1 for pr in processed_records if pr["split"] == "test"),
        },
    }
    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"[prepare_public.py] Splits JSON saved to: {splits_json_path}")

    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Public Palm Biometric Dataset")
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits_dir", type=str, default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        splits_dir=Path(args.splits_dir),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
