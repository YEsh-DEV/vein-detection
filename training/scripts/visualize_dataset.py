#!/usr/bin/env python3
"""
training/scripts/visualize_dataset.py
-------------------------------------
Phase 3: Visual Data Quality Audit & Visualization Utility.

Generates:
1. Landmark overlay & Scalable ROI (beta=1.6) visual comparisons.
2. Multi-sample Contact Sheets for GOOD, QUESTIONABLE, and INVALID captures.
3. Quantitative Quality Report saved to training/data_processed/quality_report.json.

Classification Criteria:
- GOOD:
    * 21 MediaPipe landmarks detected.
    * Knuckle distance d >= 40px (hand sufficiently close to camera).
    * Scalable ROI boundary padding <= 15% (hand well-centered in frame).
    * Dynamic range std >= 20.0 (good vascular contrast).
- QUESTIONABLE:
    * Landmarks detected, but ROI requires 15% - 40% boundary padding (hand near frame edge), OR
    * Hand distance small (knuckle distance d between 25px and 40px), OR
    * Moderate contrast (std between 12.0 and 20.0).
- INVALID:
    * MediaPipe failed to detect hand landmarks (due to fingers/knuckles cropped out of frame,
      hand pressed against lens/boundary touching all 4 edges, or extreme occlusion), OR
    * ROI boundary padding > 40% (extreme edge truncation), OR
    * Knuckle distance d < 25px (hand way too far).
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mediapipe_img import (
    build_landmarker,
    detect_hand_landmarks,
    extract_valleys_from_landmarks,
    extract_ma2017_scaled_roi,
)

PREVIEWS_DIR = PROJECT_ROOT / "training" / "logs" / "previews"
PROCESSED_DIR = PROJECT_ROOT / "training" / "data_processed"


def draw_landmarks_and_roi(
    gray_img: np.ndarray,
    landmarks_px: list,
    pv1: tuple,
    pv2: tuple,
    bbox: tuple,
    status: str
) -> np.ndarray:
    """
    Renders 21 skeletal landmarks, knuckle valley anchors, palm axis vector,
    and oriented ROI bounding box onto an RGB canvas.
    """
    canvas = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    h, w = gray_img.shape

    # Hand connections (MediaPipe skeleton)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    if landmarks_px and len(landmarks_px) == 21:
        # Draw bones
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = tuple(int(x) for x in landmarks_px[p1_idx])
            pt2 = tuple(int(x) for x in landmarks_px[p2_idx])
            cv2.line(canvas, pt1, pt2, (0, 200, 100), 2, cv2.LINE_AA)

        # Draw joints
        for idx, (x, y) in enumerate(landmarks_px):
            color = (255, 100, 0) if idx in [0, 5, 9, 13, 17] else (0, 255, 255)
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

        # Draw Pv1 & Pv2 (Knuckle Valley Anchors)
        if pv1 and pv2:
            p1_pt = tuple(int(x) for x in pv1)
            p2_pt = tuple(int(x) for x in pv2)
            cv2.circle(canvas, p1_pt, 6, (0, 0, 255), -1, cv2.LINE_AA) # Red: Pv1
            cv2.circle(canvas, p2_pt, 6, (255, 0, 0), -1, cv2.LINE_AA) # Blue: Pv2
            cv2.line(canvas, p1_pt, p2_pt, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw middle MCP to wrist axis
            wrist = tuple(int(x) for x in landmarks_px[0])
            m_mcp = tuple(int(x) for x in landmarks_px[9])
            cv2.line(canvas, wrist, m_mcp, (200, 0, 255), 2, cv2.LINE_AA)

    # Status tag
    status_color = (0, 200, 0) if status == "GOOD" else ((0, 165, 255) if status == "QUESTIONABLE" else (0, 0, 255))
    cv2.putText(canvas, f"STATUS: {status}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

    return canvas


def classify_sample(
    gray_img: np.ndarray,
    landmarks_px: list,
    pv1: tuple,
    pv2: tuple,
    bbox: tuple,
    roi_img: np.ndarray = None
) -> tuple:
    """
    Evaluates quality and returns (status, reason, metrics).
    """
    h, w = gray_img.shape
    std_val = float(np.std(gray_img))

    if landmarks_px is None or len(landmarks_px) != 21:
        # Check boundary collision
        _, thresh = cv2.threshold(cv2.GaussianBlur(gray_img, (15, 15), 0), 40, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reason = "MediaPipe landmark detection failed"
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            boundary_touch = (bx <= 5) + (by <= 5) + ((bx + bw) >= (w - 5)) + ((by + bh) >= (h - 5))
            if boundary_touch >= 3:
                reason += f" (Hand cropped: touches {boundary_touch}/4 frame edges, area={cv2.contourArea(c):.0f})"
            else:
                reason += " (Insufficient palm contrast / pose distortion)"
        return "INVALID", reason, {"std": std_val, "pad_pct": 1.0, "knuckle_dist": 0.0}

    # Landmark metrics
    dx = pv2[0] - pv1[0]
    dy = pv2[1] - pv1[1]
    knuckle_dist = float(np.hypot(dx, dy))

    # Boundary padding in bbox (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    L = max(x2 - x1, y2 - y1)
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    total_pad_px = pad_left + pad_top + pad_right + pad_bottom
    pad_pct = float(total_pad_px / (2 * L)) if L > 0 else 0.0

    metrics = {
        "std": round(std_val, 2),
        "pad_pct": round(pad_pct, 4),
        "knuckle_dist": round(knuckle_dist, 2),
        "roi_size_L": L
    }

    if knuckle_dist < 25.0:
        return "INVALID", f"Hand too far from camera (knuckle_dist={knuckle_dist:.1f}px < 25px)", metrics

    if pad_pct > 0.40:
        return "INVALID", f"Severe ROI boundary clipping ({pad_pct*100:.1f}% padded)", metrics

    if pad_pct > 0.15 or std_val < 20.0 or knuckle_dist < 40.0:
        reasons = []
        if pad_pct > 0.15:
            reasons.append(f"Moderate boundary padding ({pad_pct*100:.1f}%)")
        if std_val < 20.0:
            reasons.append(f"Lower contrast (std={std_val:.1f})")
        if knuckle_dist < 40.0:
            reasons.append(f"Small palm scale (knuckle_dist={knuckle_dist:.1f}px)")
        return "QUESTIONABLE", "; ".join(reasons), metrics

    return "GOOD", "High quality presentation: valid landmarks, centered scalable ROI, robust contrast", metrics


def run_visual_audit():
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    audit_file = PROCESSED_DIR / "audit_metadata.json"
    if not audit_file.exists():
        print(f"[!] Running audit_dataset.py first...")
        from audit_dataset import audit_dataset
        audit_dataset()

    with open(audit_file) as f:
        imgs = json.load(f)["all_images"]

    print(f"[+] Initializing MediaPipe HandLandmarker for visual quality audit...")
    landmarker = build_landmarker(str(PROJECT_ROOT / "models" / "hand_landmarker.task"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    classified_results = []
    status_counts = Counter()

    samples_by_status = defaultdict(list)

    print(f"[+] Processing all {len(imgs)} images through landmark detection and ROI cropping...")
    for idx, im in enumerate(imgs):
        p = PROJECT_ROOT / im["path"]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape

        landmarks = None
        pv1, pv2 = None, None
        bbox = None
        roi_norm = None

        # 1. Attempt detection on original image
        try:
            landmarks = detect_hand_landmarks(gray, landmarker)
        except Exception:
            # 2. Fallback: attempt detection on CLAHE enhanced image
            try:
                c_gray = clahe.apply(gray)
                landmarks = detect_hand_landmarks(c_gray, landmarker)
            except Exception:
                landmarks = None

        if landmarks is not None:
            try:
                pv1, pv2 = extract_valleys_from_landmarks(landmarks)
                roi_norm, bbox, _ = extract_ma2017_scaled_roi(
                    gray, pv1, pv2, target_size=224, scale_factor=1.6, landmarks_px=landmarks
                )
            except Exception as e:
                landmarks = None

        status, reason, metrics = classify_sample(gray, landmarks, pv1, pv2, bbox, roi_norm)
        status_counts[status] += 1

        record = {
            "path": im["path"],
            "filename": im["filename"],
            "subject_id": im["subject_id"],
            "hand": im["hand"],
            "session": im["session"],
            "top_folder": im["top_folder"],
            "status": status,
            "reason": reason,
            "metrics": metrics,
            "has_landmarks": landmarks is not None,
            "has_roi": roi_norm is not None
        }
        classified_results.append(record)

        # Retain for preview contact sheets
        samples_by_status[status].append({
            "record": record,
            "gray": gray,
            "landmarks": landmarks,
            "pv1": pv1,
            "pv2": pv2,
            "bbox": bbox,
            "roi": roi_norm
        })

    # Save Quality Report JSON
    quality_report = {
        "total_audited": len(classified_results),
        "status_summary": dict(status_counts),
        "status_percentages": {
            k: round(v / len(classified_results) * 100, 2) for k, v in status_counts.items()
        },
        "records": classified_results
    }
    report_path = PROCESSED_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    print(f"\n[+] Quality Report saved to: {report_path}")
    print(f"    Summary: {dict(status_counts)}")

    # Generate Contact Sheets
    print(f"[+] Generating contact sheets in: {PREVIEWS_DIR}")
    for status_key in ["GOOD", "QUESTIONABLE", "INVALID"]:
        items = samples_by_status[status_key]
        if not items:
            continue

        # Pick up to 8 representative diverse samples (different subjects)
        selected = []
        seen_subs = set()
        for item in items:
            sub = item["record"]["subject_id"]
            if sub not in seen_subs:
                selected.append(item)
                seen_subs.add(sub)
            if len(selected) >= 8:
                break
        # If fewer than 8 subjects, fill with remaining items
        if len(selected) < 8:
            for item in items:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= 8:
                    break

        # Compose a 2-column or 4x2 grid of preview pairs (Overlay + ROI)
        grid_rows = []
        for pair_idx, item in enumerate(selected):
            gray = item["gray"]
            rec = item["record"]
            overlay = draw_landmarks_and_roi(
                gray, item["landmarks"], item["pv1"], item["pv2"], item["bbox"], rec["status"]
            )
            # Resize overlay for contact sheet (e.g. 240 x 320)
            overlay_small = cv2.resize(overlay, (240, 320), interpolation=cv2.INTER_AREA)

            if item["roi"] is not None:
                roi_vis = cv2.cvtColor(item["roi"], cv2.COLOR_GRAY2BGR)
                roi_small = cv2.resize(roi_vis, (240, 240), interpolation=cv2.INTER_AREA)
                # Pad roi to match height 320 (add black border 40 top, 40 bottom)
                roi_padded = cv2.copyMakeBorder(roi_small, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
            else:
                roi_padded = np.zeros((320, 240, 3), dtype=np.uint8)
                cv2.putText(roi_padded, "NO ROI", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Combine side by side: [Overlay | ROI] -> (320, 480)
            pair_card = np.hstack([overlay_small, roi_padded])
            # Add subject & file tag
            tag = f"Sub:{rec['subject_id']} {rec['hand']} {rec['session']} ({rec['top_folder'][:5]})"
            cv2.putText(pair_card, tag, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            grid_rows.append(pair_card)

        # Arrange in grid (e.g. 4 rows of 2 pairs if 8, or vertical stack)
        cols = 2
        rows = (len(grid_rows) + cols - 1) // cols
        while len(grid_rows) < rows * cols:
            grid_rows.append(np.zeros_like(grid_rows[0]))

        row_images = []
        for r in range(rows):
            row_images.append(np.hstack(grid_rows[r * cols:(r + 1) * cols]))
        contact_sheet = np.vstack(row_images)

        out_name = f"contact_sheet_{status_key.lower()}.png"
        cv2.imwrite(str(PREVIEWS_DIR / out_name), contact_sheet)
        print(f"  - Saved {out_name} ({contact_sheet.shape[1]}x{contact_sheet.shape[0]}px)")

    # Save a high-resolution individual sample comparison for demonstration
    if samples_by_status["GOOD"]:
        sample = samples_by_status["GOOD"][0]
        ov = draw_landmarks_and_roi(
            sample["gray"], sample["landmarks"], sample["pv1"], sample["pv2"], sample["bbox"], "GOOD"
        )
        roi_rgb = cv2.cvtColor(sample["roi"], cv2.COLOR_GRAY2BGR)
        roi_resized = cv2.resize(roi_rgb, (ov.shape[0], ov.shape[0]))
        demo_card = np.hstack([ov, roi_resized])
        cv2.imwrite(str(PREVIEWS_DIR / "sample_good_demo.png"), demo_card)
        print(f"  - Saved sample_good_demo.png")

    return quality_report


if __name__ == "__main__":
    run_visual_audit()
