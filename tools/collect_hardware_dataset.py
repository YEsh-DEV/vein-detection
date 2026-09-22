#!/usr/bin/env python3
"""
tools/collect_hardware_dataset.py — Reproducible Hardware Data Collection Tool
=============================================================================
Dedicated Palm-Vein Data Collector for Raspberry Pi NoIR / USB NIR Cameras.

Enforces:
1. Immutable storage of raw uncompressed images.
2. Real-time pre-landmark diagnostic feedback (Phase 2).
3. Enrollment & capture quality gate validation (Phase 3).
4. Structured metadata logging into dataset/collected_dataset_index.csv.
5. Multi-session, multi-hand, multi-sample guided collection.
"""

import os
import sys
import time
import argparse
import datetime
import csv
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.mediapipe_img import (
    build_landmarker,
    detect_hand_landmarks_with_diagnostics,
    extract_valleys_from_landmarks,
    extract_ma2017_scaled_roi,
    compute_roi_quality,
    diagnose_hand_positioning,
)
from app.constants import DATASET_DIR

DEFAULT_RAW_DIR = os.path.join(DATASET_DIR, "raw_captures")
DEFAULT_ROI_DIR = os.path.join(DATASET_DIR, "rois")
DEFAULT_MANIFEST = os.path.join(DATASET_DIR, "collected_dataset_index.csv")

CSV_HEADER = [
    "subject_id",
    "hand",
    "session_id",
    "capture_idx",
    "timestamp",
    "raw_path",
    "roi_path",
    "passed_landmarks",
    "landmark_failure_reason",
    "positioning_instruction",
    "pad_pct",
    "contrast_std",
    "mean_brightness",
    "occupancy_pct",
    "roi_valid",
    "camera_config",
]

# Guided positioning variation prompts
POSITION_VARIATIONS = [
    "Position 1: Flat, centered 10-15cm above camera",
    "Position 2: Slight tilt left (~5 deg)",
    "Position 3: Slight tilt right (~5 deg)",
    "Position 4: Slightly higher (~15-18cm)",
    "Position 5: Fingers spread naturally",
    "Position 6: Flat confirmation capture",
]


class CameraHandler:
    """Manages Picamera2 (Pi NoIR) or OpenCV VideoCapture."""
    def __init__(self, camera_index=0, exposure_us=5000, gain=1.0, mock=False):
        self.mock = mock
        self.cam_type = "mock" if mock else "unknown"
        self.cam = None
        self.exposure_us = exposure_us
        self.gain = gain
        self.frame_width = 1280
        self.frame_height = 720

        if mock:
            print("[CameraHandler] Running in MOCK frame mode.")
            return

        # 1. Try Picamera2 (Raspberry Pi native)
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            preview_cfg = self.cam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.cam.configure(preview_cfg)
            self.cam.start()
            self.cam.set_controls({
                "AeEnable": False,
                "ExposureTime": exposure_us,
                "AnalogueGain": gain,
            })
            self.cam_type = "picamera2"
            print(f"[CameraHandler] Picamera2 initialized ({exposure_us}µs, gain {gain}).")
            return
        except Exception as e:
            print(f"[CameraHandler] Picamera2 unavailable ({e}). Trying OpenCV VideoCapture...")

        # 2. Try OpenCV VideoCapture
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cam = cap
                self.cam_type = "opencv"
                print(f"[CameraHandler] OpenCV VideoCapture({camera_index}) initialized.")
                return
            else:
                print(f"[CameraHandler] OpenCV VideoCapture({camera_index}) failed to open.")
        except Exception as e:
            print(f"[CameraHandler] OpenCV VideoCapture failed: {e}")

        print("[CameraHandler] No hardware camera available. Defaulting to synthetic frame fallback.")
        self.mock = True
        self.cam_type = "synthetic_fallback"

    def read_frame(self):
        """Returns BGR frame and Grayscale frame."""
        if self.mock or self.cam is None:
            # Generate a clean synthetic test frame with an ellipse palm
            frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
            cv2.ellipse(frame, (640, 360), (160, 210), 0, 0, 360, (140, 140, 140), -1)
            # Add synthetic fingers
            for offset_x in [-90, -30, 30, 90]:
                cv2.ellipse(frame, (640 + offset_x, 150), (25, 80), 0, 0, 360, (130, 130, 130), -1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True, frame, gray

        if self.cam_type == "picamera2":
            frame_rgb = self.cam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True, frame, gray

        if self.cam_type == "opencv":
            ret, frame = self.cam.read()
            if not ret or frame is None:
                return False, None, None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True, frame, gray

        return False, None, None

    def release(self):
        if self.cam_type == "opencv" and self.cam is not None:
            self.cam.release()
        elif self.cam_type == "picamera2" and self.cam is not None:
            self.cam.stop()


def ensure_manifest(manifest_path: str):
    """Ensures CSV manifest exists with header."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def log_sample_to_manifest(manifest_path: str, record: dict):
    """Appends sample record to manifest CSV."""
    ensure_manifest(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(record)


def process_and_save_capture(
    raw_frame: np.ndarray,
    gray_frame: np.ndarray,
    subject_id: str,
    hand: str,
    session_id: str,
    capture_idx: int,
    landmarker,
    raw_dir: str = DEFAULT_RAW_DIR,
    roi_dir: str = DEFAULT_ROI_DIR,
    manifest_path: str = DEFAULT_MANIFEST,
    camera_config: str = "default",
) -> dict:
    """
    Evaluates capture quality, saves immutable raw image, extracts ROI if valid,
    and logs metadata to manifest.
    """
    os.makedirs(os.path.join(raw_dir, subject_id, session_id), exist_ok=True)
    os.makedirs(os.path.join(roi_dir, subject_id, session_id), exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    base_name = f"{subject_id}_{hand}_{session_id}_{capture_idx:03d}_{timestamp_str}"
    raw_filename = f"{base_name}_raw.png"
    roi_filename = f"{base_name}_roi.png"

    raw_path = os.path.join(raw_dir, subject_id, session_id, raw_filename)
    roi_path = os.path.join(roi_dir, subject_id, session_id, roi_filename)

    # 1. Save raw image immutably
    cv2.imwrite(raw_path, raw_frame)

    # 2. Pre-landmark diagnostics & landmark detection
    diag = diagnose_hand_positioning(gray_frame)
    lm_result = detect_hand_landmarks_with_diagnostics(gray_frame, landmarker)

    passed_landmarks = lm_result["success"]
    failure_reason = lm_result["reason"]
    instruction = lm_result["instruction"]

    pad_pct = 0.0
    contrast_std = 0.0
    mean_brightness = float(np.mean(gray_frame))
    occupancy_pct = float(diag.get("occupancy_pct", 0.0))
    roi_valid = False

    # 3. If landmarks detected, extract ROI and validate quality gate
    if passed_landmarks:
        try:
            landmarks_21 = lm_result["landmarks"]
            valleys = extract_valleys_from_landmarks(landmarks_21)
            if valleys is not None:
                pv1, pv2 = valleys
                roi_res = extract_ma2017_scaled_roi(gray_frame, pv1, pv2)
                if roi_res is not None:
                    roi_224, bbox, _ = roi_res
                    quality = compute_roi_quality(roi_224, bbox, gray_frame.shape)
                    pad_pct = quality["pad_pct"]
                    contrast_std = quality["contrast_std"]
                    roi_valid = quality["valid"]

                    if roi_valid:
                        cv2.imwrite(roi_path, roi_224)
                    else:
                        roi_path = ""
                        instruction = f"Quality gate rejected: {quality['reasons']}"
                else:
                    roi_path = ""
                    instruction = "Failed MA2017 ROI extraction"
            else:
                roi_path = ""
                instruction = "Failed knuckle valley anchor extraction (Pv1/Pv2)"
        except Exception as e:
            roi_path = ""
            instruction = f"ROI extraction error: {e}"
    else:
        roi_path = ""

    # 4. Record to manifest
    record = {
        "subject_id": subject_id,
        "hand": hand.lower(),
        "session_id": session_id,
        "capture_idx": capture_idx,
        "timestamp": datetime.datetime.now().isoformat(),
        "raw_path": raw_path,
        "roi_path": roi_path,
        "passed_landmarks": passed_landmarks,
        "landmark_failure_reason": failure_reason,
        "positioning_instruction": instruction,
        "pad_pct": round(pad_pct, 4),
        "contrast_std": round(contrast_std, 2),
        "mean_brightness": round(mean_brightness, 2),
        "occupancy_pct": round(occupancy_pct, 2),
        "roi_valid": roi_valid,
        "camera_config": camera_config,
    }
    log_sample_to_manifest(manifest_path, record)
    return record


def run_batch_collection(
    subject_id: str,
    hand: str,
    session_id: str,
    num_samples: int = 6,
    camera_index: int = 0,
    exposure_us: int = 5000,
    gain: float = 1.0,
    mock: bool = False,
    countdown_sec: int = 3,
):
    """
    Automated or guided batch collection session.
    """
    print(f"\n==================================================")
    print(f"  STARTING DATA COLLECTION SESSION")
    print(f"  Subject: {subject_id} | Hand: {hand.upper()} | Session: {session_id}")
    print(f"  Target: {num_samples} samples")
    print(f"==================================================\n")

    camera = CameraHandler(camera_index=camera_index, exposure_us=exposure_us, gain=gain, mock=mock)
    landmarker = build_landmarker()

    successful_rois = 0
    total_captures = 0

    try:
        for idx in range(1, num_samples + 1):
            guide_idx = (idx - 1) % len(POSITION_VARIATIONS)
            prompt = POSITION_VARIATIONS[guide_idx]
            print(f"\n[{idx}/{num_samples}] Guide: {prompt}")

            # Countdown
            if not mock:
                for c in range(countdown_sec, 0, -1):
                    print(f"  Capturing in {c}... (press Ctrl+C to cancel)", end="\r", flush=True)
                    time.sleep(1.0)
                print("  CAPTURE!                                          ")

            ret, frame, gray = camera.read_frame()
            if not ret or gray is None:
                print(f"  [ERROR] Camera read failed for sample {idx}.")
                continue

            total_captures += 1
            record = process_and_save_capture(
                raw_frame=frame,
                gray_frame=gray,
                subject_id=subject_id,
                hand=hand,
                session_id=session_id,
                capture_idx=idx,
                landmarker=landmarker,
                camera_config=f"cam_type={camera.cam_type},exp={exposure_us}us,gain={gain}",
            )

            status_symbol = "✓" if record["roi_valid"] else "✗"
            print(
                f"  [{status_symbol}] Raw saved: {os.path.basename(record['raw_path'])} | "
                f"Landmarks: {record['passed_landmarks']} | "
                f"ROI Valid: {record['roi_valid']} | "
                f"Feedback: {record['positioning_instruction']}"
            )
            if record["roi_valid"]:
                successful_rois += 1

            if not mock and idx < num_samples:
                time.sleep(0.5)

    finally:
        camera.release()

    print(f"\n==================================================")
    print(f"  SESSION COMPLETED")
    print(f"  Total Captures: {total_captures}")
    print(f"  Valid ROIs: {successful_rois} ({(successful_rois/total_captures*100.0 if total_captures else 0):.1f}%)")
    print(f"  Manifest: {DEFAULT_MANIFEST}")
    print(f"==================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Palm-Vein Hardware Dataset Collection Tool")
    parser.add_argument("--subject-id", type=str, default="test_subject_01", help="Subject identifier (e.g. 035, subj_01)")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default="right", help="Palm hand (left or right)")
    parser.add_argument("--session-id", type=str, default="S1", help="Session ID (e.g. S1, S2)")
    parser.add_argument("--samples", type=int, default=6, help="Number of samples to collect")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--exposure-us", type=int, default=5000, help="Exposure time in microseconds")
    parser.add_argument("--gain", type=float, default=1.0, help="Analogue gain")
    parser.add_argument("--mock", action="store_true", help="Run with mock synthetic frames for testing/CI")
    args = parser.parse_args()

    run_batch_collection(
        subject_id=args.subject_id,
        hand=args.hand,
        session_id=args.session_id,
        num_samples=args.samples,
        camera_index=args.camera_index,
        exposure_us=args.exposure_us,
        gain=args.gain,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
