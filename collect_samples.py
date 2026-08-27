#!/usr/bin/env python3
"""
collect_samples.py — Dedicated Palm Sample Dataset Collector
=============================================================
Designed for Raspberry Pi 5 (NoIR Camera / Picamera2) & USB Webcams.
Captures raw, full-resolution palm images for preprocessing model development,
annotation, and fine-tuning.

Directory Structure:
    dataset/
      ├── subject_01/
      │     ├── subject_01_right_001_20260827_120000.png
      │     ├── subject_01_right_002_20260827_120005.png
      │     └── subject_01_left_001_20260827_120015.png
      └── dataset_log.csv

Key Controls (press in OpenCV window):
    SPACE / C : Instant Single Capture
    B         : Batch Guided Capture (6 samples with 3s/5s countdowns)
    H         : Toggle Hand (Right ⟷ Left)
    U         : Change Subject / Person Name
    [ / ]     : Adjust Exposure Time (darker / brighter for NIR illumination)
    L         : Show Dataset Summary in Terminal
    Q / ESC   : Quit
"""

import os
import sys
import time
import re
import csv
import json
import cv2
import numpy as np

DATASET_DIR = "dataset"
LOG_CSV = os.path.join(DATASET_DIR, "dataset_log.csv")
os.makedirs(DATASET_DIR, exist_ok=True)

# Default Capture Settings
DEFAULT_EXPOSURE_US = 5000   # 5ms default for Pi NoIR
DEFAULT_GAIN = 1.0

# Batch Position Guidance Hints
GUIDED_POSITIONS = [
    "FLAT (10-15cm centered above camera)",
    "SLIGHT TILT LEFT (~5 degrees)",
    "SLIGHT TILT RIGHT (~5 degrees)",
    "SLIGHTLY HIGHER (~15-18cm)",
    "FINGERS SPREAD WIDE",
    "FLAT (final confirmation)",
    "ROTATE CLOCKWISE (~5 degrees)",
    "ROTATE COUNTER-CLOCKWISE (~5 degrees)",
    "CLOSER TO CAMERA (~8-10cm)",
    "FARTHER FROM CAMERA (~18-20cm)",
]


def init_camera(exposure_us=DEFAULT_EXPOSURE_US, gain=DEFAULT_GAIN):
    """
    Initialize Picamera2 (Pi NoIR Camera) or OpenCV VideoCapture fallback.
    Returns (cam, cam_type, preview_cfg, still_cfg).
    """
    # 1. Try Picamera2
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        preview_cfg = cam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        still_cfg = cam.create_still_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )
        cam.configure(preview_cfg)
        cam.start()
        cam.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_us,
            "AnalogueGain": gain,
        })
        print(f"[+] Picamera2 (Pi NoIR) initialized. Exposure: {exposure_us}µs, Gain: {gain}")
        return cam, "picamera2", preview_cfg, still_cfg
    except Exception as e:
        print(f"[!] Picamera2 not available: {e}")

    # 2. Fall back to OpenCV
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Request higher resolution if supported
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, _ = cap.read()
            if ret:
                print("[+] OpenCV webcam initialized.")
                return cap, "opencv", None, None
            cap.release()
    except Exception as e:
        print(f"[!] OpenCV camera error: {e}")

    raise RuntimeError("No camera found! Connect Pi NoIR camera or USB webcam.")


def set_camera_exposure(cam, cam_type, exposure_us):
    """Dynamically adjust exposure on Pi NoIR Camera."""
    if cam_type == "picamera2":
        try:
            cam.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us)})
            print(f"[+] Camera exposure set to {exposure_us}µs")
        except Exception as e:
            print(f"[!] Failed to set exposure: {e}")


def get_preview_frame(cam, cam_type):
    """Capture a single preview BGR frame for live display."""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        return frame if ret else None


def capture_high_res_still(cam, cam_type, still_cfg, preview_cfg):
    """
    Capture a raw, high-resolution still frame.
    Returns (raw_gray, raw_bgr).
    """
    if cam_type == "picamera2":
        cam.switch_mode(still_cfg)
        frame_rgb = cam.capture_array()
        cam.switch_mode(preview_cfg)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return gray, bgr
    else:
        # Flush stale frames from buffer
        for _ in range(3):
            cam.grab()
        ret, frame_bgr = cam.read()
        if not ret or frame_bgr is None:
            raise ValueError("Failed to capture frame from camera.")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        return gray, frame_bgr


def log_sample_to_csv(subject, hand, filename, width, height, mean_brightness, exposure_us):
    """Append sample metadata to dataset_log.csv."""
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "subject", "hand", "filename", "width", "height", "mean_brightness", "exposure_us"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            subject,
            hand,
            filename,
            width,
            height,
            f"{mean_brightness:.2f}",
            exposure_us
        ])


def save_raw_sample(gray, bgr, subject, hand, exposure_us):
    """
    Save the raw capture into dataset/<subject>/ folder.
    Returns the saved file path and sample index.
    """
    subject_dir = os.path.join(DATASET_DIR, subject)
    os.makedirs(subject_dir, exist_ok=True)

    # Count existing samples for this subject & hand
    existing = [f for f in os.listdir(subject_dir) if f.startswith(f"{subject}_{hand}_") and f.endswith(".png")]
    idx = len(existing) + 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{subject}_{hand}_{idx:03d}_{ts}.png"
    filepath = os.path.join(subject_dir, filename)

    # Save high-resolution raw grayscale image
    cv2.imwrite(filepath, gray)

    # Calculate image statistics for quality monitoring
    mean_val = float(np.mean(gray))
    h, w = gray.shape[:2]

    # Log to CSV
    log_sample_to_csv(subject, hand, filename, w, h, mean_val, exposure_us)

    return filepath, idx, mean_val, (w, h)


def draw_hud(frame, subject, hand, total_count, exposure_us, extra_lines=None):
    """Draw professional HUD with alignment guide, status, and hotkeys."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    box = 130

    # Alignment Box
    cv2.rectangle(frame, (cx - box, cy - box), (cx + box, cy + box), (0, 255, 200), 2)
    cv2.putText(frame, "PALM ALIGNMENT AREA", (cx - 95, cy - box - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 2)

    # Top Status Bar
    cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
    status_text = f"Subject: [{subject.upper()}]  |  Hand: [{hand.upper()}]  |  Saved: {total_count}  |  Exp: {exposure_us}us"
    cv2.putText(frame, status_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

    # Bottom Instructions Bar
    lines = [
        "SPACE/C: Snap  |  B: Batch 6-Snap  |  H: Toggle Hand  |  U: Change User",
        "[/]: Adjust Exposure  |  L: List Stats  |  Q/ESC: Quit"
    ]
    if extra_lines:
        lines = extra_lines + lines

    y = h - (len(lines) * 26) - 8
    cv2.rectangle(frame, (0, y - 10), (w, h), (15, 15, 15), -1)
    for line in lines:
        cv2.putText(frame, line, (12, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        y += 26

    return frame


def countdown_preview(cam, cam_type, seconds, label, subject, hand, total_count, exposure_us):
    """
    Live camera countdown overlay. Returns True on finish, False if cancelled.
    """
    start = time.time()
    while True:
        frame = get_preview_frame(cam, cam_type)
        if frame is None:
            time.sleep(0.01)
            continue

        elapsed = time.time() - start
        remaining = seconds - int(elapsed)

        if remaining <= 0:
            break

        h, w = frame.shape[:2]

        # Draw HUD first
        draw_hud(frame, subject, hand, total_count, exposure_us, [f"--- {label} ---"])

        # Big Countdown Counter
        cv2.putText(frame, str(remaining), (w // 2 - 35, h // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 255, 255), 7)
        cv2.putText(frame, label, (max(10, w // 2 - 170), h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("PALM SAMPLE COLLECTOR — Pi 5", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            return False

    return True


def run_batch_capture(cam, cam_type, preview_cfg, still_cfg, subject, hand, exposure_us, count=6):
    """
    Run an automated guided batch capture session with countdowns.
    """
    print("\n" + "=" * 55)
    print(f"  STARTING BATCH CAPTURE: {count} samples for [{subject.upper()} - {hand.upper()}]")
    print("=" * 55)

    for i in range(count):
        hint = GUIDED_POSITIONS[i % len(GUIDED_POSITIONS)]
        label = f"SAMPLE {i+1}/{count} — {hint}"
        print(f"\n[{i+1}/{count}] Position hand: {hint}")
        print("  -> Watch the 4-second live countdown in the camera window...")

        ok = countdown_preview(cam, cam_type, 4, label, subject, hand, count_subject_samples(subject), exposure_us)
        if not ok:
            print("[!] Batch capture cancelled by user.")
            return

        # Snap high-res still
        try:
            gray, bgr = capture_high_res_still(cam, cam_type, still_cfg, preview_cfg)
            path, idx, mean_v, dims = save_raw_sample(gray, bgr, subject, hand, exposure_us)
            print(f"  [+] Saved: {path} ({dims[0]}x{dims[1]}, Mean brightness: {mean_v:.1f})")

            # Quick flash thumbnail (0.6s)
            thumb = cv2.resize(gray, (320, 240))
            cv2.putText(thumb, f"SAVED #{idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            cv2.imshow("Captured Frame", thumb)
            cv2.waitKey(600)
            cv2.destroyWindow("Captured Frame")

        except Exception as e:
            print(f"  [!] Capture error: {e}")
            time.sleep(1)

    print("\n" + "=" * 55)
    print(f"  [+] BATCH FINISHED! Total for {subject}: {count_subject_samples(subject)} samples.")
    print("=" * 55 + "\n")


def count_subject_samples(subject):
    """Count total images saved for this subject."""
    subject_dir = os.path.join(DATASET_DIR, subject)
    if not os.path.exists(subject_dir):
        return 0
    return len([f for f in os.listdir(subject_dir) if f.endswith(".png")])


def list_dataset_summary():
    """Print clean summary of collected samples in dataset/."""
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY (dataset/)")
    print("=" * 55)
    if not os.path.exists(DATASET_DIR):
        print("  No dataset directory found.")
        return

    subjects = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not subjects:
        print("  No subjects collected yet.")
        return

    total_all = 0
    for subj in sorted(subjects):
        sdir = os.path.join(DATASET_DIR, subj)
        right_cnt = len([f for f in os.listdir(sdir) if f.startswith(f"{subj}_right_") and f.endswith(".png")])
        left_cnt = len([f for f in os.listdir(sdir) if f.startswith(f"{subj}_left_") and f.endswith(".png")])
        total = right_cnt + left_cnt
        total_all += total
        print(f"  * {subj:18s} -> Right Hand: {right_cnt:3d} | Left Hand: {left_cnt:3d} | Total: {total:3d}")

    print("-" * 55)
    print(f"  TOTAL SAMPLES COLLECTED: {total_all}")
    print(f"  CSV LOG: {LOG_CSV}")
    print("=" * 55 + "\n")


def main():
    print("\n" + "=" * 55)
    print("  PALM SAMPLE DATASET COLLECTOR (Pi 5 / Webcam)")
    print("=" * 55)

    current_subject = "subject_01"
    current_hand = "right"
    exposure_us = DEFAULT_EXPOSURE_US

    # Prompt initial subject name
    user_input = input(f"Enter initial subject name [{current_subject}]: ").strip().lower()
    if user_input:
        cleaned = re.sub(r'[^a-z0-9_-]', '_', user_input)
        if cleaned:
            current_subject = cleaned

    hand_input = input(f"Enter initial hand (r/right or l/left) [{current_hand}]: ").strip().lower()
    if hand_input in ['l', 'left']:
        current_hand = "left"

    print(f"\n[+] Active Subject: [{current_subject}]")
    print(f"[+] Active Hand:    [{current_hand}]")
    print("[+] Initializing camera hardware...")

    try:
        cam, cam_type, preview_cfg, still_cfg = init_camera(exposure_us=exposure_us)
    except RuntimeError as e:
        print(f"[!] {e}")
        sys.exit(1)

    print("\n[+] Camera ready. Opening preview window...")
    print("    Press SPACE/C to snap, B for batch, H to toggle hand, Q to quit.\n")

    try:
        while True:
            frame = get_preview_frame(cam, cam_type)
            if frame is None:
                time.sleep(0.01)
                continue

            total_samples = count_subject_samples(current_subject)
            draw_hud(frame, current_subject, current_hand, total_samples, exposure_us)
            cv2.imshow("PALM SAMPLE COLLECTOR — Pi 5", frame)

            key = cv2.waitKey(30) & 0xFF

            # 1. Instant Capture (SPACE or 'C')
            if key == 32 or key == ord('c') or key == ord('C'):
                try:
                    gray, bgr = capture_high_res_still(cam, cam_type, still_cfg, preview_cfg)
                    path, idx, mean_v, dims = save_raw_sample(gray, bgr, current_subject, current_hand, exposure_us)
                    print(f"[+] Snapped: {path} ({dims[0]}x{dims[1]}, Mean brightness: {mean_v:.1f})")

                    # Visual feedback: flash green border
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 12)
                    cv2.imshow("PALM SAMPLE COLLECTOR — Pi 5", frame)
                    cv2.waitKey(120)

                except Exception as e:
                    print(f"[!] Capture failed: {e}")

            # 2. Batch Guided Capture ('B')
            elif key == ord('b') or key == ord('B'):
                run_batch_capture(cam, cam_type, preview_cfg, still_cfg, current_subject, current_hand, exposure_us, count=6)

            # 3. Toggle Hand ('H')
            elif key == ord('h') or key == ord('H'):
                current_hand = "left" if current_hand == "right" else "right"
                print(f"[+] Active hand switched to: [{current_hand.upper()}]")

            # 4. Change Subject Name ('U')
            elif key == ord('u') or key == ord('U'):
                cv2.destroyAllWindows()
                new_subj = input(f"\nEnter new subject name (currently '{current_subject}'): ").strip().lower()
                cleaned = re.sub(r'[^a-z0-9_-]', '_', new_subj)
                if cleaned:
                    current_subject = cleaned
                    print(f"[+] Active subject changed to: [{current_subject}]")
                else:
                    print("[!] Invalid name, keeping current.")

            # 5. Adjust Exposure ('[' to decrease, ']' to increase)
            elif key == ord('['):
                exposure_us = max(500, exposure_us - 1000)
                set_camera_exposure(cam, cam_type, exposure_us)
            elif key == ord(']'):
                exposure_us = min(30000, exposure_us + 1000)
                set_camera_exposure(cam, cam_type, exposure_us)

            # 6. List Dataset Statistics ('L')
            elif key == ord('l') or key == ord('L'):
                list_dataset_summary()

            # 7. Quit ('Q' or ESC)
            elif key == ord('q') or key == ord('Q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        if cam_type == "picamera2":
            try:
                cam.stop()
            except Exception:
                pass
        elif cam_type == "opencv":
            try:
                cam.release()
            except Exception:
                pass
        print("[+] Camera released. Sample collection finished.")
        list_dataset_summary()


if __name__ == "__main__":
    main()
