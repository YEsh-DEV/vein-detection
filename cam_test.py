#!/usr/bin/env python3
"""
cam_test.py — Live Camera Terminal Test for Palm Vein Authentication
Run on Raspberry Pi 5:
    python3 cam_test.py

Keys:
    N = Enroll new user (6 samples, 5s countdown each)
    S = Scan & identify (3s countdown)
    L = List enrolled users
    Q = Quit
"""

import os
import sys
import time
import re
import cv2
import numpy as np

from db_manager import init_db, enroll_user, user_exists, list_users, log_access
from search_engine import SearchEngine
from mediapipe_img import (
    build_landmarker,
    detect_hand_landmarks,
    extract_valleys_from_landmarks,
    segment_hand,
    extract_ma2017_scaled_roi,
    enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD

CAPTURE_DIR = "captures"
ROI_DIR = "roi_clahe"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)


def save_capture_to_disk(gray: np.ndarray, roi: np.ndarray, username: str, mode: str, idx: int = 0):
    """Save raw capture and CLAHE ROI to captures/ and roi_clahe/."""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cap_name = f"{username}_{mode}_{idx}_{ts}.png" if mode == "enroll" else f"{username}_{mode}_{ts}.png"
        roi_name = f"{username}_{mode}_{idx}_{ts}_clahe.png" if mode == "enroll" else f"{username}_{mode}_{ts}_clahe.png"
        cv2.imwrite(os.path.join(CAPTURE_DIR, cap_name), gray)
        cv2.imwrite(os.path.join(ROI_DIR, roi_name), roi)
    except Exception as e:
        print(f"[!] Warning: Failed saving capture to disk: {e}")


POSITION_HINTS = [
    "FLAT — hold palm flat, centered, 10-15cm above camera",
    "TILT LEFT — tilt palm ~5 degrees to the left",
    "TILT RIGHT — tilt palm ~5 degrees to the right",
    "HIGHER — raise palm slightly to 15-18cm",
    "WIDER — spread fingers slightly wider than normal",
    "FLAT AGAIN — return to flat position for final confirmation",
]


def init_camera():
    """
    Try Picamera2 first (Pi NoIR camera).
    Fall back to OpenCV VideoCapture(0) for webcam.
    Returns (camera_object, camera_type_string, preview_cfg, still_cfg).
    camera_type is 'picamera2' or 'opencv'.
    Raises RuntimeError if no camera found.
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
            "ExposureTime": 5000,
            "AnalogueGain": 1.0,
        })
        print("[+] Picamera2 (Pi NoIR) initialized.")
        return cam, "picamera2", preview_cfg, still_cfg
    except Exception as e:
        print(f"[!] Picamera2 not available: {e}")

    # 2. Fall back to OpenCV
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print("[+] OpenCV webcam initialized.")
                return cap, "opencv", None, None
            cap.release()
    except Exception as e:
        print(f"[!] OpenCV camera error: {e}")

    raise RuntimeError("No camera found. Connect Pi NoIR camera or USB webcam.")


def get_preview_frame(cam, cam_type):
    """Get a single BGR frame for display."""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        return frame if ret else None


def capture_still_gray(cam, cam_type, still_cfg, preview_cfg):
    """
    Capture a high-resolution still frame as grayscale.
    For Picamera2: switch to still mode, capture, switch back.
    For OpenCV: flush buffer and read.
    """
    if cam_type == "picamera2":
        cam.switch_mode(still_cfg)
        frame = cam.capture_array()
        cam.switch_mode(preview_cfg)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        # Flush stale buffer
        for _ in range(3):
            cam.grab()
        ret, frame = cam.read()
        if not ret or frame is None:
            raise ValueError("Failed to capture from webcam.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame


def run_pipeline(gray, landmarker):
    """
    Full pipeline: grayscale frame -> (clahe_roi, veincode)
    Raises ValueError with descriptive message on any failure.
    """
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    if landmarks is None or len(landmarks) < 21:
        raise ValueError("No hand landmarks detected. Hold palm flat ~10-15cm above camera.")

    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise ValueError("Cannot detect finger valley landmarks. Spread fingers slightly.")

    hand_mask = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35
    )
    if roi_256 is None or roi_256.size == 0:
        raise ValueError("Failed to extract palm ROI bounding box.")

    clahe_roi = enhance_roi_vessels(roi_256)
    code = extract_veincode(clahe_roi)
    return clahe_roi, code


def draw_hud(frame, text_lines, box_color=(0, 255, 200)):
    """Draw alignment box and text overlay on preview frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    box = 130

    # Alignment rectangle
    cv2.rectangle(frame, (cx - box, cy - box), (cx + box, cy + box), box_color, 2)
    cv2.putText(frame, "ALIGN PALM HERE", (cx - 90, cy - box - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Status lines at bottom
    y = h - (len(text_lines) * 28) - 10
    for line in text_lines:
        cv2.rectangle(frame, (0, y - 20), (w, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 28

    return frame


def countdown_with_preview(cam, cam_type, seconds, label, preview_cfg=None, still_cfg=None):
    """
    Show live camera feed with countdown overlay.
    Returns True when countdown finishes.
    Returns False if user presses Q to cancel.
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

        # Big countdown number in center
        h, w = frame.shape[:2]
        cv2.putText(frame, str(remaining),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                    (0, 255, 255), 7)
        cv2.putText(frame, label,
                    (max(10, w // 2 - 160), h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        draw_hud(frame, [f"HOLD STEADY — {remaining}s remaining", "Press Q or ESC to cancel"])
        cv2.imshow("PALM VEIN AUTH — Pi 5 Test", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            return False

    return True


def do_enroll(cam, cam_type, preview_cfg, still_cfg, landmarker, engine):
    print("\n" + "=" * 50)
    print("  ENROLL NEW USER")
    print("=" * 50)

    username = input("Enter username (letters, numbers, hyphens only): ").strip().lower()
    if not username or len(username) < 2:
        print("[!] Username too short. Minimum 2 characters.")
        return

    if not re.match(r'^[a-z0-9][a-z0-9_-]{1,29}$', username):
        print("[!] Invalid username. Use letters, numbers, hyphens, underscores only.")
        return

    if user_exists(username):
        print(f"[!] User '{username}' already enrolled. Delete first to re-enroll.")
        return

    print(f"\nEnrolling '{username}' — 6 samples required.")
    print("Camera window will open. Align your palm in the green box.")
    print("A 5-second countdown will appear before each capture.\n")

    veincode_list = []
    i = 0
    while i < 6:
        hint = POSITION_HINTS[i]
        print(f"\n--- Sample {i+1} of 6 ---")
        print(f"Position: {hint}")
        input("Press ENTER when ready (then watch the countdown)...")

        # Show 5-second countdown with live preview
        cancelled = not countdown_with_preview(
            cam, cam_type, 5,
            f"SAMPLE {i+1}/6 — {hint[:30]}",
            preview_cfg, still_cfg
        )

        if cancelled:
            print("[!] Cancelled by user.")
            cv2.destroyAllWindows()
            return

        # Capture still
        print("  Capturing...")
        try:
            gray = capture_still_gray(cam, cam_type, still_cfg, preview_cfg)
        except Exception as e:
            print(f"  [!] Capture failed: {e}")
            retry = input("  Retry this sample? [y/N]: ").strip().lower()
            if retry == 'y':
                continue
            else:
                cv2.destroyAllWindows()
                return

        # Run pipeline
        print("  Processing (MediaPipe + Gabor)...")
        t0 = time.time()
        try:
            clahe_roi, code = run_pipeline(gray, landmarker)
        except ValueError as e:
            print(f"  [!] Pipeline failed: {e}")
            print("  Adjust hand position and try this sample again.")
            retry = input("  Retry? [y/N]: ").strip().lower()
            if retry == 'y':
                continue
            else:
                cv2.destroyAllWindows()
                return

        elapsed = time.time() - t0
        veincode_list.append(code)
        save_capture_to_disk(gray, clahe_roi, username, "enroll", idx=i+1)
        print(f"  OK — VR mean: {code['VR'].mean():.3f}  ({elapsed:.2f}s) [Saved to captures/ and roi_clahe/]")

        # Show the CLAHE ROI so user can see what was captured
        display_roi = cv2.resize(clahe_roi, (320, 320))
        cv2.imshow(f"Sample {i+1} — CLAHE ROI (auto-close)", display_roi)
        cv2.waitKey(1500)  # Show for 1.5 seconds then auto-close
        cv2.destroyAllWindows()
        i += 1

    cv2.destroyAllWindows()

    # Consistency check
    print("\nRunning consistency check across 6 samples...")
    bad_pairs = []
    for a in range(len(veincode_list)):
        for b in range(a + 1, len(veincode_list)):
            score = match_templates(veincode_list[a], veincode_list[b])
            if score > 0.50:
                bad_pairs.append((a + 1, b + 1, score))

    if bad_pairs:
        print(f"[!] WARNING: {len(bad_pairs)} inconsistent sample pair(s):")
        for a, b, s in bad_pairs:
            print(f"    Sample {a} vs Sample {b}: {s:.4f}")
        print("    Consider re-enrolling for better accuracy.")
    else:
        print("[+] All samples consistent — quality: GOOD")

    # Save to DB
    print(f"\nSaving {len(veincode_list)} templates to database...")
    enroll_user(username, veincode_list)
    engine.refresh_cache()
    print(f"[+] ENROLLED: '{username}' with {len(veincode_list)} samples.")
    print("=" * 50)


def do_scan(cam, cam_type, preview_cfg, still_cfg, landmarker, engine):
    print("\n" + "=" * 50)
    print("  SCAN & IDENTIFY")
    print("=" * 50)
    print("Camera window will open. Place palm in the green box.")
    print("3-second countdown, then capture.\n")
    input("Press ENTER to start...")

    # 3-second countdown
    cancelled = not countdown_with_preview(
        cam, cam_type, 3, "HOLD PALM STEADY", preview_cfg, still_cfg
    )
    if cancelled:
        print("[!] Cancelled.")
        cv2.destroyAllWindows()
        return

    # Capture
    print("Capturing...")
    try:
        gray = capture_still_gray(cam, cam_type, still_cfg, preview_cfg)
    except Exception as e:
        print(f"[!] Capture failed: {e}")
        cv2.destroyAllWindows()
        return

    # Pipeline
    print("Processing (MediaPipe + Gabor)...")
    t0 = time.time()
    try:
        clahe_roi, probe_code = run_pipeline(gray, landmarker)
    except ValueError as e:
        print(f"[!] Pipeline failed: {e}")
        print("Adjust lighting and hand position, then try again.")
        cv2.destroyAllWindows()
        return

    pipeline_time = time.time() - t0
    save_capture_to_disk(gray, clahe_roi, "unknown", "scan")

    # Show CLAHE ROI
    display_roi = cv2.resize(clahe_roi, (320, 320))
    cv2.imshow("Scanned ROI — CLAHE", display_roi)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Identify
    print("Matching against enrolled users...")
    t1 = time.time()
    username, score = engine.identify(probe_code)
    match_time = time.time() - t1

    total_time = time.time() - t0

    print("\n" + "=" * 50)
    if username:
        print(f"  RESULT:     AUTHENTICATED")
        print(f"  User:       {username}")
    else:
        print(f"  RESULT:     NOT RECOGNISED")
        print(f"  User:       ---")

    print(f"  Score:      {score:.4f}  (threshold: {MATCH_THRESHOLD:.4f})")
    print(f"  Pipeline:   {pipeline_time:.2f}s")
    print(f"  Matching:   {match_time:.2f}s")
    print(f"  Total:      {total_time:.2f}s")
    print("=" * 50)

    if username:
        save_capture_to_disk(gray, clahe_roi, username, "scan")
    log_access(user_id=None, score=score, accepted=(username is not None))


def main():
    print("\n" + "=" * 50)
    print("  PALM VEIN AUTH — Pi 5 Terminal Test")
    print("=" * 50)

    # Init DB and engine
    print("Initialising database...")
    init_db()
    engine = SearchEngine(n_workers=4)

    # Load MediaPipe
    print("Loading MediaPipe hand landmarker...")
    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
        print("[+] Landmarker ready.")
    except Exception as e:
        print(f"[!] Landmarker warning/error: {e}")
        engine.shutdown()
        sys.exit(1)

    # Init camera
    print("Initialising camera...")
    try:
        cam, cam_type, preview_cfg, still_cfg = init_camera()
    except RuntimeError as e:
        print(f"[!] {e}")
        engine.shutdown()
        sys.exit(1)

    # Show enrolled users
    users = list_users()
    print(f"\n[+] System ready. {len(users)} user(s) enrolled.")
    if users:
        for u in users:
            print(f"    - {u['username']} ({u['sample_count']} samples)")

    print("\nControls:")
    print("  N = Enroll new user")
    print("  S = Scan & identify")
    print("  L = List enrolled users")
    print("  Q = Quit\n")

    # Live preview loop
    print("Opening camera preview... (press a key in the OpenCV window or Ctrl+C in terminal)")
    try:
        while True:
            # Show live preview
            frame = get_preview_frame(cam, cam_type)
            if frame is not None:
                users_count = len(list_users())
                draw_hud(frame, [
                    f"Users enrolled: {users_count}  |  Threshold: {MATCH_THRESHOLD}",
                    "N=Enroll  S=Scan  L=List  Q=Quit"
                ])
                cv2.imshow("PALM VEIN AUTH — Pi 5 Test", frame)

            # Check for key press in OpenCV window
            key = cv2.waitKey(30) & 0xFF
            if key == ord('n') or key == ord('N'):
                cv2.destroyAllWindows()
                do_enroll(cam, cam_type, preview_cfg, still_cfg, landmarker, engine)
            elif key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                do_scan(cam, cam_type, preview_cfg, still_cfg, landmarker, engine)
            elif key == ord('l') or key == ord('L'):
                users = list_users()
                print(f"\nEnrolled users ({len(users)}):")
                for u in users:
                    print(f"  - {u['username']}  ({u['sample_count']} samples, enrolled {u['enrolled_at']})")
                print()
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
        engine.shutdown()
        print("[+] Clean exit.")


if __name__ == "__main__":
    main()
