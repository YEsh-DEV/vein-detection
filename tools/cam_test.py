#!/usr/bin/env python3
"""
cam_test.py — Live Camera Terminal Test for Palm Vein Authentication
Run on Raspberry Pi 5:
    python3 tools/cam_test.py

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
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.constants import (
        CAPTURE_DIR, ROI_DIR, MODEL_PATH, MATCH_THRESHOLD,
        ENROLL_SAMPLE_MAX,
    )
    from app.db_manager import init_db, enroll_user, user_exists, list_users, log_access
    from app.search_engine import SearchEngine
    from app.mediapipe_img import (
        build_landmarker,
        detect_hand_landmarks,
        extract_valleys_from_landmarks,
        segment_hand,
        extract_ma2017_scaled_roi,
        enhance_roi_vessels,
    )
    from app.gabor import extract_veincode, match_templates
except ImportError:
    from constants import (
        CAPTURE_DIR, ROI_DIR, MODEL_PATH, MATCH_THRESHOLD,
        ENROLL_SAMPLE_MAX,
    )
    from db_manager import init_db, enroll_user, user_exists, list_users, log_access
    from search_engine import SearchEngine
    from mediapipe_img import (
        build_landmarker,
        detect_hand_landmarks,
        extract_valleys_from_landmarks,
        segment_hand,
        extract_ma2017_scaled_roi,
        enhance_roi_vessels,
    )
    from gabor import extract_veincode, match_templates


def save_capture_to_disk(gray: np.ndarray, roi: np.ndarray, username: str, mode: str, idx: int = 0):
    """Save raw capture and CLAHE ROI to captures/ and roi_clahe/."""
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        os.makedirs(ROI_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cap_name = f"{username}_{mode}_{idx}_{ts}.png" if mode == "enroll" else f"{username}_{mode}_{ts}.png"
        roi_name = f"{username}_{mode}_{idx}_{ts}_clahe.png" if mode == "enroll" else f"{username}_{mode}_{ts}_clahe.png"
        cap_path = os.path.join(CAPTURE_DIR, cap_name)
        roi_path = os.path.join(ROI_DIR, roi_name)
        ok1 = cv2.imwrite(cap_path, gray)
        ok2 = cv2.imwrite(roi_path, roi)
        if ok1 and ok2:
            print(f"[+] Saved capture: {cap_path} & {roi_path}")
        else:
            print(f"[!] Warning: cv2.imwrite failed (cap={ok1}, roi={ok2}) for {cap_path}")
    except Exception as e:
        print(f"[!] Warning: Failed saving capture to disk: {e}")


# Guide box specs on 640x480 preview
BOX_W = 280
BOX_H = 340
BOX_X1 = (640 - BOX_W) // 2
BOX_Y1 = (480 - BOX_H) // 2
BOX_X2 = BOX_X1 + BOX_W
BOX_Y2 = BOX_Y1 + BOX_H

POSITION_HINTS = [
    "Pose 1/6: Hand flat, palm facing camera directly",
    "Pose 2/6: Fingers spread slightly wider",
    "Pose 3/6: Hand tilted slightly LEFT (~5 deg)",
    "Pose 4/6: Hand tilted slightly RIGHT (~5 deg)",
    "Pose 5/6: Hand slightly closer to camera",
    "Pose 6/6: Hand slightly further from camera",
]


def init_camera():
    """Initialises Picamera2 or OpenCV fallback."""
    try:
        from picamera2 import Picamera2
        p = Picamera2()
        preview_cfg = p.create_preview_configuration(
            main={"size": (640, 480), "format": "XBGR8888"}
        )
        still_cfg = p.create_still_configuration(
            main={"size": (1640, 1232), "format": "XBGR8888"}
        )
        p.configure(preview_cfg)
        p.start()
        print("[+] Picamera2 camera initialised.")
        return p, "picamera2", preview_cfg, still_cfg
    except Exception as e:
        print(f"[-] Picamera2 unavailable ({e}). Trying OpenCV...")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[+] OpenCV VideoCapture(0) initialised.")
        return cap, "opencv", None, None

    print("[!] No camera found.")
    return None, None, None, None


def get_preview_frame(cam, cam_type):
    """Fetches a single 640x480 frame for live preview."""
    if cam_type == "picamera2":
        frame = cam.capture_array("main")
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    elif cam_type == "opencv":
        ret, frame = cam.read()
        return frame if ret else None
    return None


def capture_still_gray(cam, cam_type, still_cfg=None, preview_cfg=None):
    """Captures a full-res still frame and converts to grayscale."""
    if cam_type == "picamera2":
        cam.switch_mode_and_capture_array(still_cfg)
        still = cam.capture_array("main")
        cam.configure(preview_cfg)
        cam.start()
        if len(still.shape) == 3:
            return cv2.cvtColor(still, cv2.COLOR_BGR2GRAY)
        return still
    elif cam_type == "opencv":
        for _ in range(3):
            cam.grab()
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Failed to grab still from OpenCV camera")
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    raise RuntimeError("No camera available")


def draw_overlay(frame, text="", countdown=None, color=(0, 255, 0)):
    """Draws palm placement guide box and text on preview frame."""
    out = frame.copy()
    cv2.rectangle(out, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), color, 2)

    cx, cy = (BOX_X1 + BOX_X2) // 2, (BOX_Y1 + BOX_Y2) // 2
    cv2.circle(out, (cx, cy), 8, color, -1)

    cv2.putText(out, "Align palm in box", (BOX_X1 + 10, BOX_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if text:
        cv2.putText(out, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if countdown is not None:
        cv2.putText(out, str(countdown), (cx - 25, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

    return out


def countdown_with_preview(cam, cam_type, seconds, text, preview_cfg=None, still_cfg=None):
    """Runs a visual countdown overlaying the live camera feed."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = int(seconds - elapsed) + 1
        if elapsed >= seconds:
            break

        frame = get_preview_frame(cam, cam_type)
        if frame is None:
            time.sleep(0.05)
            continue

        overlay = draw_overlay(frame, text=text, countdown=remaining, color=(0, 255, 0))
        cv2.imshow("Palm Vein Pi 5", overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            return False

    return True


def process_image(gray: np.ndarray, landmarker):
    """Runs MediaPipe landmark detection -> Valley extraction -> ROI crop -> CLAHE -> Gabor."""
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    landmarks = detect_hand_landmarks(stretched, landmarker)
    if landmarks is None or len(landmarks) < 21:
        raise ValueError("No hand detected. Make sure palm is fully visible in frame.")

    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise ValueError("Could not find finger valleys. Spread fingers slightly.")

    hand_mask = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35,
        landmarks_px=landmarks
    )
    if roi_256 is None or roi_256.size == 0:
        raise ValueError("Failed to extract ROI bounding box.")

    clahe_roi = enhance_roi_vessels(roi_256)
    code = extract_veincode(clahe_roi)
    return clahe_roi, code


def enroll_flow(cam, cam_type, landmarker, engine, preview_cfg, still_cfg):
    """Enrolls a new user by capturing 6 samples with position guidance."""
    print("\n" + "=" * 50)
    print("  NEW USER ENROLLMENT")
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

    print(f"\nEnrolling '{username}' — {ENROLL_SAMPLE_MAX} samples required.")
    print("Camera window will open. Align your palm in the green box.")
    print("A 5-second countdown will appear before each capture.\n")

    veincode_list = []
    i = 0
    while i < ENROLL_SAMPLE_MAX:
        hint = POSITION_HINTS[i]
        print(f"\n--- Sample {i+1} of {ENROLL_SAMPLE_MAX} ---")
        print(f"Position: {hint}")
        input("Press ENTER when ready (then watch the countdown)...")

        cancelled = not countdown_with_preview(
            cam, cam_type, 5,
            f"SAMPLE {i+1}/{ENROLL_SAMPLE_MAX} — {hint[:30]}",
            preview_cfg, still_cfg
        )

        if cancelled:
            print("[!] Cancelled by user.")
            cv2.destroyAllWindows()
            return

        print("  Capturing...")
        try:
            gray = capture_still_gray(cam, cam_type, still_cfg, preview_cfg)
        except Exception as e:
            print(f"  [!] Capture failed: {e}")
            retry = input("  Retry this sample? [y/N]: ").strip().lower()
            if retry == 'y':
                continue
            else:
                return

        print("  Extracting features (MediaPipe + Gabor)...")
        t0 = time.time()
        try:
            clahe_roi, code = process_image(gray, landmarker)
        except ValueError as e:
            print(f"  [!] Extraction failed: {e}")
            retry = input("  Retry this sample? [y/N]: ").strip().lower()
            if retry == 'y':
                continue
            else:
                return
        except Exception as e:
            print(f"  [!] Unexpected error: {e}")
            return

        elapsed = time.time() - t0
        save_capture_to_disk(gray, clahe_roi, username, "enroll", idx=i+1)
        print(f"  OK — VR mean: {code['VR'].mean():.3f}  ({elapsed:.2f}s) [Saved to captures/ and roi_clahe/]")

        # Show extracted ROI preview briefly
        roi_preview = cv2.resize(clahe_roi, (240, 240))
        cv2.imshow("Extracted ROI", roi_preview)
        cv2.waitKey(1000)

        veincode_list.append(code)
        i += 1

    cv2.destroyAllWindows()

    print("\nValidating enrollment set consistency...")
    diffs = []
    for a in range(len(veincode_list)):
        for b in range(a + 1, len(veincode_list)):
            d = match_templates(veincode_list[a], veincode_list[b])
            diffs.append(d)

    mean_d = sum(diffs) / len(diffs) if diffs else 0.0
    print(f"Mean intra-enrollment distance: {mean_d:.4f}")

    if mean_d > 0.42:
        print("[!] Warning: Intra-set distance is high. Hand position may have varied too much.")
        proceed = input("Save anyway? [y/N]: ").strip().lower()
        if proceed != 'y':
            print("[*] Enrollment aborted. Nothing saved.")
            return

    print("Saving to database...")
    try:
        user_id = enroll_user(username, veincode_list)
        engine.refresh_cache()
        print(f"[+] SUCCESS: User '{username}' enrolled with {len(veincode_list)} templates (id={user_id})")
    except Exception as e:
        print(f"[!] Database error: {e}")


def scan_flow(cam, cam_type, landmarker, engine, preview_cfg, still_cfg):
    """Scans palm and identifies against enrolled database."""
    print("\n" + "=" * 50)
    print("  PALM SCAN & IDENTIFICATION")
    print("=" * 50)
    print("Align palm in the box. 3-second countdown will start...\n")

    cancelled = not countdown_with_preview(
        cam, cam_type, 3, "IDENTIFICATION SCAN — HOLD STILL",
        preview_cfg, still_cfg
    )

    if cancelled:
        print("[!] Cancelled by user.")
        cv2.destroyAllWindows()
        return

    print("Capturing...")
    try:
        gray = capture_still_gray(cam, cam_type, still_cfg, preview_cfg)
    except Exception as e:
        print(f"[!] Capture failed: {e}")
        cv2.destroyAllWindows()
        return

    cv2.destroyAllWindows()

    print("Extracting features (MediaPipe + Gabor)...")
    t0 = time.time()
    try:
        clahe_roi, code = process_image(gray, landmarker)
    except ValueError as e:
        print(f"[!] Pipeline error: {e}")
        return
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        return

    pipeline_time = time.time() - t0

    roi_preview = cv2.resize(clahe_roi, (200, 200))
    cv2.imshow("Scan ROI", roi_preview)
    cv2.waitKey(500)

    print("Matching against database...")
    t_m = time.time()
    username, score, user_id = engine.identify(code)
    match_time = time.time() - t_m
    total_time = pipeline_time + match_time

    cv2.destroyAllWindows()

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
    log_access(user_id=user_id if username else None, score=score, accepted=(username is not None))


def main():
    print("\n" + "=" * 50)
    print("  PALM VEIN AUTH — Pi 5 Terminal Test")
    print("=" * 50)

    print("Initialising database...")
    init_db()
    engine = SearchEngine(n_workers=4)

    print("Loading MediaPipe hand landmarker...")
    try:
        landmarker = build_landmarker(MODEL_PATH)
        print("[+] Landmarker ready.")
    except Exception as e:
        print(f"[!] Landmarker warning/error: {e}")
        landmarker = None

    print("Initialising camera...")
    cam, cam_type, preview_cfg, still_cfg = init_camera()
    if cam is None:
        print("[!] No camera. Exiting.")
        return

    print("\nReady. Commands:")
    print("  N = Enroll new user")
    print("  S = Scan / identify palm")
    print("  L = List enrolled users")
    print("  Q = Quit\n")

    try:
        while True:
            cmd = input("Command [N/S/L/Q]: ").strip().upper()
            if cmd == 'N':
                enroll_flow(cam, cam_type, landmarker, engine, preview_cfg, still_cfg)
            elif cmd == 'S':
                scan_flow(cam, cam_type, landmarker, engine, preview_cfg, still_cfg)
            elif cmd == 'L':
                users = list_users()
                print(f"\nEnrolled users ({len(users)}):")
                for u in users:
                    print(f"  - {u['username']} ({u['sample_count']} templates, enrolled {u['enrolled_at']})")
                print()
            elif cmd == 'Q':
                print("Exiting.")
                break
            else:
                print("Unknown command. Use N, S, L, or Q.")
    finally:
        cv2.destroyAllWindows()
        if cam_type == "picamera2" and cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass
        elif cam_type == "opencv" and cam is not None:
            cam.release()
        engine.close()


if __name__ == "__main__":
    main()
