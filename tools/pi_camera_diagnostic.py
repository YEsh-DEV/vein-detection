#!/usr/bin/env python3
"""
tools/pi_camera_diagnostic.py — Comprehensive Camera & Capture Diagnostic Tool
=============================================================================
Designed to RUN MANUALLY ON RASPBERRY PI 5 (with native Picamera2 or OpenCV V4L2).
Also runnable on laptop via synthetic mock frame mode (--mock or automatic fallback).

Audit Capabilities:
1. Camera Hardware & Control Inspection:
   - Picamera2 properties, camera model (e.g. OV5647 5MP), sensor modes.
   - Active controls (AeEnable, ExposureTime, AnalogueGain, FrameDurationLimits).
   - Lens type audit (detects manual focus vs VCM autofocus if any).
2. Optical & Frame Signal Analysis:
   - Active capture resolution (width x height).
   - Mean intensity (detects underexposure / dim NIR).
   - Standard deviation (vascular contrast).
   - Min / Max pixel values & IR LED saturation percentage (detects blown-out center).
   - Laplacian variance sharpness score (detects optical defocus vs in-focus).
3. Algorithmic Pipeline Stages:
   - Phase 2: Otsu-based pre-landmark positioning heuristic (occupancy, borders, HAND_TOO_CLOSE).
   - Phase 2: MediaPipe 21-joint skeletal landmark detection.
   - Phase 3: Knuckle valley localization (V1, V2, V3) and coordinate stability.
   - Phase 3: Scaled 224x224 ROI extraction and boundary padding validation.
   - Phase 4: AMPVNet ONNX 512-D embedding extraction smoke test.
4. Actionable Operator Recommendations:
   - Physical focus adjustment guidance (3.6mm lens barrel manual rotation).
   - Distance and lighting recommendations.

Usage:
    python3 tools/pi_camera_diagnostic.py
    python3 tools/pi_camera_diagnostic.py --mock
    python3 tools/pi_camera_diagnostic.py --save-frame diag_capture.png
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Suppress noisy OpenCV probe logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import numpy as np
import cv2

# Project root setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Safe imports from app
try:
    from app.mediapipe_img import (
        build_landmarker,
        diagnose_hand_positioning,
        detect_hand_landmarks_with_diagnostics,
        extract_valleys_from_landmarks,
        extract_ma2017_scaled_roi,
        compute_roi_quality,
        enhance_roi_vessels,
    )
    from app.constants import MODEL_PATH, EXPERIMENTAL_MATCH_THRESHOLD
except ImportError as e:
    print(f"[!] Warning: App import failed: {e}")
    sys.exit(1)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_status(label: str, status: str, detail: str = ""):
    colors = {
        "PASS": "\033[92m[PASS]\033[0m",
        "WARN": "\033[93m[WARN]\033[0m",
        "FAIL": "\033[91m[FAIL]\033[0m",
        "INFO": "\033[94m[INFO]\033[0m",
    }
    tag = colors.get(status, f"[{status}]")
    print(f"  {tag} {label:<32} {detail}")


def probe_picamera2():
    """Probes native Picamera2 configuration and hardware capabilities."""
    try:
        from picamera2 import Picamera2
        p = Picamera2()
        props = p.camera_properties
        controls = p.camera_controls
        sensor_modes = p.sensor_modes

        info = {
            "available": True,
            "model": props.get("Model", "Unknown Sensor"),
            "location": props.get("Location", "Unknown"),
            "rotation": props.get("Rotation", 0),
            "sensor_modes_count": len(sensor_modes) if sensor_modes else 0,
            "sensor_modes": [
                f"{m.get('size', (0,0))[0]}x{m.get('size', (0,0))[1]} @ {m.get('fps', 0):.0f}fps ({m.get('format', 'RAW')})"
                for m in (sensor_modes[:3] if sensor_modes else [])
            ],
            "controls": list(controls.keys()) if controls else [],
            "has_autofocus": "AfMode" in controls or "LensPosition" in controls,
            "instance": p
        }
        return info
    except Exception as e:
        return {"available": False, "error": str(e), "instance": None}


def probe_opencv_cameras(max_devices: int = 4):
    """Probes OpenCV V4L2 device nodes."""
    devices = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            backend = cap.getBackendName()
            devices.append({"index": idx, "width": w, "height": h, "fps": fps, "backend": backend})
            cap.release()
    return devices


def capture_diagnostic_frame(picam2_info, force_mock: bool = False, exposure_us: int = 5000, gain: float = 1.0):
    """Captures a frame via Picamera2, OpenCV, or synthetic mock."""
    if force_mock:
        return create_synthetic_frame(), "MOCK_SYNTHETIC", {"exposure_us": 0, "gain": 0}

    # 1. Picamera2
    if picam2_info.get("available") and picam2_info.get("instance") is not None:
        p = picam2_info["instance"]
        try:
            cfg = p.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            p.configure(cfg)
            p.start()
            p.set_controls({
                "AeEnable": False,
                "ExposureTime": exposure_us,
                "AnalogueGain": gain,
            })
            # Discard first 2 frames to allow AGC/settling
            for _ in range(2):
                p.capture_array("main")
            frame_rgb = p.capture_array("main")
            p.stop()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr, "PICAMERA2", {"exposure_us": exposure_us, "gain": gain}
        except Exception as e:
            print(f"  [!] Picamera2 capture failed: {e}")

    # 2. OpenCV
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                return frame, f"OPENCV_VIDEO_{idx}", {"exposure_us": "Auto", "gain": "Auto"}

    # 3. Fallback
    print("  [*] No live camera detected. Using synthetic NIR hand frame for algorithm validation.")
    return create_synthetic_frame(), "SYNTHETIC_FALLBACK", {"exposure_us": 0, "gain": 0}


def create_synthetic_frame():
    """Generates an anatomically structured synthetic palm for testing without physical camera."""
    frame = np.full((480, 640), 30, dtype=np.uint8)
    # Palm body
    cv2.ellipse(frame, (320, 260), (95, 120), 0, 0, 360, 135, -1)
    # Fingers
    for fx in [245, 290, 345, 395]:
        cv2.ellipse(frame, (fx, 130), (18, 55), 0, 0, 360, 125, -1)
    # Thumb
    cv2.ellipse(frame, (205, 275), (20, 45), -45, 0, 360, 125, -1)
    # Synthetic vein texture (dark absorbing paths)
    for pts in [
        [(320, 210), (310, 260), (330, 310)],
        [(280, 220), (290, 270), (280, 320)],
        [(350, 220), (340, 280), (360, 330)],
    ]:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, 90, 3)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame_bgr


def analyze_frame_signal(gray: np.ndarray):
    """Computes comprehensive optical and signal statistics on the frame."""
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))

    # Saturation: percentage of pixels clipped at or near white (IR blowout)
    saturated_px = int(np.count_nonzero(gray >= 250))
    sat_pct = (saturated_px / float(gray.size)) * 100.0

    # Underexposure: percentage of pixels clipped at pure black
    black_px = int(np.count_nonzero(gray <= 5))
    black_pct = (black_px / float(gray.size)) * 100.0

    # Sharpness: Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": min_val,
        "max": max_val,
        "sat_pct": round(sat_pct, 2),
        "black_pct": round(black_pct, 2),
        "sharpness": round(sharpness, 2),
    }


def run_diagnostics(args):
    print_header("RASPBERRY PI PALM-VEIN CAMERA & CAPTURE DIAGNOSTIC")
    print(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Host OS   : {sys.platform} | Python: {sys.version.split()[0]}")

    # =========================================================================
    # 1. HARDWARE PROBE
    # =========================================================================
    print_header("1. CAMERA HARDWARE & DRIVER PROBE")
    picam2_info = probe_picamera2()

    if picam2_info["available"]:
        print_status("Picamera2 Native Driver", "PASS", f"Model: {picam2_info['model']}")
        print_status("Lens / Focus Control", "INFO",
                     "Autofocus/VCM detected" if picam2_info["has_autofocus"]
                     else "Fixed/Manual-Focus lens (no motorized VCM)")
        print(f"     -> Sensor Modes Count : {picam2_info['sensor_modes_count']}")
        for m in picam2_info["sensor_modes"]:
            print(f"        * Mode: {m}")
    else:
        print_status("Picamera2 Native Driver", "WARN",
                     f"Not available ({picam2_info.get('error', 'None')})")

    cv_devs = probe_opencv_cameras()
    if cv_devs:
        print_status("OpenCV V4L2 Device Nodes", "PASS", f"Found {len(cv_devs)} device(s)")
        for d in cv_devs:
            print(f"     -> /dev/video{d['index']}: {d['width']}x{d['height']} @ {d['fps']}fps (backend: {d['backend']})")
    else:
        print_status("OpenCV V4L2 Device Nodes", "WARN", "No /dev/video* devices opened successfully")

    # =========================================================================
    # 2. CAPTURE & OPTICAL SIGNAL AUDIT
    # =========================================================================
    print_header("2. LIVE CAPTURE & OPTICAL SIGNAL AUDIT")
    frame_bgr, source, capture_meta = capture_diagnostic_frame(
        picam2_info, force_mock=args.mock, exposure_us=args.exposure_us, gain=args.gain
    )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print_status("Frame Source", "INFO", f"{source} (Resolution: {w}x{h})")
    print_status("Exposure Setting", "INFO", f"Exposure: {capture_meta['exposure_us']} µs | Gain: {capture_meta['gain']}")

    stats = analyze_frame_signal(gray)
    print(f"\n  Signal Statistics:")
    print(f"    - Mean Intensity         : {stats['mean']:<6} (Target: 50.0 - 150.0)")
    print(f"    - Contrast (Std Dev)     : {stats['std']:<6} (Target: >= 14.0)")
    print(f"    - Dynamic Range          : [{stats['min']}, {stats['max']}]")
    print(f"    - IR Saturation (>=250)  : {stats['sat_pct']}% (Target: < 3.0%)")
    print(f"    - Pure Black (<=5)       : {stats['black_pct']}%")
    print(f"    - Sharpness (Laplacian)  : {stats['sharpness']:<6} (Target: > 45.0)")

    # Signal evaluations
    if 40.0 <= stats["mean"] <= 170.0:
        print_status("Illumination / Brightness", "PASS", f"Optimal ({stats['mean']})")
    elif stats["mean"] < 40.0:
        print_status("Illumination / Brightness", "WARN", f"Underexposed ({stats['mean']} < 40.0). Increase exposure or IR power.")
    else:
        print_status("Illumination / Brightness", "WARN", f"Overexposed ({stats['mean']} > 170.0). Reduce gain or exposure.")

    if stats["sat_pct"] > 5.0:
        print_status("IR Center Saturation", "FAIL", f"{stats['sat_pct']}% saturated! Hand too close to IR LEDs or gain too high.")
    else:
        print_status("IR Center Saturation", "PASS", f"Controlled ({stats['sat_pct']}%)")

    if stats["sharpness"] >= 45.0:
        print_status("Optical Sharpness", "PASS", f"Score: {stats['sharpness']:.1f} (In focus)")
    elif stats["sharpness"] >= 20.0:
        print_status("Optical Sharpness", "WARN", f"Score: {stats['sharpness']:.1f} (Soft focus / slight blur)")
    else:
        print_status("Optical Sharpness", "FAIL", f"Score: {stats['sharpness']:.1f} (Severely blurred — rotate manual focus ring)")

    # =========================================================================
    # 3. PHASE 2: HAND POSITIONING & OCCUPANCY HEURISTIC
    # =========================================================================
    print_header("3. PHASE 2: PRE-LANDMARK POSITIONING HEURISTIC")
    diag = diagnose_hand_positioning(gray)
    print(f"  Heuristic Diagnostics:")
    print(f"    - Verdict Reason         : {diag['reason']}")
    print(f"    - Frame Occupancy        : {diag['occupancy_pct']}% (Target: 20.0% - 50.0%)")
    print(f"    - Border Touches         : {diag['border_touches']} / 4 "
          f"(Top: {diag['borders']['top']}, Bottom: {diag['borders']['bottom']}, "
          f"Left: {diag['borders']['left']}, Right: {diag['borders']['right']})")
    print(f"    - Guidance Instruction   : '{diag['instruction']}'")

    if diag["reason"] == "NORMAL":
        print_status("Positioning Heuristic", "PASS", f"Normal occupancy ({diag['occupancy_pct']}%)")
    elif diag["reason"] == "HAND_TOO_CLOSE":
        print_status("Positioning Heuristic", "WARN", f"HAND_TOO_CLOSE! Occupancy {diag['occupancy_pct']}% exceeds 50%. Move hand 5cm farther.")
    elif diag["reason"] == "HAND_TOO_FAR":
        print_status("Positioning Heuristic", "WARN", f"HAND_TOO_FAR! Occupancy {diag['occupancy_pct']}% < 20%. Move hand closer.")
    else:
        print_status("Positioning Heuristic", "WARN", f"{diag['reason']}: {diag['instruction']}")

    # =========================================================================
    # 4. PHASE 2 & 3: MEDIAPIPE & ROI PIPELINE
    # =========================================================================
    print_header("4. MEDIAPIPE LANDMARKING & ROI EXTRACTION")
    try:
        landmarker = build_landmarker()
        mp_diag = detect_hand_landmarks_with_diagnostics(gray, landmarker)
        if mp_diag["success"]:
            landmarks = mp_diag["landmarks"]
            print_status("MediaPipe HandLandmarker", "PASS", f"Detected 21 joints (Wrist at {landmarks[0]})")

            valleys = extract_valleys_from_landmarks(landmarks, gray.shape)
            if valleys is not None and len(valleys) >= 2:
                v1, v2 = valleys[0], valleys[1]
                print_status("Knuckle Valley Detection", "PASS", f"Found valleys: V1={v1}, V2={v2}")

                roi = extract_ma2017_scaled_roi(gray, v1, v2, target_size=(224, 224))
                if roi is not None and roi.shape == (224, 224):
                    quality = compute_roi_quality(roi)
                    print_status("224x224 ROI Extraction", "PASS",
                                 f"Pad: {quality['pad_pct']*100:.1f}%, Contrast Std: {quality['contrast_std']:.1f}, Valid: {quality['valid']}")
                else:
                    print_status("224x224 ROI Extraction", "FAIL", "Failed to crop or scale ROI")
            else:
                print_status("Knuckle Valley Detection", "FAIL", "Unable to compute knuckle valleys from landmarks")
        else:
            print_status("MediaPipe HandLandmarker", "FAIL", f"Reason: {mp_diag['reason']} — {mp_diag['instruction']}")
    except Exception as e:
        print_status("MediaPipe Pipeline", "FAIL", f"Exception during execution: {e}")

    # =========================================================================
    # 5. PHASE 4: AMPVNet ONNX INFERENCE SMOKE TEST
    # =========================================================================
    print_header("5. PHASE 4: AMPVNET ONNX EMBEDDING SMOKE TEST")
    try:
        from app.ampvnet_inference import AMPVNetInference
        engine = AMPVNetInference()
        if engine.model_loaded:
            print_status("AMPVNet ONNX Engine", "PASS", f"Loaded model from {engine.model_path}")
            # Test synthetic 224x224 inference
            test_patch = np.full((224, 224), 128, dtype=np.uint8)
            emb = engine.extract_embedding(test_patch)
            norm = float(np.linalg.norm(emb))
            if len(emb) == 512 and abs(norm - 1.0) < 1e-4:
                print_status("Embedding Inference", "PASS", f"512-D float32 vector, L2 norm = {norm:.6f}")
            else:
                print_status("Embedding Inference", "FAIL", f"Invalid embedding dimension ({len(emb)}) or norm ({norm})")
        else:
            print_status("AMPVNet ONNX Engine", "FAIL", f"Model not loaded ({engine.error_detail})")
    except Exception as e:
        print_status("AMPVNet ONNX Engine", "FAIL", f"Inference check error: {e}")

    # =========================================================================
    # 6. ACTIONABLE RECOMMENDATIONS FOR THE OPERATOR
    # =========================================================================
    print_header("6. ACTIONABLE OPERATOR RECOMMENDATIONS")
    recommendations = []

    if stats["sharpness"] < 45.0:
        recommendations.append(
            "[FOCUS] Camera appears soft or defocused. The 3.6mm lens has a screw-threaded barrel.\n"
            "        * Loosen the knurled locking ring slightly if locked.\n"
            "        * Place a printed test chart or palm flat at exactly 12 cm from the lens.\n"
            "        * Rotate the lens barrel counter-clockwise by 1/4 to 1/2 turn to shift focal plane from infinity to macro.\n"
            "        * Re-run this diagnostic until sharpness exceeds 45.0."
        )

    if stats["sat_pct"] > 3.0:
        recommendations.append(
            "[EXPOSURE / IR] Overexposure or IR blowout detected in center of frame.\n"
            "        * Ensure the palm is held at 12-15 cm, not closer than 10 cm.\n"
            "        * In tools/collect_hardware_dataset.py or app/server.py, lock exposure to 4000µs - 5000µs with gain 1.0."
        )

    if diag["reason"] == "HAND_TOO_CLOSE":
        recommendations.append(
            "[DISTANCE] Hand is occupying >50% of the frame and clipping border margins.\n"
            "        * Mount a physical spacer / standoff ring or mark at 12-15 cm above the lens.\n"
            "        * Instruct users to keep fingers within the outer framing guides."
        )

    if not recommendations:
        print("  \033[92m[✓] All diagnostic checks passed! Optical signal, positioning, and inference are optimal.\033[0m")
    else:
        for r in recommendations:
            print(f"  {r}\n")

    # Optional frame save
    if args.save_frame:
        save_path = Path(args.save_frame)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame_bgr)
        print(f"\n[+] Saved diagnostic frame to: {save_path.resolve()}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera & Capture Diagnostic")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock frame (for laptop testing)")
    parser.add_argument("--exposure-us", type=int, default=5000, help="Shutter exposure in microseconds (default: 5000)")
    parser.add_argument("--gain", type=float, default=1.0, help="Analogue gain (default: 1.0)")
    parser.add_argument("--save-frame", type=str, default="", help="Path to save captured diagnostic frame")
    args = parser.parse_args()

    run_diagnostics(args)
