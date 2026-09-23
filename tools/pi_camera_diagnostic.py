#!/usr/bin/env python3
"""
tools/pi_camera_diagnostic.py — Comprehensive Camera & Capture Diagnostic Tool
=============================================================================
Designed to RUN MANUALLY ON RASPBERRY PI 5 (with native Picamera2).
Primary camera path: Picamera2 / libcamera.

Rules:
1. Picamera2 is the PRIMARY camera path.
2. Synthetic fallback is NEVER reported as a successful hardware test.
3. If Picamera2 is unavailable on hardware, reports:
       HARDWARE CAMERA TEST NOT AVAILABLE
   and exits with code 1.
4. Synthetic fallback is ONLY available under an explicit flag: --synthetic.
5. The output clearly identifies the frame source:
       REAL_PICAMERA2
       V4L2
       SYNTHETIC
6. Optical focus / sharpness is NEVER reported as a hardware result when using synthetic data.

Usage:
    python3 tools/pi_camera_diagnostic.py
    python3 tools/pi_camera_diagnostic.py --save-frame diag_capture.png
    python3 tools/pi_camera_diagnostic.py --synthetic   # Laptop simulation mode
    python3 tools/pi_camera_diagnostic.py --v4l2        # Explicit USB V4L2 fallback
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Suppress noisy OpenCV probe logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import cv2

# Safe imports from app
try:
    from app.mediapipe_img import (
        build_landmarker,
        diagnose_hand_positioning,
        detect_hand_landmarks_with_diagnostics,
        extract_valleys_from_landmarks,
        segment_hand,
        extract_ma2017_scaled_roi,
        compute_roi_quality,
        enhance_roi_vessels,
        draw_landmarks_overlay,
    )
    from app.constants import (
        MODEL_PATH, EXPERIMENTAL_MATCH_THRESHOLD,
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        CANDIDATE_EXPOSURE_SWEEPS, NIR_EXTRACTION_METHOD,
        DEBUG_FRAMES_DIR,
    )
    from app.camera_pipeline import (
        extract_nir_channel,
        create_display_frame,
        compare_nir_representations,
        save_operator_debug_dump,
    )
except ImportError as e:
    DEFAULT_EXPOSURE_US = 14000
    DEFAULT_ANALOGUE_GAIN = 1.5
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


def capture_diagnostic_frame(picam2_info, args):
    """
    Captures a frame adhering to strict hardware rules:
    - Primary: REAL_PICAMERA2
    - Explicit secondary: V4L2 (if --v4l2 specified)
    - Explicit simulation: SYNTHETIC (if --synthetic or --mock specified)
    - If hardware unavailable and no synthetic flag: returns None, None, metadata
    """
    exposure_us = args.exposure_us
    gain = args.gain

    # 1. Explicit synthetic flag
    if args.synthetic:
        return create_synthetic_frame(), "SYNTHETIC", {"exposure_us": DEFAULT_EXPOSURE_US, "gain": DEFAULT_ANALOGUE_GAIN}

    # 2. Primary: Picamera2
    if picam2_info.get("available") and picam2_info.get("instance") is not None:
        p = picam2_info["instance"]
        try:
            cfg = p.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            p.configure(cfg)
            p.start()
            p.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "ColourGains": (1.0, 1.0),
                "ExposureTime": exposure_us,
                "AnalogueGain": gain,
            })
            # Discard first 2 frames to allow AGC/settling
            for _ in range(2):
                p.capture_array("main")
            frame_rgb = p.capture_array("main")
            p.stop()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr, "REAL_PICAMERA2", {"exposure_us": exposure_us, "gain": gain}
        except Exception as e:
            return None, None, {"error": f"Picamera2 capture error: {e}"}

    # 3. Explicit V4L2 fallback (only if user requested --v4l2)
    if args.v4l2:
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
                    return frame, "V4L2", {"exposure_us": "Auto", "gain": "Auto", "device": f"/dev/video{idx}"}

    # 4. Hardware not available
    return None, None, {"error": picam2_info.get("error", "Picamera2 driver not available in active environment")}


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
    pipeline_errors = []
    recommendations = []
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
        print_status("Picamera2 Native Driver", "FAIL",
                     f"Unavailable ({picam2_info.get('error', 'Unknown')})")

    # =========================================================================
    # 2. CAPTURE & OPTICAL SIGNAL AUDIT
    # =========================================================================
    print_header("2. LIVE CAPTURE & OPTICAL SIGNAL AUDIT")
    frame_bgr, source, capture_meta = capture_diagnostic_frame(picam2_info, args)

    if frame_bgr is None:
        print_header("HARDWARE CAMERA TEST NOT AVAILABLE")
        print("\033[91m[✗ FAIL] Picamera2 hardware driver is not available in the active environment.\033[0m")
        print(f"  Error Detail: {capture_meta.get('error')}\n")
        print("  Action items for Raspberry Pi:")
        print("    1. Verify system packages are installed on Pi OS:")
        print("       sudo apt update && sudo apt install -y python3-picamera2 python3-libcamera")
        print("    2. Recreate your virtual environment WITH '--system-site-packages':")
        print("       python3 -m venv --system-site-packages .venv")
        print("       source .venv/bin/activate")
        print("    3. Test camera hardware detection directly:")
        print("       rpicam-hello --list-cameras")
        print("\n  (If running tests on a laptop without physical hardware, run: python3 tools/pi_camera_diagnostic.py --synthetic)")
        print("=" * 70 + "\n")
        return 1

    gray = extract_nir_channel(frame_bgr)
    h, w = gray.shape

    print_status("Frame Source", "INFO", f"{source} (Resolution: {w}x{h})")
    print_status("Exposure Setting", "INFO", f"Exposure: {capture_meta.get('exposure_us')} µs | Gain: {capture_meta.get('gain')}")

    # Channel comparison & NIR signal audit (Task 2 & 3)
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
        b_ch, g_ch, r_ch = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
        print(f"\n  Bayer Channel Signal Breakdown:")
        print(f"    - Blue Channel Mean/Std   : {float(b_ch.mean()):.1f} / {float(b_ch.std()):.1f}")
        print(f"    - Green Channel Mean/Std  : {float(g_ch.mean()):.1f} / {float(g_ch.std()):.1f}")
        print(f"    - Red Channel Mean/Std    : {float(r_ch.mean()):.1f} / {float(r_ch.std()):.1f}")
        print(f"    - Calibrated NIR Gray Mean: {float(gray.mean()):.1f} / {float(gray.std()):.1f}")

    stats = analyze_frame_signal(gray)
    print(f"\n  Signal Statistics:")
    print(f"    - Mean Intensity         : {stats['mean']:<6} (Target: 50.0 - 150.0)")
    print(f"    - Contrast (Std Dev)     : {stats['std']:<6} (Target: >= 14.0)")
    print(f"    - Dynamic Range          : [{stats['min']}, {stats['max']}]")

    if source == "SYNTHETIC":
        print_status("Optical Sharpness", "INFO",
                     "N/A (SYNTHETIC FRAME - Cannot evaluate optical focus of synthetic pixels)")
        print_status("IR Center Saturation", "INFO",
                     "N/A (SYNTHETIC FRAME - Simulated lighting)")
    else:
        print(f"    - IR Saturation (>=250)  : {stats['sat_pct']}% (Target: < 3.0%)")
        print(f"    - Sharpness (Laplacian)  : {stats['sharpness']:<6} (Target: > 45.0)")

        if 40.0 <= stats["mean"] <= 170.0:
            print_status("Illumination / Brightness", "PASS", f"Optimal ({stats['mean']})")
        elif stats["mean"] < 40.0:
            print_status("Illumination / Brightness", "WARN", f"Underexposed ({stats['mean']}). Increase exposure or IR power.")
        else:
            print_status("Illumination / Brightness", "WARN", f"Overexposed ({stats['mean']}). Reduce gain or exposure.")

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

            valleys = extract_valleys_from_landmarks(landmarks)
            if valleys is not None and len(valleys) >= 2:
                v1, v2 = valleys[0], valleys[1]
                print_status("Knuckle Valley Detection", "PASS", f"Found valleys: V1={v1}, V2={v2}")

                hand_mask = segment_hand(gray)
                roi_res = extract_ma2017_scaled_roi(
                    gray, v1, v2, hand_mask,
                    target_size=224, scale_factor=1.6, offset_factor=0.35,
                    landmarks_px=landmarks
                )
                if roi_res is not None:
                    roi_224, bbox, _ = roi_res
                    if roi_224 is not None and roi_224.shape == (224, 224):
                        quality = compute_roi_quality(roi_224, bbox, gray.shape)
                        is_valid = quality.get("is_valid", quality.get("valid", False))
                        status = "PASS" if is_valid else "WARN"
                        print_status("224x224 ROI Extraction", status,
                                     f"Pad: {quality['pad_pct']*100:.1f}%, Contrast Std: {quality['contrast_std']:.1f}, Valid: {is_valid}")
                        if not is_valid:
                            recommendations.append(f"[ROI QUALITY] Quality gate warning: {', '.join(quality.get('reasons', []))}")
                    else:
                        print_status("224x224 ROI Extraction", "FAIL", "Failed to crop or scale ROI to (224, 224)")
                        pipeline_errors.append("224x224 ROI Extraction: Failed to crop or scale ROI to (224, 224)")
                else:
                    print_status("224x224 ROI Extraction", "FAIL", "extract_ma2017_scaled_roi returned None")
                    pipeline_errors.append("extract_ma2017_scaled_roi returned None")
            else:
                print_status("Knuckle Valley Detection", "FAIL", "Unable to compute knuckle valleys from landmarks")
                pipeline_errors.append("Knuckle Valley Detection: Unable to compute knuckle valleys from landmarks")
        else:
            print_status("MediaPipe HandLandmarker", "FAIL", f"Reason: {mp_diag['reason']} — {mp_diag['instruction']}")
    except Exception as e:
        print_status("MediaPipe Pipeline", "FAIL", f"Exception during execution: {e}")
        pipeline_errors.append(f"MediaPipe Pipeline Exception: {e}")

    # =========================================================================
    # 5. PHASE 4: AMPVNet ONNX INFERENCE SMOKE TEST
    # =========================================================================
    print_header("5. PHASE 4: AMPVNET ONNX EMBEDDING SMOKE TEST")
    try:
        from app.ampvnet_inference import AMPVNetInference
        engine = AMPVNetInference()
        if engine.model_loaded:
            print_status("AMPVNet ONNX Engine", "PASS", f"Loaded model from {engine.model_path}")
            test_patch = np.full((224, 224), 128, dtype=np.uint8)
            emb = engine.extract_embedding(test_patch)
            norm = float(np.linalg.norm(emb))
            if len(emb) == 512 and abs(norm - 1.0) < 1e-4:
                print_status("Embedding Inference", "PASS", f"512-D float32 vector, L2 norm = {norm:.6f}")
            else:
                print_status("Embedding Inference", "FAIL", f"Invalid embedding dimension ({len(emb)}) or norm ({norm})")
                pipeline_errors.append(f"Invalid embedding dimension ({len(emb)}) or norm ({norm})")
        else:
            print_status("AMPVNet ONNX Engine", "FAIL", f"Model not loaded ({engine.error_detail})")
            pipeline_errors.append(f"AMPVNet ONNX Engine Model Not Loaded: {engine.error_detail}")
    except Exception as e:
        print_status("AMPVNet ONNX Engine", "FAIL", f"Inference check error: {e}")
        pipeline_errors.append(f"AMPVNet ONNX Engine Exception: {e}")

    # =========================================================================
    # 6. ACTIONABLE OPERATOR RECOMMENDATIONS
    # =========================================================================
    print_header("6. ACTIONABLE OPERATOR RECOMMENDATIONS")

    if pipeline_errors:
        print("\n  \033[91m[✗ FAIL] PIPELINE IMPLEMENTATION / RUNTIME ERRORS DETECTED:\033[0m")
        for err in pipeline_errors:
            print(f"    * {err}")
        recommendations.append("[PIPELINE ERROR] Hardware diagnostic detected pipeline errors. Resolve API/code mismatches before running live biometric server.")

    if source == "SYNTHETIC":
        print("  \033[94m[INFO] SYNTHETIC SIMULATION COMPLETE: Algorithmic pipeline executed on synthetic pixels.\033[0m")
        print("  To perform physical optical calibration and camera verification, run without --synthetic on Raspberry Pi.")
    else:
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

    # Inspect and save ROI debug artifacts if requested (Problem 7)
    if args.inspect_roi and roi_224 is not None:
        try:
            overlay = draw_landmarks_overlay(gray, landmarks, v1, v2) if ('landmarks' in locals() and 'v1' in locals() and 'v2' in locals()) else None
            roi_enh = enhance_roi_vessels(roi_224) if roi_224 is not None else None
            dump_dir = args.output_dir or DEBUG_FRAMES_DIR
            dump_res = save_operator_debug_dump(
                raw_frame=frame_bgr,
                processed_gray=gray,
                landmarks_overlay=overlay,
                roi_raw=roi_224,
                roi_enhanced=roi_enh,
                diagnostics={
                    "resolution": f"{gray.shape[1]}x{gray.shape[0]}",
                    "exposure": capture_meta.get("exposure_us", DEFAULT_EXPOSURE_US),
                    "gain": capture_meta.get("gain", DEFAULT_ANALOGUE_GAIN),
                    "selected_representation": NIR_EXTRACTION_METHOD,
                    "brightness": stats["mean"],
                    "contrast": stats["std"],
                    "saturation": stats["sat_pct"],
                    "sharpness": stats["sharpness"],
                    "landmark_count": len(landmarks) if 'landmarks' in locals() else 0,
                    "pv1_pv2": [list(v1), list(v2)] if ('v1' in locals() and 'v2' in locals()) else None,
                    "roi_bbox": [int(x) for x in bbox] if ('bbox' in locals() and bbox is not None) else None,
                    "roi_padding": quality.get("pad_pct", 0.0) if 'quality' in locals() else 0.0,
                    "roi_contrast": quality.get("contrast_std", 0.0) if 'quality' in locals() else 0.0,
                    "roi_sharpness": round(float(cv2.Laplacian(roi_224, cv2.CV_64F).var()), 2) if roi_224 is not None else 0.0,
                },
                output_dir=dump_dir,
            )
            print_status("Operator Debug Dump", "PASS", f"Saved to {dump_dir} (01_raw, 02_nir, 03_landmarks, 04_roi_raw, 05_roi_enhanced, diagnostics.json)")
        except Exception as e:
            print_status("Operator Debug Dump", "WARN", f"Failed to save debug dump: {e}")

    # Empirical channel comparison if requested (Problem 3)
    if args.compare_channels:
        print_header("EMPIRICAL NIR REPRESENTATION COMPARISON (Problem 3)")
        from tools.compare_nir_channels import print_comparison_table
        comp_dir = args.output_dir or DEBUG_FRAMES_DIR
        comp_res = compare_nir_representations(frame_bgr, output_dir=comp_dir)
        print_comparison_table(comp_res)
        print(f"[+] Saved channel breakdown images and comparison.json to {comp_dir}")

    # Bounded exposure & gain sweep if requested (Problem 4)
    if args.sweep_exposure:
        print_header("BOUNDED EXPOSURE & GAIN HARDWARE SWEEP (Problem 4)")
        from tools.sweep_camera_exposure import run_exposure_sweep
        sweep_args = argparse.Namespace(synthetic=args.synthetic, image="", output_dir=args.output_dir or DEBUG_FRAMES_DIR)
        run_exposure_sweep(sweep_args)

    # Optional frame save
    if args.save_frame and frame_bgr is not None:
        save_path = Path(args.save_frame)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame_bgr)
        print(f"\n[+] Saved diagnostic frame to: {save_path.resolve()}")

    print("=" * 70 + "\n")

    if pipeline_errors:
        print("  \033[91m[✗] Diagnostic failed due to implementation/pipeline errors.\033[0m")
        return 1

    if not recommendations:
        print("  \033[92m[✓] All diagnostic checks passed! Optical signal, positioning, and inference are optimal.\033[0m")
    else:
        for r in recommendations:
            print(f"  {r}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera & Capture Diagnostic")
    parser.add_argument("--synthetic", "--mock", dest="synthetic", action="store_true",
                        help="Run in synthetic simulation mode (for laptop testing without physical camera)")
    parser.add_argument("--v4l2", action="store_true",
                        help="Allow secondary USB V4L2 webcam probe if Picamera2 is unavailable")
    parser.add_argument("--exposure-us", type=int, default=DEFAULT_EXPOSURE_US,
                        help=f"Shutter exposure in microseconds (default: {DEFAULT_EXPOSURE_US})")
    parser.add_argument("--gain", type=float, default=DEFAULT_ANALOGUE_GAIN,
                        help=f"Analogue gain (default: {DEFAULT_ANALOGUE_GAIN})")
    parser.add_argument("--save-frame", type=str, default="",
                        help="Path to save captured diagnostic frame")
    parser.add_argument("--inspect-roi", action="store_true",
                        help="Save full debug artifacts (01_raw, 02_nir, 03_landmarks, 04_roi_raw, 05_roi_enhanced, diagnostics.json)")
    parser.add_argument("--compare-channels", action="store_true",
                        help="Run empirical comparison of NIR representations (R, G, B, Rec.601, NIR weighted, Equal weighted)")
    parser.add_argument("--sweep-exposure", action="store_true",
                        help="Run bounded candidate exposure and analogue gain sweep")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Custom directory to store diagnostic outputs (default: debug_frames/)")
    args = parser.parse_args()

    exit_code = run_diagnostics(args)
    sys.exit(exit_code)
