#!/usr/bin/env python3
"""
tools/pi_runtime_smoke_test.py — Dedicated Raspberry Pi Runtime Smoke Test
==========================================================================
Verifies the production edge runtime environment on Raspberry Pi 5.
Runs standalone without development dependencies (no PyTorch, no httpx).

Checks:
  1. Python 3.10+ & 64-bit Architecture
  2. OpenCV (cv2)
  3. NumPy
  4. MediaPipe & HandLandmarker model asset
  5. Picamera2 hardware driver
  6. ONNX Runtime (onnxruntime)
  7. CPUExecutionProvider
  8. Model Existence (models/ampvnet_finetuned.onnx)
  9. AMPVNetInference model session load
 10. 512-D Embedding extraction
 11. Strict L2 unit normalization (||v||_2 ~= 1.0)
 12. [Optional: --camera] Single live frame capture and pipeline test

Usage:
    python3 tools/pi_runtime_smoke_test.py
    python3 tools/pi_runtime_smoke_test.py --camera
"""

import os
import sys
import platform
import argparse
from pathlib import Path

# Suppress noisy probe logging
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FINETUNED_ONNX_PATH = _PROJECT_ROOT / "models" / "ampvnet_finetuned.onnx"
DEFAULT_ONNX_PATH = _PROJECT_ROOT / "models" / "ampvnet.onnx"
LANDMARKER_MODEL_PATH = _PROJECT_ROOT / "models" / "hand_landmarker.task"


def print_banner(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


class RuntimeSmokeTester:
    def __init__(self, camera_mode: bool = False):
        self.camera_mode = camera_mode
        self.passed_checks = 0
        self.failed_checks = 0
        self.warn_checks = 0
        self.critical_failures = []

    def log_result(self, name: str, status: str, detail: str = ""):
        symbols = {
            "PASS": "\033[92m[✓ PASS]\033[0m",
            "FAIL": "\033[91m[✗ FAIL]\033[0m",
            "WARN": "\033[93m[! WARN]\033[0m",
            "INFO": "\033[94m[i INFO]\033[0m",
        }
        sym = symbols.get(status, f"[{status}]")
        print(f"  {sym:<18} {name:<28} {detail}")

        if status == "PASS":
            self.passed_checks += 1
        elif status == "FAIL":
            self.failed_checks += 1
            self.critical_failures.append((name, detail))
        elif status == "WARN":
            self.warn_checks += 1

    def test_python_and_arch(self):
        py_ver = sys.version_info
        py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
        arch = platform.machine()
        is_64bit = sys.maxsize > 2**32

        if py_ver >= (3, 10):
            self.log_result("Python Version", "PASS", f"Python {py_str} (>= 3.10 required)")
        else:
            self.log_result("Python Version", "FAIL", f"Python {py_str} is too old (need 3.10+)")

        if is_64bit and arch in ("aarch64", "arm64", "x86_64"):
            self.log_result("System Architecture", "PASS", f"{arch} (64-bit OS)")
        elif not is_64bit:
            self.log_result("System Architecture", "FAIL",
                            f"{arch} (32-bit OS detected! Raspberry Pi OS 64-bit is required for ONNX Runtime)")
        else:
            self.log_result("System Architecture", "WARN", f"{arch} (Non-standard architecture)")

    def test_opencv(self):
        try:
            import cv2
            self.log_result("OpenCV (cv2)", "PASS", f"v{cv2.__version__}")
        except Exception as e:
            self.log_result("OpenCV (cv2)", "FAIL", f"Missing cv2 ({e}). Install via apt or pip.")

    def test_numpy(self):
        try:
            import numpy as np
            self.log_result("NumPy", "PASS", f"v{np.__version__}")
        except Exception as e:
            self.log_result("NumPy", "FAIL", f"Missing numpy ({e})")

    def test_mediapipe(self):
        try:
            import mediapipe as mp
            self.log_result("MediaPipe Package", "PASS", f"v{mp.__version__}")
        except Exception as e:
            self.log_result("MediaPipe Package", "FAIL",
                            f"Missing mediapipe ({e}). Run: pip install mediapipe")
            return

        # Check landmarker model asset
        try:
            from app.mediapipe_img import ensure_model_exists, build_landmarker
            model_path = ensure_model_exists(str(LANDMARKER_MODEL_PATH))
            if Path(model_path).exists():
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                self.log_result("HandLandmarker Asset", "PASS",
                                f"{Path(model_path).name} ({size_mb:.2f} MB)")
            else:
                self.log_result("HandLandmarker Asset", "FAIL", f"Model not found at {model_path}")
        except Exception as e:
            self.log_result("HandLandmarker Asset", "FAIL", f"Initialization error: {e}")

    def test_picamera2(self):
        try:
            from picamera2 import Picamera2
            self.log_result("Picamera2 Hardware Driver", "PASS", "Available (installed via apt)")
        except Exception as e:
            if platform.machine() in ("aarch64", "arm64") and "linux" in sys.platform:
                self.log_result(
                    "Picamera2 Hardware Driver", "FAIL",
                    f"Unavailable ({e}). Picamera2 is not accessible in this venv. "
                    "Recreate venv with: python3 -m venv --system-site-packages .venv"
                )
            else:
                self.log_result(
                    "Picamera2 Hardware Driver", "INFO",
                    "Not available on non-Pi host (Picamera2 required on physical Raspberry Pi)"
                )

    def test_onnxruntime(self):
        try:
            import onnxruntime as ort
            self.log_result("ONNX Runtime Package", "PASS", f"v{ort.__version__}")
            providers = ort.get_available_providers()
            if "CPUExecutionProvider" in providers:
                self.log_result("CPUExecutionProvider", "PASS", f"Available ({providers})")
            else:
                self.log_result("CPUExecutionProvider", "FAIL",
                                f"CPUExecutionProvider not found in {providers}")
        except Exception as e:
            self.log_result("ONNX Runtime Package", "FAIL",
                            f"Not installed ({e}). Run on Pi: pip install onnxruntime")

    def test_model_and_inference(self):
        # 1. Model existence
        active_model = None
        for p in [FINETUNED_ONNX_PATH, DEFAULT_ONNX_PATH]:
            if p.exists():
                active_model = p
                break

        if active_model is not None:
            size_mb = active_model.stat().st_size / (1024 * 1024)
            self.log_result("ONNX Model File", "PASS", f"{active_model.name} ({size_mb:.2f} MB)")
        else:
            self.log_result("ONNX Model File", "FAIL",
                            f"Missing ONNX model at {FINETUNED_ONNX_PATH} or {DEFAULT_ONNX_PATH}")
            return

        # 2. Model load
        try:
            from app.ampvnet_inference import AMPVNetInference
            engine = AMPVNetInference(str(active_model))
            if engine.model_loaded:
                self.log_result("Model Session Load", "PASS",
                                f"InferenceSession initialized on CPU")
            else:
                self.log_result("Model Session Load", "FAIL",
                                f"Model not loaded ({engine.error_detail})")
                return
        except Exception as e:
            self.log_result("Model Session Load", "FAIL", f"Exception during load: {e}")
            return

        # 3. Embedding extraction & L2 norm
        try:
            import numpy as np
            test_roi = np.full((224, 224), 128, dtype=np.uint8)
            emb = engine.extract_embedding(test_roi)

            if len(emb) == 512:
                self.log_result("512-D Embedding Extraction", "PASS",
                                f"Shape: {emb.shape}, Dtype: {emb.dtype}")
            else:
                self.log_result("512-D Embedding Extraction", "FAIL",
                                f"Expected 512 dims, got {len(emb)}")

            norm = float(np.linalg.norm(emb))
            if abs(norm - 1.0) < 1e-4:
                self.log_result("L2 Unit Normalization", "PASS", f"||v||_2 = {norm:.6f}")
            else:
                self.log_result("L2 Unit Normalization", "FAIL",
                                f"Norm deviation too large: ||v||_2 = {norm:.6f}")
        except Exception as e:
            self.log_result("Embedding Extraction", "FAIL", f"Inference execution failed: {e}")

    def test_live_camera_capture(self):
        """Optional test capturing 1 live frame from physical camera."""
        print_banner("OPTIONAL: LIVE CAMERA CAPTURE TEST (--camera)")
        import numpy as np
        import cv2

        cap = None
        cam_type = None

        # Picamera2 is the primary and required hardware camera path
        try:
            from picamera2 import Picamera2
            p = Picamera2()
            cfg = p.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            p.configure(cfg)
            p.start()
            frame_rgb = p.capture_array("main")
            p.stop()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cam_type = "REAL_PICAMERA2 (CSI)"
            self.log_result("Live Frame Acquisition", "PASS", f"Captured 640x480 via {cam_type}")
        except Exception as e:
            self.log_result(
                "Live Frame Acquisition", "FAIL",
                f"Picamera2 hardware camera not available ({e}). "
                "Hardware camera test failed — Picamera2 is required on Raspberry Pi."
            )
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.log_result("Optical Sharpness",
                        "PASS" if laplacian_var >= 45.0 else "WARN",
                        f"Laplacian Var = {laplacian_var:.1f} (Target > 45.0)")

        # Landmark & ROI pipeline check
        try:
            from app.mediapipe_img import (
                build_landmarker,
                detect_hand_landmarks_with_diagnostics,
                extract_valleys_from_landmarks,
                segment_hand,
                extract_ma2017_scaled_roi,
                compute_roi_quality,
            )
            landmarker = build_landmarker()
            diag = detect_hand_landmarks_with_diagnostics(gray, landmarker)
            if diag["success"]:
                landmarks = diag["landmarks"]
                self.log_result("Live Palm Detection", "PASS", "21 hand landmarks detected!")
                try:
                    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
                    if pv1 is not None and pv2 is not None:
                        self.log_result("Knuckle Valley Extraction", "PASS", f"Pv1={pv1}, Pv2={pv2}")
                        hand_mask = segment_hand(gray)
                        roi_res = extract_ma2017_scaled_roi(
                            gray, pv1, pv2, hand_mask,
                            target_size=224, scale_factor=1.6, offset_factor=0.35,
                            landmarks_px=landmarks
                        )
                        if roi_res is not None:
                            roi_224, bbox, _ = roi_res
                            if roi_224 is not None and roi_224.shape == (224, 224):
                                quality = compute_roi_quality(roi_224, bbox, gray.shape)
                                is_valid = quality.get("is_valid", quality.get("valid", False))
                                status = "PASS" if is_valid else "WARN"
                                self.log_result("224x224 ROI Extraction", status,
                                                f"Pad: {quality['pad_pct']*100:.1f}%, Contrast Std: {quality['contrast_std']:.1f}, Valid: {is_valid}")
                            else:
                                self.log_result("224x224 ROI Extraction", "FAIL", "Invalid ROI shape")
                        else:
                            self.log_result("224x224 ROI Extraction", "FAIL", "extract_ma2017_scaled_roi returned None")
                    else:
                        self.log_result("Knuckle Valley Extraction", "FAIL", "Could not derive knuckle valleys")
                except Exception as e:
                    self.log_result("Pipeline Execution", "FAIL", f"ROI/Valley pipeline error: {e}")
            else:
                self.log_result("Live Palm Detection", "INFO",
                                f"{diag['reason']}: {diag['instruction']}")
        except Exception as e:
            self.log_result("Live Palm Detection", "WARN", f"Landmarking check: {e}")

    def run_all(self):
        print_banner("RASPBERRY PI RUNTIME ENVIRONMENT SMOKE TEST")
        print(f"Platform : {platform.platform()}")
        print(f"Target   : Raspberry Pi 5 / Debian 12 Bookworm (ARM64)")

        print("\n--- Core Runtime Dependencies ---")
        self.test_python_and_arch()
        self.test_opencv()
        self.test_numpy()
        self.test_mediapipe()
        self.test_picamera2()
        self.test_onnxruntime()

        print("\n--- Biometric Model & Inference Pipeline ---")
        self.test_model_and_inference()

        if self.camera_mode:
            self.test_live_camera_capture()

        print_banner("TEST SUMMARY")
        print(f"  Passed Checks   : {self.passed_checks}")
        print(f"  Warnings        : {self.warn_checks}")
        print(f"  Failed Checks   : {self.failed_checks}")

        if self.failed_checks == 0:
            print("\n  \033[92m[✓] SUCCESS: Raspberry Pi runtime environment is fully operational!\033[0m")
            print("  You can now start the production backend:")
            print("      python3 -m app.server\n")
            return 0
        else:
            print("\n  \033[91m[✗] CRITICAL FAILURES DETECTED:\033[0m")
            for name, detail in self.critical_failures:
                print(f"      - {name}: {detail}")
            print("\n  Please resolve the above failures before running the backend server.\n")
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Runtime Smoke Test")
    parser.add_argument("--camera", action="store_true", help="Test live frame capture from camera")
    args = parser.parse_args()

    tester = RuntimeSmokeTester(camera_mode=args.camera)
    exit_code = tester.run_all()
    sys.exit(exit_code)
