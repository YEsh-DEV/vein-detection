#!/usr/bin/env python3
r"""
server.py
---------
FastAPI + Uvicorn backend serving the Palm Vein Biometrics API & Neobrutalism Web UI.
Directly connects to Raspberry Pi NoIR Camera (Picamera2) or USB Webcam (OpenCV).
Features thread-safe camera locking, non-blocking threadpool offloading, and typed Pydantic contracts.

v2 Architecture: Powered by AMPVNet CNN embeddings (ONNX Runtime) and in-memory cosine matching.
"""

import os
import sys
import time
import base64
import json
import threading
import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
try:
    cv2.setLogLevel(0)
except Exception:
    pass
import numpy as np
import mimetypes

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.constants import (
        PROJECT_ROOT, STATIC_DIR, CAPTURE_DIR, ROI_DIR, MODEL_PATH,
        LOGS_DIR, SCAN_DIAGNOSTICS_LOG, CAPTURE_DIAGNOSTICS_LOG,
        MATCH_THRESHOLD, ENROLL_CONSISTENCY_THRESHOLD,
        ENROLL_SAMPLE_MIN, ENROLL_SAMPLE_MAX, ENROLLMENT_CACHE_TTL,
        BIOMETRIC_ENGINE, DEBUG_DIAGNOSTICS_MODE, DEBUG_FRAMES_DIR,
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        BURST_CAPTURE_FRAMES, BURST_FRAME_INTERVAL_S,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        CODE_HAND_TOO_CLOSE, CODE_HAND_TOO_FAR, CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS, CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED, CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST, CODE_QUALITY_EXCESSIVE_PADDING,
        CODE_MODEL_NOT_LOADED, CODE_CAMERA_ERROR, CODE_UNKNOWN_PIPELINE_ERROR,
    )
    from app.camera_pipeline import (
        extract_nir_channel,
        create_display_frame,
        compute_frame_quality_score,
        select_best_frame,
        calculate_calibrated_exposure_and_gain,
        save_operator_debug_dump,
    )
    from app.capture_errors import CaptureError, ERROR_PRIORITY
    from app.db_manager import (
        init_db, enroll_user, user_exists, list_users,
        delete_user, log_access, get_all_embeddings,
        get_templates_by_ids, get_username, reset_all_tables,
    )
    from app.search_engine import SearchEngine
    from app.mediapipe_img import (
        build_landmarker, detect_hand_landmarks,
        detect_hand_landmarks_with_diagnostics,
        extract_valleys_from_landmarks, segment_hand,
        extract_ma2017_scaled_roi, enhance_roi_vessels,
        compute_roi_quality, draw_landmarks_overlay,
    )
    from app.cnn_extractor import extract_embedding, MODEL_LOADED, MODEL_ERROR_DETAIL
except ImportError:
    from constants import (
        PROJECT_ROOT, STATIC_DIR, CAPTURE_DIR, ROI_DIR, MODEL_PATH,
        LOGS_DIR, SCAN_DIAGNOSTICS_LOG, CAPTURE_DIAGNOSTICS_LOG,
        MATCH_THRESHOLD, ENROLL_CONSISTENCY_THRESHOLD,
        ENROLL_SAMPLE_MIN, ENROLL_SAMPLE_MAX, ENROLLMENT_CACHE_TTL,
        BIOMETRIC_ENGINE, DEBUG_DIAGNOSTICS_MODE, DEBUG_FRAMES_DIR,
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        BURST_CAPTURE_FRAMES, BURST_FRAME_INTERVAL_S,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        CODE_HAND_TOO_CLOSE, CODE_HAND_TOO_FAR, CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS, CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED, CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST, CODE_QUALITY_EXCESSIVE_PADDING,
        CODE_MODEL_NOT_LOADED, CODE_CAMERA_ERROR, CODE_UNKNOWN_PIPELINE_ERROR,
    )
    from camera_pipeline import (
        extract_nir_channel,
        create_display_frame,
        compute_frame_quality_score,
        select_best_frame,
        calculate_calibrated_exposure_and_gain,
        save_operator_debug_dump,
    )
    from capture_errors import CaptureError, ERROR_PRIORITY
    from db_manager import (
        init_db, enroll_user, user_exists, list_users,
        delete_user, log_access, get_all_embeddings,
        get_templates_by_ids, get_username, reset_all_tables,
    )
    from search_engine import SearchEngine
    from mediapipe_img import (
        build_landmarker, detect_hand_landmarks,
        detect_hand_landmarks_with_diagnostics,
        extract_valleys_from_landmarks, segment_hand,
        extract_ma2017_scaled_roi, enhance_roi_vessels,
        compute_roi_quality, draw_landmarks_overlay,
    )
    from cnn_extractor import extract_embedding, MODEL_LOADED, MODEL_ERROR_DETAIL

# Ensure proper MIME types on all OS platforms (especially Windows)
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")

# Camera Threading Lock (prevents concurrent access clashes between MJPEG stream and still capture)
_camera_lock = threading.Lock()

# Hardware & Engine Globals
engine = None
landmarker = None
picam2 = None
cv_cap = None
CAMERA_AVAILABLE = False
_debug_diagnostics_mode = DEBUG_DIAGNOSTICS_MODE
CAMERA_TYPE = None
CAMERA_DEVICE = None
CAMERA_ERROR_DETAIL = ""
preview_cfg = None
still_cfg = None


# ---------------------------------------------------------------------------
# Camera Lifecycle & Initialization
# ---------------------------------------------------------------------------
def init_hardware_camera():
    """Attempts to initialize Raspberry Pi Camera via Picamera2 or USB Webcam via OpenCV."""
    global picam2, cv_cap, CAMERA_AVAILABLE, CAMERA_TYPE, CAMERA_DEVICE, CAMERA_ERROR_DETAIL, preview_cfg, still_cfg

    picam2_err = None
    # Attempt 1: Picamera2 (Raspberry Pi native CSI camera / NoIR)
    try:
        from picamera2 import Picamera2
        p = Picamera2()
        p.configure(p.create_preview_configuration(main={"size": (640, 480), "format": "XBGR8888"}))
        # Initial safe low exposure to prevent saturation
        try:
            p.set_controls({
                "AeEnable":     False,
                "AwbEnable":    False,
                "ColourGains":  (1.0, 1.0),
                "ExposureTime": 4000,
                "AnalogueGain": 1.0,
            })
        except Exception as ctrl_err:
            print(f"[!] Warning: Failed setting initial Picamera2 controls ({ctrl_err})")

        p.start()
        time.sleep(0.5)

        # Auto-calibrate to find the right exposure for THIS LED setup
        try:
            from app.camera_pipeline import auto_calibrate_exposure
        except ImportError:
            from camera_pipeline import auto_calibrate_exposure

        try:
            best_exp, best_gain = auto_calibrate_exposure(
                p,
                target_mean=100.0,
                min_exp=1000,
                max_exp=8000,
                min_gain=1.0,
                max_gain=1.0,
                max_iterations=14,
                tolerance=6.0,
            )
            p.set_controls({
                "ExposureTime": best_exp,
                "AnalogueGain": best_gain,
            })
            print(f"[Camera] Auto-calibrated: {best_exp}µs @ gain {best_gain:.1f}")
        except Exception as cal_err:
            print(f"[!] Warning: Auto-calibration failed ({cal_err}), using 4000µs @ gain 1.0")

        picam2 = p
        CAMERA_AVAILABLE = True
        CAMERA_TYPE = "picamera2"
        CAMERA_DEVICE = "CSI"
        CAMERA_ERROR_DETAIL = ""
        print("[+] Picamera2 camera hardware initialized successfully.")
        return
    except Exception as e:
        picam2_err = str(e)
        print(f"[-] Picamera2 unavailable ({e}). Probing OpenCV V4L2 device nodes...")

    # Attempt 2: OpenCV Multi-Index VideoCapture Probe (V4L2 device index 0 through 7)
    for idx in range(8):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    cv_cap = cap
                    CAMERA_AVAILABLE = True
                    CAMERA_TYPE = "opencv"
                    CAMERA_DEVICE = f"/dev/video{idx}"
                    CAMERA_ERROR_DETAIL = ""
                    print(f"[+] OpenCV VideoCapture camera initialized successfully on /dev/video{idx}.")
                    return
                cap.release()
        except Exception as e:
            print(f"[-] OpenCV probe failed on index {idx}: {e}")

    CAMERA_AVAILABLE = False
    CAMERA_TYPE = None
    CAMERA_DEVICE = None
    CAMERA_ERROR_DETAIL = f"Picamera2: {picam2_err or 'Not detected'}; OpenCV V4L2 (0-7): No working video device found"
    print(f"[!] WARNING: Running in CAMERA-FREE mode. {CAMERA_ERROR_DETAIL}")


def release_hardware_camera():
    """Safely closes active camera handles."""
    global picam2, cv_cap, CAMERA_AVAILABLE
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        picam2 = None
    if cv_cap is not None:
        try:
            cv_cap.release()
        except Exception:
            pass
        cv_cap = None
    CAMERA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Application Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, landmarker
    print("[*] Initializing SQLite database...")
    init_db()

    print("[*] Initializing MediaPipe Hand Landmarker...")
    try:
        landmarker = build_landmarker(MODEL_PATH)
        print("[+] MediaPipe Hand Landmarker ready.")
    except Exception as e:
        print(f"[!] Warning: Hand Landmarker failed to load ({e}).")

    print("[*] Initializing Biometric Search Engine (In-RAM Cosine)...")
    engine = SearchEngine()
    print("[+] Search Engine cache loaded.")

    print("[*] Probing camera hardware...")
    init_hardware_camera()

    yield

    print("[*] Shutting down application...")
    if engine is not None:
        engine.close()
    release_hardware_camera()


# ---------------------------------------------------------------------------
# FastAPI App Definition & Static File Mounts
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Palm Vein Biometrics Server",
    description="Local Biometric Authentication System for Raspberry Pi 5 (AMPVNet CNN Engine)",
    version="2.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------
class EnrollReq(BaseModel):
    username: str


class SampleReq(BaseModel):
    username: str


class SaveReq(BaseModel):
    username: str


class CancelReq(BaseModel):
    username: str


class DeleteReq(BaseModel):
    username: str


class ScanResponse(BaseModel):
    """
    Public API response for /api/scan.

    Security boundary:
      accepted=True  → username, score, threshold all present
      accepted=False → username=null, score=-1.0 (sentinel), threshold present

    ranked_candidates and internal diagnostic scores are NEVER sent to the client.
    They are written to logs/scan_diagnostics.jsonl for research purposes only.
    """
    accepted: bool
    username: Optional[str]   # null when rejected
    score: float              # -1.0 sentinel when rejected; do NOT expose candidate score
    threshold: float
    time_ms: int
    clahe_base64: str


class SampleResponse(BaseModel):
    success: bool
    sample_count: int
    vr_mean: float
    thumb: str


class SaveResponse(BaseModel):
    success: bool
    username: str
    samples_stored: int


class StatusResponse(BaseModel):
    camera_available: bool
    camera_type: Optional[str]
    camera_device: Optional[str] = None
    camera_error: Optional[str] = None
    model_loaded: bool
    model_error: Optional[str] = None
    enrolled_users_count: int
    total_templates: int
    match_threshold: float
    biometric_engine: Optional[str] = "v2"


class ReportResponse(BaseModel):
    users: List[dict]
    total_templates: int
    cross_match_summary: Optional[dict] = None


class DebugModeReq(BaseModel):
    enabled: bool


class DebugModeResponse(BaseModel):
    enabled: bool
    frames_dir: str
    message: str


# ---------------------------------------------------------------------------
# In-Memory Enrollment Cache with TTL Expiry
# ---------------------------------------------------------------------------
# Structure: uname -> {"timestamp": float, "samples": list of np.ndarray embeddings}
enrollment_cache = {}


def _cleanup_expired_enrollment_cache():
    """Removes abandoned enrollment sessions older than ENROLLMENT_CACHE_TTL."""
    now = time.time()
    expired = [
        u for u, v in enrollment_cache.items()
        if now - v.get("timestamp", 0) > ENROLLMENT_CACHE_TTL
    ]
    for u in expired:
        enrollment_cache.pop(u, None)


# ---------------------------------------------------------------------------
# Image Processing & Capture Helpers
# ---------------------------------------------------------------------------
def capture_frame_raw_and_gray() -> Tuple[np.ndarray, np.ndarray]:
    """Captures a raw camera frame and its calibrated NIR grayscale representation thread-safely."""
    with _camera_lock:
        if CAMERA_TYPE == "picamera2" and picam2 is not None:
            arr = picam2.capture_array("main")
            gray = extract_nir_channel(arr)
            return arr, gray

        if CAMERA_TYPE == "opencv" and cv_cap is not None:
            for _ in range(2):
                cv_cap.grab()
            ret, frame = cv_cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture frame from webcam.")
            gray = extract_nir_channel(frame)
            return frame, gray

        raise ValueError("No live camera available. Please connect Raspberry Pi camera or webcam.")


def capture_frame_gray() -> np.ndarray:
    """Captures a single calibrated NIR grayscale frame thread-safely."""
    _, gray = capture_frame_raw_and_gray()
    return gray


def process_image_with_timing(gray: np.ndarray, raw_frame: Optional[np.ndarray] = None):
    """
    Extract CLAHE ROI and AMPVNet CNN embedding from a grayscale hand frame while capturing
    granular per-stage latency (landmark detection, ROI alignment/CLAHE, CNN embedding extraction).
    Raises CaptureError if palm landmarks, knuckle valleys, or ROI cannot be detected.
    """
    t_land0 = time.time()
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    if landmarks is None or len(landmarks) < 21:
        raise CaptureError(
            error_code=CODE_MEDIAPIPE_NO_LANDMARKS,
            instruction="Hand landmarks not detected. Hold palm flat ~10-15cm above camera.",
            stage="mediapipe"
        )

    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise CaptureError(
            error_code=CODE_VALLEY_EXTRACTION_FAILED,
            instruction="Cannot detect finger valley landmarks. Spread fingers slightly.",
            stage="valleys",
            diagnostics={"landmarks_count": len(landmarks)}
        )
    t_landmark_ms = round((time.time() - t_land0) * 1000, 2)

    t_roi0 = time.time()
    hand_mask = segment_hand(stretched)
    roi_224, bbox, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=224, scale_factor=1.6, offset_factor=0.35,
        landmarks_px=landmarks
    )
    if roi_224 is None or roi_224.size == 0:
        raise CaptureError(
            error_code=CODE_ROI_EXTRACTION_FAILED,
            instruction="Failed to extract palm ROI bounding box.",
            stage="roi"
        )

    clahe_roi = enhance_roi_vessels(roi_224)
    t_roi_ms = round((time.time() - t_roi0) * 1000, 2)

    # Diagnostic frame export if debug mode is active (Task 9)
    if _debug_diagnostics_mode:
        try:
            overlay = draw_landmarks_overlay(stretched, landmarks, pv1, pv2)
            quality_info = compute_roi_quality(roi_224, bbox, stretched.shape)
            score_info = compute_frame_quality_score(
                gray,
                roi_224=roi_224,
                pad_pct=quality_info.get("pad_pct", 0.0)
            )
            save_operator_debug_dump(
                raw_frame=raw_frame if raw_frame is not None else gray,
                processed_gray=gray,
                landmarks_overlay=overlay,
                roi_raw=roi_224,
                roi_enhanced=clahe_roi,
                diagnostics={
                    "resolution": f"{gray.shape[1]}x{gray.shape[0]}",
                    "exposure_us": DEFAULT_EXPOSURE_US,
                    "analogue_gain": DEFAULT_ANALOGUE_GAIN,
                    "mean": float(np.mean(gray)),
                    "contrast_std": float(np.std(gray)),
                    "dynamic_range": [int(np.min(gray)), int(np.max(gray))],
                    "saturation_pct": round(float(np.count_nonzero(gray >= 250)) / gray.size * 100.0, 2),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                    "roi_bbox": [int(x) for x in bbox] if bbox is not None else None,
                    "roi_padding_pct": quality_info.get("pad_pct", 0.0),
                    "roi_contrast_std": quality_info.get("contrast_std", 0.0),
                    "selected_frame_score": score_info.get("score", 0.0),
                }
            )
        except Exception as e:
            print(f"[!] Warning: Failed to save operator debug frames: {e}")

    t_cnn0 = time.time()
    embedding = extract_embedding(clahe_roi)
    t_cnn_ms = round((time.time() - t_cnn0) * 1000, 2)

    timing = {
        'landmark_ms': t_landmark_ms,
        'roi_ms': t_roi_ms,
        'cnn_embedding_ms': t_cnn_ms,
    }
    return clahe_roi, embedding, timing


def process_image(gray: np.ndarray):
    """Backward-compatible wrapper returning (clahe_roi, embedding)."""
    clahe_roi, embedding, _ = process_image_with_timing(gray)
    return clahe_roi, embedding


def process_enrollment_sample(gray: np.ndarray):
    """
    Extracts CLAHE ROI and AMPVNet embedding with Phase 3 Enrollment Quality Gate checks:
      1. Valid landmarks & knuckle valleys
      2. Boundary padding <= 25.0%
      3. Valid 224x224 dimensions
      4. Contrast std >= 10.0
      5. Embedding L2 norm == 1.0 +- 1e-3
    Raises CaptureError with actionable machine-readable code & UI guidance if check fails.
    """
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    diag_res = detect_hand_landmarks_with_diagnostics(stretched, landmarker)
    if not diag_res["success"]:
        stage = "positioning" if diag_res["reason"] in (CODE_HAND_TOO_CLOSE, CODE_HAND_TOO_FAR, CODE_HAND_OUTSIDE_FRAME) else "mediapipe"
        raise CaptureError(
            error_code=diag_res["reason"],
            instruction=diag_res["instruction"],
            stage=stage,
            diagnostics=diag_res.get("diagnostics")
        )

    landmarks = diag_res["landmarks"]
    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise CaptureError(
            error_code=CODE_VALLEY_EXTRACTION_FAILED,
            instruction="Cannot detect finger valleys. Spread fingers slightly and hold palm flat.",
            stage="valleys",
            diagnostics={"landmarks_count": len(landmarks)}
        )

    hand_mask = segment_hand(stretched)
    roi_224, bbox, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=224, scale_factor=1.6, offset_factor=0.35,
        landmarks_px=landmarks
    )
    if roi_224 is None or roi_224.size == 0:
        raise CaptureError(
            error_code=CODE_ROI_EXTRACTION_FAILED,
            instruction="Failed to extract palm ROI bounding box.",
            stage="roi"
        )

    # Phase 3 Quality Gate
    quality = compute_roi_quality(roi_224, bbox, stretched.shape)
    if not quality["is_valid"]:
        reasons_str = "; ".join(quality["reasons"])
        err_code = quality.get("error_code") or (CODE_QUALITY_EXCESSIVE_PADDING if quality["pad_pct"] > 0.25 else CODE_QUALITY_LOW_CONTRAST)
        raise CaptureError(
            error_code=err_code,
            instruction=f"Quality gate rejected sample: {reasons_str}. Please reposition palm.",
            stage="quality_gate",
            diagnostics={
                "pad_pct": quality["pad_pct"],
                "contrast_std": quality["contrast_std"],
                "mean_intensity": quality["mean_intensity"],
                "reasons": quality["reasons"],
            }
        )

    clahe_roi = enhance_roi_vessels(roi_224)
    embedding = extract_embedding(clahe_roi)

    # Embedding L2 norm check
    norm_val = float(np.linalg.norm(embedding))
    if abs(norm_val - 1.0) > 1e-3:
        raise CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction=f"Degenerate embedding norm ({norm_val:.4f} != 1.0). Please re-capture.",
            stage="cnn_embedding",
            diagnostics={"norm": norm_val}
        )

    return clahe_roi, embedding, quality


def prune_scan_captures(max_count: int = 200, max_age_seconds: int = 48 * 3600):
    """
    Bounded data retention policy for temporary scan-time captures only.
    CRITICAL SAFETY: Strictly inspects files containing '_scan_' in their filename.
    NEVER touches or deletes any enrollment captures, templates, or database records.
    Keeps at most max_count (200) most recent scan files and purges files older than 48 hours.
    """
    now = time.time()
    for directory in (CAPTURE_DIR, ROI_DIR):
        if not os.path.exists(directory):
            continue
        try:
            scan_files = []
            for fname in os.listdir(directory):
                if "_scan_" in fname and fname.lower().endswith((".png", ".jpg")):
                    fpath = os.path.join(directory, fname)
                    if os.path.isfile(fpath):
                        mtime = os.path.getmtime(fpath)
                        scan_files.append((fpath, mtime))

            # Sort descending by mtime (newest first)
            scan_files.sort(key=lambda x: x[1], reverse=True)

            # Prune files exceeding max_count or older than 48h
            for idx, (fpath, mtime) in enumerate(scan_files):
                is_expired = (now - mtime) > max_age_seconds
                exceeds_count = (idx >= max_count)
                if is_expired or exceeds_count:
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
        except Exception as e:
            print(f"[!] Warning: Scan capture pruning error in {directory}: {e}")


def save_capture_to_disk(gray: np.ndarray, roi: np.ndarray, username: str, mode: str, idx: int = 0):
    """Save raw capture and CLAHE ROI to captures/ and roi_clahe/, returning their paths."""
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
            print(f"[+] Saved biometric capture: {cap_path} & {roi_path}")
        else:
            print(f"[!] Warning: cv2.imwrite failed (cap={ok1}, roi={ok2}) for {cap_path}")

        # Enforce bounded retention on scan captures
        if mode == "scan":
            prune_scan_captures(max_count=200, max_age_seconds=48 * 3600)

        return cap_path, roi_path
    except Exception as e:
        print(f"[!] Warning: Failed saving capture to disk: {e}")
        return None, None


def log_scan_diagnostic(
    cap_path: Optional[str],
    roi_path: Optional[str],
    search_diag: dict,
    latency_breakdown: dict
):
    """
    Appends a structured diagnostic JSON line to logs/scan_diagnostics.jsonl.
    Separate from console/uvicorn log for programmatic parsing and offline analysis.
    """
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        cap_rel = os.path.relpath(cap_path, PROJECT_ROOT) if cap_path else None
        roi_rel = os.path.relpath(roi_path, PROJECT_ROOT) if roi_path else None

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": "scan",
            "engine": BIOMETRIC_ENGINE,
            "capture_file": cap_rel,
            "roi_file": roi_rel,
            "decision": "ACCEPTED" if search_diag.get("accepted") else "REJECTED",
            "matched_user": search_diag.get("username"),
            "matched_user_id": search_diag.get("user_id"),
            "score": search_diag.get("score"),
            "threshold": search_diag.get("threshold", MATCH_THRESHOLD),
            "ranked_candidates": search_diag.get("ranked_candidates", []),
            "latency_ms": latency_breakdown,
        }

        with open(SCAN_DIAGNOSTICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[!] Warning: Failed writing to scan_diagnostics.jsonl: {e}")


def log_capture_failure_diagnostic(
    error: CaptureError,
    attempts: int = 1,
    attempt_details: list = None
):
    """
    Appends a structured JSON line to logs/capture_diagnostics.jsonl for dev diagnostics (Task 4).
    Never exposes internal tracebacks to the client.
    Captures:
      - timestamp
      - stage
      - error_code
      - instruction
      - occupancy
      - border_touches
      - contrast
      - padding
      - mediapipe_landmarks (bool)
      - roi_status (str)
      - attempts_total
      - attempt_details
    """
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        diag = getattr(error, "diagnostics", {}) or {}
        if not isinstance(diag, dict):
            diag = {}

        # Determine MediaPipe landmark detection status
        stage = getattr(error, "stage", "pipeline")
        if "landmarks_count" in diag:
            has_landmarks = bool(diag["landmarks_count"] >= 21)
        elif stage in ("valleys", "roi", "quality_gate", "cnn_embedding"):
            has_landmarks = True
        else:
            has_landmarks = False

        # Determine ROI status
        if stage == "quality_gate":
            roi_status = "rejected_quality"
        elif stage in ("positioning", "mediapipe", "valleys", "camera", "model"):
            roi_status = "not_reached"
        elif stage == "roi":
            roi_status = "failed_extraction"
        else:
            roi_status = "extracted"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "error_code": getattr(error, "error_code", CODE_UNKNOWN_PIPELINE_ERROR),
            "instruction": getattr(error, "instruction", str(error)),
            "occupancy": float(diag.get("occupancy_pct", 0.0)),
            "border_touches": int(diag.get("border_touches", 0)),
            "contrast": float(diag.get("contrast_std", 0.0)),
            "padding": float(diag.get("pad_pct", 0.0)),
            "mediapipe_landmarks": has_landmarks,
            "roi_status": roi_status,
            "attempts_total": attempts,
            "attempt_details": attempt_details or [],
        }

        with open(CAPTURE_DIAGNOSTICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[!] Warning: Failed writing to capture_diagnostics.jsonl: {e}")


def capture_burst_and_process_enrollment(
    uname: str,
    max_frames: int = BURST_CAPTURE_FRAMES,
    frame_interval_s: float = BURST_FRAME_INTERVAL_S
):
    """
    Acquires a short burst of frames during enrollment, runs landmark & quality gates,
    and applies Best-Frame Selection (Task 5).
    Selects the single best real frame based on contrast std and Laplacian sharpness.
    Zero frame blending, zero synthetic averaging.
    Returns (gray, clahe_roi, embedding, quality, attempt_idx) of the top-scoring frame.
    If all candidate frames fail, aggregates diagnostics and raises the most actionable CaptureError.
    """
    if not CAMERA_AVAILABLE:
        err = CaptureError(
            error_code=CODE_CAMERA_ERROR,
            instruction="Camera hardware not available. Check camera connection.",
            stage="camera",
            diagnostics={"camera_available": False, "detail": CAMERA_ERROR_DETAIL}
        )
        log_capture_failure_diagnostic(err, attempts=0, attempt_details=[err.to_dict()])
        raise err

    attempt_errors = []
    valid_candidates = []

    for attempt_idx in range(1, max_frames + 1):
        try:
            gray = capture_frame_gray()
            clahe_roi, embedding, quality = process_enrollment_sample(gray)
            score_dict = compute_frame_quality_score(
                gray,
                roi_224=clahe_roi,
                pad_pct=quality.get("pad_pct", 0.0)
            )
            valid_candidates.append({
                "gray": gray,
                "clahe_roi": clahe_roi,
                "embedding": embedding,
                "quality": quality,
                "score_dict": score_dict,
                "attempt_idx": attempt_idx,
            })
        except CaptureError as ce:
            attempt_errors.append(ce)
        except Exception as ex:
            attempt_errors.append(
                CaptureError(
                    error_code=CODE_UNKNOWN_PIPELINE_ERROR,
                    instruction=f"Pipeline error: {ex}",
                    stage="pipeline",
                    diagnostics={"error": str(ex)}
                )
            )

        if attempt_idx < max_frames:
            time.sleep(frame_interval_s)

    # Best-Frame Selection: Pick the top-scoring real frame from candidates
    if valid_candidates:
        best = select_best_frame(valid_candidates)
        return (
            best["gray"],
            best["clahe_roi"],
            best["embedding"],
            best["quality"],
            best["attempt_idx"],
        )

    # All frames failed — select the most informative error based on priority
    if attempt_errors:
        chosen_error = max(attempt_errors, key=lambda e: getattr(e, "priority", 10))
    else:
        chosen_error = CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction="No valid frames acquired during burst.",
            stage="pipeline"
        )

    log_capture_failure_diagnostic(
        error=chosen_error,
        attempts=len(attempt_errors),
        attempt_details=[e.to_dict() if hasattr(e, "to_dict") else {"detail": str(e)} for e in attempt_errors]
    )

    raise chosen_error


def capture_burst_and_process_scan(
    max_frames: int = 3,
    frame_interval_s: float = 0.04
):
    """
    Acquires up to 3 candidate frames during scan, evaluates landmark & ROI validity,
    and applies Best-Frame Selection (Task 5) to pick the real frame with highest optical quality.
    """
    if not CAMERA_AVAILABLE:
        raise CaptureError(
            error_code=CODE_CAMERA_ERROR,
            instruction="Camera hardware not available.",
            stage="camera",
            diagnostics={"camera_available": False, "detail": CAMERA_ERROR_DETAIL}
        )

    attempt_errors = []
    valid_candidates = []

    for attempt_idx in range(1, max_frames + 1):
        try:
            gray = capture_frame_gray()
            clahe_roi, embedding, proc_timing = process_image_with_timing(gray)
            score_dict = compute_frame_quality_score(gray, roi_224=clahe_roi)
            valid_candidates.append({
                "gray": gray,
                "clahe_roi": clahe_roi,
                "embedding": embedding,
                "proc_timing": proc_timing,
                "score_dict": score_dict,
                "attempt_idx": attempt_idx,
            })
        except CaptureError as ce:
            attempt_errors.append(ce)
        except Exception as ex:
            attempt_errors.append(
                CaptureError(
                    error_code=CODE_UNKNOWN_PIPELINE_ERROR,
                    instruction=f"Pipeline error: {ex}",
                    stage="pipeline",
                    diagnostics={"error": str(ex)}
                )
            )

        if attempt_idx < max_frames:
            time.sleep(frame_interval_s)

    if valid_candidates:
        best = select_best_frame(valid_candidates)
        return (
            best["gray"],
            best["clahe_roi"],
            best["embedding"],
            best["proc_timing"]
        )

    if attempt_errors:
        chosen_error = max(attempt_errors, key=lambda e: getattr(e, "priority", 10))
    else:
        chosen_error = CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction="No valid scan frames acquired during burst.",
            stage="pipeline"
        )
    raise chosen_error


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
def health_check():
    """Simple system health and engine readiness endpoint."""
    return {
        "status": "healthy",
        "engine": BIOMETRIC_ENGINE,
        "model_loaded": (landmarker is not None and MODEL_LOADED),
        "model_error": MODEL_ERROR_DETAIL if not MODEL_LOADED else None,
        "camera_available": CAMERA_AVAILABLE,
    }


def generate_video_stream():
    """Generates MJPEG multipart streaming response."""
    while True:
        if not CAMERA_AVAILABLE:
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            cv2.putText(frame, "CAMERA UNAVAILABLE", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.2)
            continue

        with _camera_lock:
            frame = None
            if CAMERA_TYPE == "picamera2" and picam2 is not None:
                frame = picam2.capture_array("main")
            elif CAMERA_TYPE == "opencv" and cv_cap is not None:
                ret, frame = cv_cap.read()
                if not ret:
                    frame = None

        if frame is None:
            continue

        # Transform raw camera frame into clean monochrome enhanced visualization (Task 7)
        disp_frame = create_display_frame(frame)
        ret, buf = cv2.imencode(".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.get("/video_feed")
@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    users = await run_in_threadpool(list_users)
    emb_data = await run_in_threadpool(get_all_embeddings)
    return {
        "camera_available": CAMERA_AVAILABLE,
        "camera_type": CAMERA_TYPE,
        "camera_device": CAMERA_DEVICE,
        "camera_error": CAMERA_ERROR_DETAIL if not CAMERA_AVAILABLE else None,
        "model_loaded": (landmarker is not None and MODEL_LOADED),
        "model_error": MODEL_ERROR_DETAIL if not MODEL_LOADED else None,
        "enrolled_users_count": len(users),
        "total_templates": len(emb_data["template_ids"]),
        "match_threshold": float(MATCH_THRESHOLD),
        "biometric_engine": BIOMETRIC_ENGINE,
    }


@app.post("/api/database/reset")
@app.delete("/api/database/reset")
async def reset_database():
    """Danger zone: completely wipes all database tables and clears enrollment cache."""
    try:
        await run_in_threadpool(reset_all_tables)
        enrollment_cache.clear()
        await run_in_threadpool(engine.refresh_cache)
        return {"success": True, "message": "Database wiped successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset error: {e}")


@app.post("/api/scan", response_model=ScanResponse)
async def scan_palm():
    if not MODEL_LOADED:
        err_msg = (
            f"Biometric model not loaded: {MODEL_ERROR_DETAIL}"
            if MODEL_ERROR_DETAIL
            else "CNN model not loaded, cannot perform recognition."
        )
        raise HTTPException(status_code=503, detail=err_msg)

    if not CAMERA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Camera hardware not available.")

    t0 = time.time()
    t_cap0 = time.time()
    try:
        gray, clahe_roi, embedding, proc_timing = await asyncio.wait_for(
            run_in_threadpool(capture_burst_and_process_scan, 3, 0.04),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        err = CaptureError(
            error_code="PIPELINE_TIMEOUT",
            instruction="Landmark detection timeout. Ensure hand is steady and properly illuminated.",
            stage="pipeline"
        )
        log_capture_failure_diagnostic(err, attempts=1, attempt_details=[err.to_dict()])
        return JSONResponse(status_code=504, content=err.to_dict())
    except CaptureError as e:
        log_capture_failure_diagnostic(e, attempts=1, attempt_details=[e.to_dict()])
        return JSONResponse(status_code=400, content=e.to_dict())
    except ValueError as e:
        err = CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction=str(e),
            stage="pipeline"
        )
        log_capture_failure_diagnostic(err, attempts=1, attempt_details=[err.to_dict()])
        return JSONResponse(status_code=400, content=err.to_dict())
    except Exception as e:
        err = CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction=f"Pipeline error: {e}",
            stage="pipeline",
            diagnostics={"error": str(e)}
        )
        log_capture_failure_diagnostic(err, attempts=1, attempt_details=[err.to_dict()])
        return JSONResponse(status_code=500, content=err.to_dict())
    t_capture_ms = round((time.time() - t_cap0) * 1000, 2)

    try:
        search_diag = await asyncio.wait_for(
            run_in_threadpool(engine.identify_with_diagnostics, embedding),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Matching timeout. Try again."
        )

    t_total_ms = round((time.time() - t0) * 1000, 2)
    # Internal diagnostics — kept server-side only, never sent to client when rejected
    _internal_username = search_diag["username"]    # None when rejected
    _internal_score    = search_diag["score"]        # best cosine score (internal)
    _internal_user_id  = search_diag["user_id"]     # None when rejected
    accepted           = search_diag["accepted"]

    # Log the threshold used for this decision
    print(
        f"[scan] threshold={MATCH_THRESHOLD:.4f} "
        f"best_score={_internal_score:.4f} "
        f"accepted={accepted} "
        f"candidate={'<hidden>' if not accepted else _internal_username}"
    )

    # Pass resolved winning user_id when accepted=True; NULL for rejected (no identity reveal)
    await run_in_threadpool(
        log_access,
        user_id=_internal_user_id if accepted else None,
        score=_internal_score,     # store actual score internally for audit
        accepted=accepted,
        engine=BIOMETRIC_ENGINE
    )
    cap_path, roi_path = await run_in_threadpool(
        save_capture_to_disk, gray, clahe_roi, _internal_username or "unknown", "scan"
    )

    # Stage Latency Breakdown (Capture, Landmark, ROI, CNN Embedding, Matching, Total)
    latency_breakdown = {
        "capture": t_capture_ms,
        "landmark": proc_timing["landmark_ms"],
        "roi": proc_timing["roi_ms"],
        "cnn_embedding": proc_timing["cnn_embedding_ms"],
        "matching": search_diag["t_match_ms"],
        "total": t_total_ms,
    }

    # Internal JSONL diagnostic log — contains full ranked_candidates for research
    # IMPORTANT: this data MUST NOT be forwarded to the HTTP response
    await run_in_threadpool(
        log_scan_diagnostic,
        cap_path, roi_path, search_diag, latency_breakdown
    )

    # Encode CLAHE ROI as base64 thumbnail
    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    # ── PUBLIC RESPONSE BOUNDARY ──────────────────────────────────────────────
    # SECURITY: Rejected results MUST NOT expose the nearest candidate identity,
    #           score, ranking, or template IDs. The internal diagnostics above
    #           remain in logs only.
    #
    # accepted=True:  username=<enrolled name>, score=actual similarity
    # accepted=False: username=null,            score=-1.0 (opaque sentinel)
    # ─────────────────────────────────────────────────────────────────────────
    public_username = _internal_username if accepted else None
    public_score    = float(_internal_score) if accepted else -1.0

    return {
        "accepted": accepted,
        "username": public_username,
        "score": public_score,
        "threshold": float(MATCH_THRESHOLD),
        "time_ms": int(t_total_ms),
        "clahe_base64": b64_roi,
    }


@app.post("/api/enroll/sample", response_model=SampleResponse)
async def enroll_sample(req: SampleReq):
    if not MODEL_LOADED:
        err_msg = (
            f"Biometric model not loaded: {MODEL_ERROR_DETAIL}"
            if MODEL_ERROR_DETAIL
            else "CNN model not loaded, cannot enroll."
        )
        raise HTTPException(
            status_code=503,
            detail=err_msg
        )

    _cleanup_expired_enrollment_cache()
    uname = req.username.strip().lower()

    if not uname or not re.match(r'^[a-z0-9][a-z0-9_-]{1,29}$', uname):
        raise HTTPException(
            status_code=422,
            detail="Username must be 2-30 characters: letters, numbers, hyphens, underscores only."
        )

    if await run_in_threadpool(user_exists, uname):
        raise HTTPException(
            status_code=409,
            detail=f"User '{uname}' already enrolled. Delete first to re-enroll."
        )

    entry = enrollment_cache.setdefault(uname, {"timestamp": time.time(), "samples": []})
    entry["timestamp"] = time.time()
    current_samples = entry["samples"]

    if len(current_samples) >= ENROLL_SAMPLE_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {ENROLL_SAMPLE_MAX} samples reached. Save enrollment or clear and restart."
        )

    # Multi-Frame Burst Capture (Task 3: Tolerant to momentary single-frame fluctuations)
    try:
        gray, clahe_roi, embedding, quality, sample_attempt = await asyncio.wait_for(
            run_in_threadpool(capture_burst_and_process_enrollment, uname, 5, 0.08),
            timeout=20.0
        )
    except asyncio.TimeoutError:
        err = CaptureError(
            error_code="PIPELINE_TIMEOUT",
            instruction="Pipeline timeout during multi-frame acquisition. Hold hand steady and ensure good lighting.",
            stage="pipeline"
        )
        log_capture_failure_diagnostic(err, attempts=5, attempt_details=[err.to_dict()])
        return JSONResponse(status_code=504, content=err.to_dict())
    except CaptureError as e:
        return JSONResponse(status_code=400, content=e.to_dict())
    except Exception as e:
        err = CaptureError(
            error_code=CODE_UNKNOWN_PIPELINE_ERROR,
            instruction=f"Quality check error: {e}",
            stage="pipeline",
            diagnostics={"error": str(e)}
        )
        log_capture_failure_diagnostic(err, attempts=1, attempt_details=[err.to_dict()])
        return JSONResponse(status_code=400, content=err.to_dict())

    current_samples.append(embedding)
    sample_idx = len(current_samples)

    await run_in_threadpool(save_capture_to_disk, gray, clahe_roi, uname, "enroll", idx=sample_idx)

    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "sample_count": sample_idx,
        "vr_mean": 1.0,
        "thumb": b64_roi,
    }


@app.post("/api/enroll/save", response_model=SaveResponse)
async def save_enrollment(req: SaveReq):
    _cleanup_expired_enrollment_cache()
    uname = req.username.strip().lower()

    if not uname or not re.match(r'^[a-z0-9][a-z0-9_-]{1,29}$', uname):
        raise HTTPException(
            status_code=422,
            detail="Username must be 2-30 characters: letters, numbers, hyphens, underscores only."
        )

    if await run_in_threadpool(user_exists, uname):
        raise HTTPException(
            status_code=409,
            detail=f"User '{uname}' already enrolled. Delete first to re-enroll."
        )

    entry = enrollment_cache.get(uname)
    samples = entry["samples"] if entry else []

    if len(samples) < ENROLL_SAMPLE_MIN:
        enrollment_cache.pop(uname, None)
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {ENROLL_SAMPLE_MIN} samples. Got {len(samples)}. Start enrollment again."
        )

    try:
        await run_in_threadpool(enroll_user, uname, samples)
        await run_in_threadpool(engine.refresh_cache)
    except Exception as e:
        enrollment_cache.pop(uname, None)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    count = len(samples)
    enrollment_cache.pop(uname, None)
    return {"success": True, "username": uname, "samples_stored": count}


@app.post("/api/enroll/cancel")
async def cancel_enrollment(req: CancelReq):
    """Clear any partial enrollment cache for this username."""
    uname = req.username.strip().lower()
    cleared = uname in enrollment_cache
    enrollment_cache.pop(uname, None)
    return {"cleared": cleared, "username": uname}


@app.get("/api/report", response_model=ReportResponse)
async def get_report():
    def _compute_report():
        users = list_users()
        emb_data = get_all_embeddings()
        return {
            "users": users,
            "total_templates": len(emb_data["template_ids"]),
        }

    data = await run_in_threadpool(_compute_report)
    return data


@app.delete("/api/users/{username}")
async def remove_user(username: str):
    uname = username.strip().lower()
    try:
        await run_in_threadpool(delete_user, uname)
        await run_in_threadpool(engine.refresh_cache)
        return {"success": True, "message": f"User '{uname}' deleted."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {e}")


# ---------------------------------------------------------------------------
# Debug & Diagnostic Mode Endpoints (Phase A4)
# ---------------------------------------------------------------------------
@app.get("/api/debug/mode", response_model=DebugModeResponse)
def get_debug_mode():
    return {
        "enabled": _debug_diagnostics_mode,
        "frames_dir": str(DEBUG_FRAMES_DIR),
        "message": f"Debug mode is {'enabled' if _debug_diagnostics_mode else 'disabled'}"
    }


@app.post("/api/debug/mode", response_model=DebugModeResponse)
def set_debug_mode(req: DebugModeReq):
    global _debug_diagnostics_mode
    _debug_diagnostics_mode = req.enabled
    return {
        "enabled": _debug_diagnostics_mode,
        "frames_dir": str(DEBUG_FRAMES_DIR),
        "message": f"Debug mode {'enabled' if _debug_diagnostics_mode else 'disabled'}"
    }


@app.delete("/api/debug/frames")
def clear_debug_frames():
    deleted_count = 0
    if os.path.exists(DEBUG_FRAMES_DIR):
        for f in os.listdir(DEBUG_FRAMES_DIR):
            fp = os.path.join(DEBUG_FRAMES_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
                deleted_count += 1
    return {"success": True, "deleted_count": deleted_count, "message": "Debug frames directory cleared."}


# ---------------------------------------------------------------------------
# Frontend Static Asset Serving (fallback route for Single Page App)
# ---------------------------------------------------------------------------
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "Backend running. Static frontend not found."}


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("HEADLESS"):
        import webbrowser
        def _open_browser():
            time.sleep(1.0)
            try:
                webbrowser.open("http://localhost:8000")
            except Exception:
                pass
            threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
