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
from typing import Optional, List
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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.constants import (
        PROJECT_ROOT, STATIC_DIR, CAPTURE_DIR, ROI_DIR, MODEL_PATH,
        LOGS_DIR, SCAN_DIAGNOSTICS_LOG,
        MATCH_THRESHOLD, ENROLL_CONSISTENCY_THRESHOLD,
        ENROLL_SAMPLE_MIN, ENROLL_SAMPLE_MAX, ENROLLMENT_CACHE_TTL,
    )
    from app.db_manager import (
        init_db, enroll_user, user_exists, list_users,
        delete_user, log_access, get_all_embeddings,
        get_templates_by_ids, get_username, reset_all_tables,
    )
    from app.search_engine import SearchEngine
    from app.mediapipe_img import (
        build_landmarker, detect_hand_landmarks,
        extract_valleys_from_landmarks, segment_hand,
        extract_ma2017_scaled_roi, enhance_roi_vessels,
    )
    from app.cnn_extractor import extract_embedding, MODEL_LOADED
except ImportError:
    from constants import (
        PROJECT_ROOT, STATIC_DIR, CAPTURE_DIR, ROI_DIR, MODEL_PATH,
        LOGS_DIR, SCAN_DIAGNOSTICS_LOG,
        MATCH_THRESHOLD, ENROLL_CONSISTENCY_THRESHOLD,
        ENROLL_SAMPLE_MIN, ENROLL_SAMPLE_MAX, ENROLLMENT_CACHE_TTL,
    )
    from db_manager import (
        init_db, enroll_user, user_exists, list_users,
        delete_user, log_access, get_all_embeddings,
        get_templates_by_ids, get_username, reset_all_tables,
    )
    from search_engine import SearchEngine
    from mediapipe_img import (
        build_landmarker, detect_hand_landmarks,
        extract_valleys_from_landmarks, segment_hand,
        extract_ma2017_scaled_roi, enhance_roi_vessels,
    )
    from cnn_extractor import extract_embedding, MODEL_LOADED

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
        p.start()
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
    accepted: bool
    username: Optional[str]
    score: float
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
    enrolled_users_count: int
    total_templates: int
    match_threshold: float


class ReportResponse(BaseModel):
    users: List[dict]
    total_templates: int
    cross_match_summary: Optional[dict] = None


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
def capture_frame_gray() -> np.ndarray:
    """Captures a single grayscale frame thread-safely."""
    with _camera_lock:
        if CAMERA_TYPE == "picamera2" and picam2 is not None:
            arr = picam2.capture_array("main")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr

        if CAMERA_TYPE == "opencv" and cv_cap is not None:
            for _ in range(2):
                cv_cap.grab()
            ret, frame = cv_cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture frame from webcam.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        raise ValueError("No live camera available. Please connect Raspberry Pi camera or webcam.")


def process_image_with_timing(gray: np.ndarray):
    """
    Extract CLAHE ROI and AMPVNet CNN embedding from a grayscale hand frame while capturing
    granular per-stage latency (landmark detection, ROI alignment/CLAHE, CNN embedding extraction).
    Raises ValueError if palm landmarks or valleys cannot be detected.
    """
    t_land0 = time.time()
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    if landmarks is None or len(landmarks) < 21:
        raise ValueError("No hand landmarks detected. Hold palm flat ~10-15cm above camera.")

    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise ValueError("Cannot detect finger valley landmarks. Spread fingers slightly.")
    t_landmark_ms = round((time.time() - t_land0) * 1000, 2)

    t_roi0 = time.time()
    hand_mask = segment_hand(stretched)
    roi_224, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=224, scale_factor=1.6, offset_factor=0.35,
        landmarks_px=landmarks
    )
    if roi_224 is None or roi_224.size == 0:
        raise ValueError("Failed to extract palm ROI bounding box.")

    clahe_roi = enhance_roi_vessels(roi_224)
    t_roi_ms = round((time.time() - t_roi0) * 1000, 2)

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
            "capture_file": cap_rel,
            "roi_file": roi_rel,
            "decision": "ACCEPTED" if search_diag.get("accepted") else "REJECTED",
            "matched_user": search_diag.get("username"),
            "score": search_diag.get("score"),
            "threshold": search_diag.get("threshold", MATCH_THRESHOLD),
            "ranked_candidates": search_diag.get("ranked_candidates", []),
            "latency_ms": latency_breakdown,
        }

        with open(SCAN_DIAGNOSTICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[!] Warning: Failed writing to scan_diagnostics.jsonl: {e}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
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
            time.sleep(0.05)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
        "enrolled_users_count": len(users),
        "total_templates": len(emb_data["template_ids"]),
        "match_threshold": float(MATCH_THRESHOLD),
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
        raise HTTPException(status_code=503, detail="CNN model not loaded, cannot perform recognition.")

    if not CAMERA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Camera hardware not available.")

    t0 = time.time()
    t_cap0 = time.time()
    try:
        gray = await run_in_threadpool(capture_frame_gray)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    t_capture_ms = round((time.time() - t_cap0) * 1000, 2)

    try:
        clahe_roi, embedding, proc_timing = await asyncio.wait_for(
            run_in_threadpool(process_image_with_timing, gray),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Landmark detection timeout. Ensure hand is steady and properly illuminated."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {e}")

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
    username = search_diag["username"]
    score = search_diag["score"]
    user_id = search_diag["user_id"]
    accepted = search_diag["accepted"]

    # Pass resolved winning user_id when accepted=True
    await run_in_threadpool(log_access, user_id=user_id if accepted else None, score=score, accepted=accepted)
    cap_path, roi_path = await run_in_threadpool(save_capture_to_disk, gray, clahe_roi, username or "unknown", "scan")

    # Stage Latency Breakdown (Capture, Landmark, ROI, CNN Embedding, Matching, Total)
    latency_breakdown = {
        "capture": t_capture_ms,
        "landmark": proc_timing["landmark_ms"],
        "roi": proc_timing["roi_ms"],
        "cnn_embedding": proc_timing["cnn_embedding_ms"],
        "matching": search_diag["t_match_ms"],
        "total": t_total_ms,
    }

    # Structured per-scan diagnostic logging
    await run_in_threadpool(
        log_scan_diagnostic,
        cap_path, roi_path, search_diag, latency_breakdown
    )

    # Encode CLAHE ROI as base64 thumbnail
    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "accepted": accepted,
        "username": username,
        "score": float(score),
        "threshold": float(MATCH_THRESHOLD),
        "time_ms": int(t_total_ms),
        "clahe_base64": b64_roi,
    }


@app.post("/api/enroll/sample", response_model=SampleResponse)
async def enroll_sample(req: SampleReq):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="CNN model not loaded, cannot enroll"
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

    try:
        gray = await run_in_threadpool(capture_frame_gray)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        clahe_roi, embedding = await asyncio.wait_for(
            run_in_threadpool(process_image, gray),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Pipeline timeout. Move hand closer to camera and ensure good lighting."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction error: {e}")

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
