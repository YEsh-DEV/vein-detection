#!/usr/bin/env python3
r"""
server.py
---------
FastAPI + Uvicorn backend serving the Palm Vein Biometrics API & Neobrutalism Web UI.
Directly connects to Raspberry Pi NoIR Camera (Picamera2) or USB Webcam (OpenCV).
Features thread-safe camera locking, non-blocking threadpool offloading, and typed Pydantic contracts.
"""

import os
import sys
import time
import base64
import threading
import asyncio
import re
import cv2
import numpy as np
import mimetypes
import sqlite3
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Ensure proper MIME types on all OS platforms (especially Windows)
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")

# Pipeline imports
from db_manager import (
    init_db, enroll_user, user_exists, list_users,
    delete_user, log_access, get_all_signatures,
    get_templates_by_ids, get_username, compute_signature,
    DB_PATH,
)
from search_engine import SearchEngine
from mediapipe_img import (
    build_landmarker, detect_hand_landmarks,
    extract_valleys_from_landmarks, segment_hand,
    extract_ma2017_scaled_roi, enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD

from contextlib import asynccontextmanager

# Directories
CAPTURE_DIR = "captures"
ROI_DIR = "roi_clahe"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# Camera Threading Lock (prevents concurrent access clashes between MJPEG stream and still capture)
_camera_lock = threading.Lock()

# Hardware & Engine Globals
engine = None
landmarker = None
picam2 = None
cv_cap = None
CAMERA_AVAILABLE = False
CAMERA_TYPE = None
preview_cfg = None
still_cfg = None


# ── Request Models ──────────────────────────────────────────────────────────
class SampleReq(BaseModel):
    username: str
    sample_idx: int = 0


class SaveReq(BaseModel):
    username: str


class CancelReq(BaseModel):
    username: str


# ── Response Models ─────────────────────────────────────────────────────────
class StatusResponse(BaseModel):
    status: str
    camera_available: bool
    camera_type: str
    users_count: int
    total_templates: int
    match_threshold: float


class UserItem(BaseModel):
    username: str
    sample_count: int
    enrolled_at: str


class UsersResponse(BaseModel):
    users: List[UserItem]


class ScanResponse(BaseModel):
    accepted: bool
    username: Optional[str]
    score: float
    threshold: float
    time_ms: int
    clahe_base64: Optional[str] = None


class SampleResponse(BaseModel):
    success: bool
    sample_count: int
    vr_mean: float
    thumb: str


class SaveResponse(BaseModel):
    success: bool
    username: str
    samples_stored: int


class DeleteResponse(BaseModel):
    success: bool
    deleted: str


class ResetResponse(BaseModel):
    success: bool
    message: str


class SelfMatchItem(BaseModel):
    username: str
    min_score: float
    avg_score: float
    max_score: float
    quality: str   # "GOOD" or "WARN"


class CrossMatchItem(BaseModel):
    pair: str
    score: float
    status: str   # "OK" or "WARN"


class ReportResponse(BaseModel):
    self_matches: List[SelfMatchItem]
    cross_matches: List[CrossMatchItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, landmarker, picam2, cv_cap, CAMERA_AVAILABLE, CAMERA_TYPE, preview_cfg, still_cfg
    init_db()
    engine = SearchEngine(n_workers=4)

    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
    except Exception as e:
        print(f"[!] Landmarker warning: {e}")

    # 1. Try Raspberry Pi NoIR Camera (Picamera2)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        preview_cfg = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        still_cfg = picam2.create_still_configuration(main={"size": (1640, 1232), "format": "RGB888"})
        picam2.configure(preview_cfg)
        picam2.start()
        picam2.set_controls({"AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 1.0})
        CAMERA_AVAILABLE = True
        CAMERA_TYPE = "picamera2"
        print("[+] Picamera2 initialized successfully (Raspberry Pi NoIR Camera).")
    except Exception:
        picam2 = None

    # 2. Fallback to USB Webcam / V4L2 Camera (OpenCV)
    if not CAMERA_AVAILABLE:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cv_cap = cap
                    CAMERA_AVAILABLE = True
                    CAMERA_TYPE = "opencv"
                    print("[+] OpenCV VideoCapture(0) initialized successfully.")
                else:
                    cap.release()
        except Exception as e:
            print(f"[!] OpenCV Camera error: {e}")

    if not CAMERA_AVAILABLE:
        print("[!] No live camera detected. Connect a Pi NoIR camera or USB webcam.")

    yield

    if engine:
        engine.shutdown()
    if CAMERA_TYPE == "picamera2" and picam2 is not None:
        picam2.stop()
    if CAMERA_TYPE == "opencv" and cv_cap is not None:
        cv_cap.release()

# App & Middleware
app = FastAPI(title="Palm Vein Biometrics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory temporary enrollment cache: username -> list of veincodes
enrollment_cache = {}


# Helpers
def capture_frame_gray() -> np.ndarray:
    """Capture a single live grayscale frame from camera (thread-safe)."""
    with _camera_lock:
        if CAMERA_TYPE == "picamera2" and picam2 is not None:
            picam2.switch_mode(still_cfg)
            frame = picam2.capture_array()
            picam2.switch_mode(preview_cfg)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if CAMERA_TYPE == "opencv" and cv_cap is not None:
            for _ in range(2):
                cv_cap.grab()
            ret, frame = cv_cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture frame from webcam.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        raise ValueError("No live camera available. Please connect Raspberry Pi camera or webcam.")


def process_image(gray: np.ndarray):
    """
    Extract CLAHE ROI and Gabor VeinCode from a grayscale hand frame.
    Raises ValueError if palm landmarks or valleys cannot be detected.
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


# API Endpoints
def generate_video_stream():
    """MJPEG stream of live camera for real-time positioning on screen (thread-safe)."""
    while True:
        try:
            with _camera_lock:
                if CAMERA_TYPE == "picamera2" and picam2 is not None:
                    frame = picam2.capture_array()
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif CAMERA_TYPE == "opencv" and cv_cap is not None:
                    for _ in range(2):
                        cv_cap.grab()
                    ret, frame = cv_cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue
                    bgr = frame
                else:
                    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
                    cv2.putText(bgr, "CAMERA OFFLINE", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw HUD outside the lock
            h, w = bgr.shape[:2]
            cx, cy = w // 2, h // 2
            box_sz = 120
            cv2.rectangle(bgr, (cx - box_sz, cy - box_sz), (cx + box_sz, cy + box_sz),
                          (0, 255, 200), 2)
            cv2.putText(bgr, "ALIGN PALM HERE", (cx - 95, cy - box_sz - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

            ret, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.04)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
        except Exception:
            time.sleep(0.1)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    users = await run_in_threadpool(list_users)
    return {
        "status": "online",
        "camera_available": CAMERA_AVAILABLE,
        "camera_type": CAMERA_TYPE or "None",
        "users_count": len(users),
        "total_templates": sum(u["sample_count"] for u in users),
        "match_threshold": MATCH_THRESHOLD,
    }


@app.get("/api/users", response_model=UsersResponse)
async def get_users():
    users = await run_in_threadpool(list_users)
    return {"users": users}


@app.delete("/api/users/{username}", response_model=DeleteResponse)
async def remove_user(username: str):
    exists = await run_in_threadpool(user_exists, username)
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")
    await run_in_threadpool(delete_user, username)
    await run_in_threadpool(engine.refresh_cache)
    return {"success": True, "deleted": username}


@app.delete("/api/database/reset", response_model=ResetResponse)
async def reset_database():
    """
    WARNING: Deletes ALL enrolled users and templates.
    Use this to clear test data before live Pi enrollment.
    """
    def _wipe():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM templates")
            conn.execute("DELETE FROM access_log")
            conn.execute("DELETE FROM users")
            conn.commit()
    await run_in_threadpool(_wipe)
    await run_in_threadpool(engine.refresh_cache)
    return {"success": True, "message": "All biometric data cleared. Re-enroll using live camera."}


@app.post("/api/scan", response_model=ScanResponse)
async def scan_palm():
    t0 = time.time()
    try:
        gray = await run_in_threadpool(capture_frame_gray)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        clahe_roi, probe_code = await asyncio.wait_for(
            run_in_threadpool(process_image, gray),
            timeout=15.0   # 15 seconds max for MediaPipe + Gabor on Pi 5
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Pipeline timeout. Ensure hand is visible and well-lit above camera."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {e}")

    try:
        username, score = await asyncio.wait_for(
            run_in_threadpool(engine.identify, probe_code),
            timeout=10.0   # 10 seconds max for MNHD search
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Matching timeout. Try again."
        )
    elapsed = int((time.time() - t0) * 1000)
    accepted = (username is not None)

    await run_in_threadpool(log_access, user_id=None, score=score, accepted=accepted)
    await run_in_threadpool(save_capture_to_disk, gray, clahe_roi, username or "unknown", "scan")

    # Encode CLAHE ROI as base64 thumbnail
    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "accepted": accepted,
        "username": username,
        "score": float(score),
        "threshold": float(MATCH_THRESHOLD),
        "time_ms": elapsed,
        "clahe_base64": b64_roi,
    }


@app.post("/api/enroll/sample", response_model=SampleResponse)
async def enroll_sample(req: SampleReq):
    uname = req.username.strip().lower()

    # Validate: only letters, numbers, hyphens, underscores. Min 2 chars.
    if not uname or not re.match(r'^[a-z0-9][a-z0-9_-]{1,29}$', uname):
        raise HTTPException(
            status_code=422,
            detail="Username must be 2-30 characters: letters, numbers, hyphens, underscores only."
        )

    if user_exists(uname):
        raise HTTPException(
            status_code=409,
            detail=f"User '{uname}' already enrolled. Delete first to re-enroll."
        )

    current_samples = enrollment_cache.get(uname, [])
    if len(current_samples) >= 6:
        raise HTTPException(
            status_code=400,
            detail="Maximum 6 samples reached. Save enrollment or clear and restart."
        )

    try:
        gray = await run_in_threadpool(capture_frame_gray)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        clahe_roi, code = await asyncio.wait_for(
            run_in_threadpool(process_image, gray),
            timeout=15.0   # 15 seconds max for MediaPipe + Gabor on Pi 5
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

    enrollment_cache.setdefault(uname, []).append(code)
    sample_idx = len(enrollment_cache[uname])

    await run_in_threadpool(save_capture_to_disk, gray, clahe_roi, uname, "enroll", idx=sample_idx)

    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "sample_count": sample_idx,
        "vr_mean": float(code["VR"].mean()),
        "thumb": b64_roi,
    }


@app.post("/api/enroll/save", response_model=SaveResponse)
async def save_enrollment(req: SaveReq):
    uname = req.username.strip().lower()

    if not uname or not re.match(r'^[a-z0-9][a-z0-9_-]{1,29}$', uname):
        raise HTTPException(
            status_code=422,
            detail="Username must be 2-30 characters: letters, numbers, hyphens, underscores only."
        )

    if user_exists(uname):
        raise HTTPException(
            status_code=409,
            detail=f"User '{uname}' already enrolled. Delete first to re-enroll."
        )

    samples = enrollment_cache.get(uname, [])
    if len(samples) < 3:
        # Always clear on failure too
        enrollment_cache.pop(uname, None)
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 samples. Got {len(samples)}. Start enrollment again."
        )

    try:
        await run_in_threadpool(enroll_user, uname, samples)
        await run_in_threadpool(engine.refresh_cache)
    except Exception as e:
        enrollment_cache.pop(uname, None)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    count = len(samples)
    enrollment_cache.pop(uname, None)   # Always clear after save
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
        sig_data = get_all_signatures()
        uid_map = {}
        for uid in sig_data["user_ids"]:
            if uid not in uid_map.values():
                try:
                    uid_map[get_username(uid)] = uid
                except KeyError:
                    pass

        self_matches = []
        for u in users:
            uid = uid_map.get(u["username"])
            tids = [tid for tid, uid_val in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid_val == uid]
            if len(tids) >= 2:
                tmpl = get_templates_by_ids(tids)
                scores = [match_templates(tmpl[a], tmpl[b]) for a in range(len(tmpl)) for b in range(a + 1, len(tmpl))]
                if scores:
                    mn, av, mx = min(scores), sum(scores) / len(scores), max(scores)
                    qual = "GOOD" if mx < 0.35 else "WARN"
                    self_matches.append({
                        "username": u["username"],
                        "min_score": round(float(mn), 4),
                        "avg_score": round(float(av), 4),
                        "max_score": round(float(mx), 4),
                        "quality": qual
                    })

        cross_matches = []
        if len(users) >= 2:
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u1, u2 = users[i], users[j]
                    t1 = get_templates_by_ids([tid for tid, uid in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid == uid_map.get(u1["username"])][:1])
                    t2 = get_templates_by_ids([tid for tid, uid in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid == uid_map.get(u2["username"])][:1])
                    if t1 and t2:
                        sc = match_templates(t1[0], t2[0])
                        stat = "OK" if sc > 0.45 else "WARN"
                        cross_matches.append({
                            "pair": f"{u1['username']} vs {u2['username']}",
                            "score": round(float(sc), 4),
                            "status": stat
                        })

        return {"self_matches": self_matches, "cross_matches": cross_matches}

    return await run_in_threadpool(_compute_report)


# Serve Frontend Static Assets
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            media_type, _ = mimetypes.guess_type(file_path)
            return FileResponse(file_path, media_type=media_type or "application/octet-stream")
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import uvicorn
    import webbrowser
    print("\n" + "=" * 55)
    print("  [+] PALM VEIN RECOGNITION - NEOBRUTALISM WEB APP")
    print("  [+] Server running at: http://localhost:8000")
    print("=" * 55 + "\n")
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
