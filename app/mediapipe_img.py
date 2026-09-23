#!/usr/bin/env python3
"""
mediapipe_img.py
-----------------
MediaPipe Hand Landmark Detection & Canonical ROI Extraction.
Replaces contour convexity defects with 21 anatomical joint landmarks.
"""

import os
import cv2
import numpy as np
import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    BaseOptions = None
    HandLandmarker = None
    HandLandmarkerOptions = None
    RunningMode = None
    MEDIAPIPE_AVAILABLE = False

try:
    from app.constants import (
        MODEL_PATH,
        CODE_HAND_TOO_CLOSE,
        CODE_HAND_TOO_FAR,
        CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS,
        CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED,
        CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST,
        CODE_QUALITY_EXCESSIVE_PADDING,
    )
    from app.capture_errors import CaptureError
except ImportError:
    from constants import (
        MODEL_PATH,
        CODE_HAND_TOO_CLOSE,
        CODE_HAND_TOO_FAR,
        CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS,
        CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED,
        CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST,
        CODE_QUALITY_EXCESSIVE_PADDING,
    )
    from capture_errors import CaptureError

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_exists(model_path: str = MODEL_PATH) -> str:
    """Checks if MediaPipe model exists, downloading it automatically if missing."""
    if not os.path.exists(model_path):
        print(f"[*] MediaPipe model '{model_path}' not found. Downloading (~8MB)...")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("[+] MediaPipe model download complete!")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to auto-download MediaPipe model: {e}\n"
                f"Please run manually:\n"
                f"  curl -L -o {model_path} {MODEL_URL}"
            )
    return model_path


def build_landmarker(model_path: str = MODEL_PATH):
    """Creates and returns a persistent HandLandmarker instance."""
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe package is not installed. Please install with: pip install mediapipe")
    ensure_model_exists(model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    return HandLandmarker.create_from_options(options)


def diagnose_hand_positioning(gray_img: np.ndarray) -> dict:
    """
    Lightweight deterministic pre-landmark hand presence and frame occupancy heuristic (Phase 2).
    Runs in <1 ms on CPU using Otsu thresholding and boundary touch analysis.
    Distinguishes:
      - HAND_TOO_CLOSE: Palm too close to lens, causing boundary truncation.
      - HAND_TOO_FAR: Hand is too distant, occupying insufficient frame area.
      - HAND_OUTSIDE_FRAME: No hand in view or severely off-center.
      - INSUFFICIENT_VISIBILITY: Low contrast or underexposed NIR capture.
      - NORMAL: Hand occupies appropriate central area (~20-50% of frame).
    """
    if gray_img.ndim == 3:
        if gray_img.shape[2] == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        elif gray_img.shape[2] == 1:
            gray_img = gray_img.squeeze(-1)

    h, w = gray_img.shape[:2]
    total_px = h * w

    mean_val = float(np.mean(gray_img))
    std_val = float(np.std(gray_img))

    # Fast Gaussian blur + Otsu threshold
    blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hand_px = int(np.count_nonzero(thresh))
    occupancy = hand_px / float(total_px)

    # Check 5px border margins
    margin = 5
    top_touch = bool(np.count_nonzero(thresh[:margin, :]) > 15)
    bottom_touch = bool(np.count_nonzero(thresh[-margin:, :]) > 15)
    left_touch = bool(np.count_nonzero(thresh[:, :margin]) > 15)
    right_touch = bool(np.count_nonzero(thresh[:, -margin:]) > 15)
    border_touches = sum([top_touch, bottom_touch, left_touch, right_touch])

    if occupancy < 0.08 or mean_val < 20.0:
        reason = "HAND_OUTSIDE_FRAME"
        instruction = "No hand detected. Place palm flat ~10-15cm above camera."
    elif std_val < 14.0:
        reason = "INSUFFICIENT_VISIBILITY"
        instruction = "Lighting or contrast too low. Ensure proper illumination and hold hand steady."
    elif occupancy < 0.20:
        reason = "HAND_TOO_FAR"
        instruction = "Hand is too far — move closer to the camera sensor."
    elif occupancy > 0.54 or (border_touches >= 3 and occupancy > 0.40):
        reason = "HAND_TOO_CLOSE"
        instruction = "Hand is too close — move hand farther from lens (~10-15cm)."
    else:
        reason = "NORMAL"
        instruction = "Hand positioning appears normal."

    return {
        "reason": reason,
        "instruction": instruction,
        "occupancy_pct": round(occupancy * 100.0, 2),
        "mean_intensity": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "border_touches": border_touches,
        "borders": {
            "top": top_touch, "bottom": bottom_touch,
            "left": left_touch, "right": right_touch
        }
    }


def detect_hand_landmarks_with_diagnostics(gray_img: np.ndarray, landmarker) -> dict:
    """
    Executes MediaPipe HandLandmarker with deterministic positioning failure diagnostics (Phase 2).
    Returns dict:
      - 'success': bool
      - 'landmarks': list of 21 (x, y) tuples if success, else None
      - 'reason': machine-readable failure reason (HAND_TOO_CLOSE, HAND_TOO_FAR, etc.)
      - 'instruction': human-readable positioning guidance for kiosk UI
      - 'diagnostics': pre-landmark heuristic stats (occupancy, borders, contrast)
    """
    if gray_img.ndim == 3 and gray_img.shape[2] == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    elif gray_img.ndim == 3 and gray_img.shape[2] == 1:
        gray = gray_img.squeeze(-1)
    else:
        gray = gray_img

    if gray.shape[0] < 200 or gray.shape[1] < 200:
        return {
            "success": False,
            "landmarks": None,
            "reason": "IMAGE_TOO_SMALL",
            "instruction": "Camera image too small (minimum 200x200 required).",
            "diagnostics": {}
        }

    diag = diagnose_hand_positioning(gray)

    if isinstance(landmarker, str):
        landmarker = build_landmarker(landmarker)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        # Heuristic failure classification
        if diag["reason"] in (CODE_HAND_TOO_CLOSE, CODE_HAND_TOO_FAR, CODE_HAND_OUTSIDE_FRAME):
            fail_reason = diag["reason"]
            fail_instruction = diag["instruction"]
        elif diag["reason"] == "INSUFFICIENT_VISIBILITY":
            fail_reason = CODE_QUALITY_LOW_CONTRAST
            fail_instruction = "Lighting or contrast too low. Ensure proper illumination and hold hand steady."
        else:
            # MediaPipe failed despite normal occupancy (e.g. boundary clip or orientation)
            if diag["borders"]["top"] or diag["borders"]["bottom"] or diag["occupancy_pct"] > 45.0:
                fail_reason = CODE_HAND_TOO_CLOSE
                fail_instruction = "Hand is too close or fingers cropped — move hand slightly farther (~10-15cm)."
            elif diag["borders"]["left"] or diag["borders"]["right"]:
                fail_reason = CODE_HAND_OUTSIDE_FRAME
                fail_instruction = "Hand off-center — center palm within the guide frame."
            else:
                fail_reason = CODE_MEDIAPIPE_NO_LANDMARKS
                fail_instruction = "Hand landmarks not detected. Hold palm flat with fingers slightly spread ~10-15cm above camera."

        return {
            "success": False,
            "landmarks": None,
            "reason": fail_reason,
            "instruction": fail_instruction,
            "diagnostics": diag
        }

    h, w = gray_img.shape[:2]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]
    if len(landmarks) < 21:
        return {
            "success": False,
            "landmarks": None,
            "reason": CODE_INVALID_LANDMARKS,
            "instruction": "Incomplete hand landmarks. Hold palm flat and keep fingers visible.",
            "diagnostics": diag
        }

    return {
        "success": True,
        "landmarks": landmarks,
        "reason": "OK",
        "instruction": "Hand landmarks detected successfully.",
        "diagnostics": diag
    }


def detect_hand_landmarks(gray_img: np.ndarray, landmarker) -> list:
    """
    Runs MediaPipe HandLandmarker and returns all 21 (x, y) coordinates.
    Accepts either an active HandLandmarker instance or a model_path string.
    Raises CaptureError with specific positioning instruction and error_code if landmarks cannot be detected.
    """
    res = detect_hand_landmarks_with_diagnostics(gray_img, landmarker)
    if not res["success"]:
        stage = "positioning" if res["reason"] in (CODE_HAND_TOO_CLOSE, CODE_HAND_TOO_FAR, CODE_HAND_OUTSIDE_FRAME) else "mediapipe"
        raise CaptureError(
            error_code=res["reason"],
            instruction=res["instruction"],
            stage=stage,
            diagnostics=res.get("diagnostics")
        )
    return res["landmarks"]


def compute_roi_quality(roi_224: np.ndarray, bbox: tuple, frame_shape: tuple) -> dict:
    """
    Evaluates extracted 224x224 palm ROI against strict biometric quality bounds (Phase 3).
    Returns dict:
      - 'is_valid': bool
      - 'error_code': Optional[str]
      - 'pad_pct': float (percentage of ROI area derived from border padding)
      - 'contrast_std': float (intensity standard deviation across vessels)
      - 'mean_intensity': float
      - 'reasons': list of failing quality conditions
    """
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    L = max(x2 - x1, y2 - y1, 1)

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w_frame)
    pad_bottom = max(0, y2 - h_frame)

    pad_area = (pad_left + pad_right) * L + (pad_top + pad_bottom) * L
    total_area = L * L
    pad_pct = min(1.0, float(pad_area) / float(total_area))

    contrast_std = float(np.std(roi_224))
    mean_val = float(np.mean(roi_224))

    reasons = []
    error_code = None
    if pad_pct > 0.25:
        reasons.append(f"Excessive boundary padding ({pad_pct*100:.1f}% > 25.0% max)")
        error_code = CODE_QUALITY_EXCESSIVE_PADDING
    elif contrast_std < 10.0:
        reasons.append(f"Low vessel contrast (std={contrast_std:.1f} < 10.0 min)")
        error_code = CODE_QUALITY_LOW_CONTRAST
    elif roi_224.shape != (224, 224):
        reasons.append(f"Invalid ROI dimensions ({roi_224.shape} != 224x224)")
        error_code = CODE_ROI_EXTRACTION_FAILED

    return {
        "is_valid": len(reasons) == 0,
        "valid": len(reasons) == 0,
        "error_code": error_code,
        "pad_pct": round(pad_pct, 4),
        "contrast_std": round(contrast_std, 2),
        "mean_intensity": round(mean_val, 2),
        "reasons": reasons
    }


def extract_valleys_from_landmarks(landmarks_px: list) -> tuple:
    """
    Derives Pv1 and Pv2 knuckle anchors with chirality normalization.
    Pv1 and Pv2 always form a vector oriented horizontally across the hand
    so that fingers point upward (towards -y) and the palm points downward.
    """
    index_mcp  = np.array(landmarks_px[5],  dtype=float)
    middle_mcp = np.array(landmarks_px[9],  dtype=float)
    ring_mcp   = np.array(landmarks_px[13], dtype=float)
    pinky_mcp  = np.array(landmarks_px[17], dtype=float)
    wrist      = np.array(landmarks_px[0],  dtype=float)

    pv_radial = (index_mcp + middle_mcp) / 2.0
    pv_ulnar  = (ring_mcp  + pinky_mcp)  / 2.0

    # Vector from wrist (L0) to middle MCP (L9) points upward along the hand
    v_up = middle_mcp - wrist
    # Perpendicular vector pointing across the hand to the anatomical right
    v_across = np.array([-v_up[1], v_up[0]])

    # Vector from radial to ulnar side
    d_vec = pv_ulnar - pv_radial
    if np.dot(d_vec, v_across) < 0:
        # Left hand presentation: orient pv1 -> pv2 along v_across
        pv1, pv2 = pv_ulnar, pv_radial
    else:
        pv1, pv2 = pv_radial, pv_ulnar

    return tuple(pv1.astype(int)), tuple(pv2.astype(int))


def segment_hand(gray_img: np.ndarray) -> np.ndarray:
    """Binary segmentation of the hand silhouette for distance transform analysis."""
    norm    = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(norm, (11, 11), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled  = np.zeros_like(clean)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

    return filled


def extract_ma2017_scaled_roi(gray_img: np.ndarray, pv1: tuple, pv2: tuple,
                               binary_mask: np.ndarray = None, target_size: int = 256,
                               scale_factor: float = 1.5,
                               offset_factor: float = 0.35,
                               landmarks_px: list = None) -> tuple:
    """
    Extracts canonical 256x256 pixel Region of Interest (ROI) based on Ma et al. (2017).
    Uses skeletal wrist landmark (L0) to reliably determine palm direction without
    relying on binary silhouette thresholding. Symmetrically pads boundary cuts to
    guarantee strict 1:1 aspect ratio pre-resize.
    """
    dx = pv2[0] - pv1[0]
    dy = pv2[1] - pv1[1]
    dist_pv   = np.hypot(dx, dy)
    angle_deg = np.degrees(np.arctan2(dy, dx))

    mid_x = (pv1[0] + pv2[0]) / 2.0
    mid_y = (pv1[1] + pv2[1]) / 2.0

    if gray_img.ndim == 3 and gray_img.shape[2] == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    elif gray_img.ndim == 3 and gray_img.shape[2] == 1:
        gray_img = gray_img.squeeze(-1)

    h, w = gray_img.shape[:2]
    M            = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
    rotated_gray = cv2.warpAffine(gray_img, M, (w, h), flags=cv2.INTER_LINEAR)

    pt_mid_h = np.array([mid_x, mid_y, 1.0])
    rot_mid  = M.dot(pt_mid_h)
    mx_r, my_r = int(rot_mid[0]), int(rot_mid[1])

    # Determine palm direction
    direction = 1
    if landmarks_px and len(landmarks_px) > 0:
        # Skeletal anchoring via Wrist (Landmark 0)
        wrist = landmarks_px[0]
        rot_wrist = M.dot(np.array([wrist[0], wrist[1], 1.0]))
        direction = 1 if rot_wrist[1] > my_r else -1
    elif binary_mask is not None:
        rotated_bin = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        dist_map = cv2.distanceTransform(rotated_bin, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        direction = 1 if max_loc[1] > my_r else -1

    L         = int(dist_pv * scale_factor)
    offset_d0 = int(dist_pv * offset_factor)

    x1 = int(mx_r - L / 2)
    x2 = int(mx_r + L / 2)

    if direction > 0:
        y1 = my_r + offset_d0
        y2 = y1 + L
    else:
        y2 = my_r - offset_d0
        y1 = y2 - L

    # Symmetric / edge-safe padding to preserve strict L x L square aspect ratio
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded_canvas = cv2.copyMakeBorder(
            rotated_gray, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        crop_y1 = y1 + pad_top
        crop_y2 = y2 + pad_top
        crop_x1 = x1 + pad_left
        crop_x2 = x2 + pad_left
        roi_patch = padded_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        roi_patch = rotated_gray[y1:y2, x1:x2]

    if roi_patch.size == 0 or roi_patch.shape[0] < 10 or roi_patch.shape[1] < 10:
        raise CaptureError(
            error_code=CODE_ROI_EXTRACTION_FAILED,
            instruction="Failed to extract palm ROI bounding box. Center palm and hold steady.",
            stage="roi",
            diagnostics={"bbox": (x1, y1, x2, y2), "patch_shape": roi_patch.shape if hasattr(roi_patch, 'shape') else None}
        )

    roi_normalized = cv2.resize(roi_patch, (target_size, target_size),
                                interpolation=cv2.INTER_CUBIC)

    return roi_normalized, (x1, y1, x2, y2), rotated_gray


def enhance_roi_vessels(roi_img: np.ndarray) -> np.ndarray:
    """Applies bilateral filter + CLAHE to enhance sub-dermal vein contrast."""
    stretched   = cv2.normalize(roi_img, None, 0, 255, cv2.NORM_MINMAX)
    smooth      = cv2.bilateralFilter(stretched, d=7, sigmaColor=35, sigmaSpace=35)
    clahe       = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    clahe_roi   = clahe.apply(smooth)
    return clahe_roi


def draw_landmarks_overlay(
    image: np.ndarray,
    landmarks_px: list,
    pv1: tuple = None,
    pv2: tuple = None,
    bbox: tuple = None
) -> np.ndarray:
    """
    Renders diagnostic visualization of 21 hand landmarks, skeleton lines,
    Pv1/Pv2 knuckle valley anchors, and ROI bounding box (Phase A4 Debug Mode).
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17)                                 # Palm base
    ]

    # Draw skeletal bones
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(landmarks_px) and p2_idx < len(landmarks_px):
            pt1 = tuple(int(c) for c in landmarks_px[p1_idx])
            pt2 = tuple(int(c) for c in landmarks_px[p2_idx])
            cv2.line(vis, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

    # Draw 21 landmark points
    for idx, pt in enumerate(landmarks_px):
        center = tuple(int(c) for c in pt)
        cv2.circle(vis, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, center, 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Pv1 and Pv2 knuckle anchors
    if pv1 is not None and pv2 is not None:
        p1 = tuple(int(c) for c in pv1)
        p2 = tuple(int(c) for c in pv2)
        cv2.circle(vis, p1, 7, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, p2, 7, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.line(vis, p1, p2, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, "Pv1", (p1[0]-15, p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(vis, "Pv2", (p2[0]+5, p2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return vis
