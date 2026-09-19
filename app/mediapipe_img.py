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
    MEDIAPIPE_AVAILABLE = False

try:
    from app.constants import MODEL_PATH
except ImportError:
    from constants import MODEL_PATH

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


def build_landmarker(model_path: str = MODEL_PATH) -> HandLandmarker:
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


def detect_hand_landmarks(gray_img: np.ndarray, landmarker) -> list:
    """
    Runs MediaPipe HandLandmarker and returns all 21 (x, y) coordinates.
    Accepts either an active HandLandmarker instance or a model_path string.
    """
    if gray_img.shape[0] < 200 or gray_img.shape[1] < 200:
        raise ValueError(
            f"Image too small for landmark detection: {gray_img.shape}. "
            f"Minimum 200x200 required."
        )

    if isinstance(landmarker, str):
        landmarker = build_landmarker(landmarker)

    rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        raise ValueError("No hand detected. Check palm placement and lighting.")

    h, w = gray_img.shape
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]


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

    h, w = gray_img.shape
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
        raise ValueError("Invalid ROI bounding box coordinates.")

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
