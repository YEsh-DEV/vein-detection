#!/usr/bin/env python3
"""
mediapipe_img.py
-----------------
MediaPipe Hand Landmark Detection & Canonical ROI Extraction.
Replaces contour convexity defects with 21 anatomical joint landmarks.
"""

import os
import sys
import cv2
import numpy as np
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

DEFAULT_MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH) -> str:
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


def build_landmarker(model_path: str = DEFAULT_MODEL_PATH) -> HandLandmarker:
    """Creates and returns a persistent HandLandmarker instance."""
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
    Derives Pv1 (index-middle) and Pv2 (ring-little) from MCP knuckle landmarks.
    Pv1 = midpoint(Landmark 5, Landmark 9)
    Pv2 = midpoint(Landmark 13, Landmark 17)
    """
    index_mcp  = np.array(landmarks_px[5],  dtype=float)
    middle_mcp = np.array(landmarks_px[9],  dtype=float)
    ring_mcp   = np.array(landmarks_px[13], dtype=float)
    pinky_mcp  = np.array(landmarks_px[17], dtype=float)

    pv1 = tuple(((index_mcp + middle_mcp) / 2.0).astype(int))
    pv2 = tuple(((ring_mcp  + pinky_mcp)  / 2.0).astype(int))
    return pv1, pv2


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
                               binary_mask: np.ndarray, target_size: int = 256,
                               scale_factor: float = 1.5,
                               offset_factor: float = 0.35) -> tuple:
    """
    Extracts canonical 256x256 pixel Region of Interest (ROI) based on Ma et al. (2017).
    """
    dx = pv2[0] - pv1[0]
    dy = pv2[1] - pv1[1]
    dist_pv   = np.hypot(dx, dy)
    angle_deg = np.degrees(np.arctan2(dy, dx))

    mid_x = (pv1[0] + pv2[0]) / 2.0
    mid_y = (pv1[1] + pv2[1]) / 2.0

    h, w = gray_img.shape
    M            = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
    rotated_gray = cv2.warpAffine(gray_img,    M, (w, h), flags=cv2.INTER_LINEAR)
    rotated_bin  = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST)

    pt_mid_h = np.array([mid_x, mid_y, 1.0])
    rot_mid  = M.dot(pt_mid_h)
    mx_r, my_r = int(rot_mid[0]), int(rot_mid[1])

    dist_map      = cv2.distanceTransform(rotated_bin, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_map)
    direction     = 1 if max_loc[1] > my_r else -1

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

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

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
