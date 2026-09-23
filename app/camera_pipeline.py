#!/usr/bin/env python3
r"""
camera_pipeline.py
------------------
Modular camera tuning, optimal NIR channel extraction, bounded exposure calibration,
display-only enhancement, and best-frame burst selection for Raspberry Pi NoIR.

Guarantees:
1. Strict separation of DISPLAY_FRAME (enhanced monochrome for human UI) from
   MODEL_INPUT_FRAME (reproducible calibrated grayscale for AMPVNet CNN).
2. Elimination of OV5647 NoIR pink/purple color cast via optimal NIR extraction.
3. Best-frame burst selection picking a real, unblended frame with optimal contrast and sharpness.
4. Zero artificial feature generation, zero frame averaging, zero synthetic hallucination.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

try:
    from app.constants import (
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        EXPOSURE_SEARCH_BOUNDS_US, GAIN_SEARCH_BOUNDS,
        EXPOSURE_SEARCH_STEPS_US, GAIN_SEARCH_STEPS,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        MIN_CONTRAST_STD, MAX_IR_SATURATION_PCT,
    )
except ImportError:
    from constants import (
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        EXPOSURE_SEARCH_BOUNDS_US, GAIN_SEARCH_BOUNDS,
        EXPOSURE_SEARCH_STEPS_US, GAIN_SEARCH_STEPS,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        MIN_CONTRAST_STD, MAX_IR_SATURATION_PCT,
    )


# ---------------------------------------------------------------------------
# 1. Optimal NIR Channel Extraction
# ---------------------------------------------------------------------------
def extract_nir_channel(frame: np.ndarray) -> np.ndarray:
    """
    Extracts calibrated NIR grayscale representation from a camera frame.

    Physical Rationale (OV5647 NoIR with 850nm NIR illumination):
    - Silicon has peak NIR quantum efficiency around 800-850nm.
    - The Red Bayer microfilter has ~85-90% transmittance at 850nm.
    - Blue has ~75% transmittance.
    - Green has lower transmittance (~55-65%).
    - Standard OpenCV COLOR_BGR2GRAY applies Rec.601 weights: 0.299*R + 0.587*G + 0.114*B,
      which gives 58.7% weight to the Green channel (the lowest NIR transmission site).
    - Calibrated NIR weighting (0.50*R + 0.25*G + 0.25*B) maximizes 850nm signal-to-noise ratio
      while integrating all physical photodiode sites without Bayer mosaic patterning.
    """
    if frame is None:
        raise ValueError("Input frame is None")

    if frame.ndim == 2:
        return frame

    # 4-channel image (e.g. XBGR8888 or BGRA from Picamera2/V4L2)
    if frame.shape[2] == 4:
        # Channels 0, 1, 2 are B, G, R
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
    elif frame.shape[2] == 3:
        # Standard BGR
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
    elif frame.shape[2] == 1:
        return frame.squeeze(-1)
    else:
        raise ValueError(f"Unsupported frame channels: {frame.shape[2]}")

    # Calibrated NIR luminance: emphasizes the high-transmission Red channel
    nir_gray = (0.50 * r + 0.25 * g + 0.25 * b).clip(0, 255).astype(np.uint8)
    return nir_gray


# ---------------------------------------------------------------------------
# 2. Display-Only Enhancement Pipeline (Human-Readable UI)
# ---------------------------------------------------------------------------
def create_display_frame(raw_frame: np.ndarray) -> np.ndarray:
    """
    Transforms raw camera frame into a clean, contrast-stretched monochrome visualization
    for live browser preview (/api/video_feed).

    CRITICAL BOUNDARY:
    This function is strictly for OPERATOR DISPLAY ONLY.
    Its output is NEVER fed into AMPVNet or feature extraction.

    Transformations applied:
    1. Optimal NIR monochrome conversion (removes purple/pink cast completely).
    2. Dynamic range normalization (stretches contrast to visible range [0, 255]).
    3. Mild CLAHE (clipLimit=2.0, tileGridSize=(8, 8)) to reveal palm crease details.
    4. 3-channel BGR encoding for JPEG streaming.
    """
    if raw_frame is None:
        return None

    # Step 1: NIR monochrome conversion
    nir_gray = extract_nir_channel(raw_frame)

    # Step 2: Mild contrast stretch / dynamic range normalization
    # Protect against flat all-black or all-white frames
    min_val, max_val = int(nir_gray.min()), int(nir_gray.max())
    if max_val > min_val + 10:
        stretched = cv2.normalize(nir_gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        stretched = nir_gray

    # Step 3: Mild CLAHE for human operator visualization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    disp_enhanced = clahe.apply(stretched)

    # Step 4: Convert to 3-channel BGR for browser JPEG stream compatibility
    return cv2.cvtColor(disp_enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 3. Frame Quality Scoring
# ---------------------------------------------------------------------------
def compute_frame_quality_score(
    gray: np.ndarray,
    landmarks_px: Optional[List[Tuple[int, int]]] = None,
    roi_224: Optional[np.ndarray] = None,
    pad_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes an objective, deterministic quality score for a captured frame candidate.

    Scoring criteria:
    - Target mean intensity: 75 - 145 (sweet spot of 8-bit dynamic range).
    - Contrast: Higher standard deviation indicates rich vascular/tissue transitions.
    - Sharpness: Laplacian variance measures focus and lack of motion blur.
    - Saturation: Harsh penalty if IR LEDs clip pixels (>250).
    - Padding: Penalizes boundary clipping.
    """
    if gray is None:
        return {"score": -999.0, "valid": False, "reason": "Frame is None"}

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # IR Saturation check (percentage of clipped white pixels)
    sat_px = int(np.count_nonzero(gray >= 250))
    sat_pct = (sat_px / float(gray.size)) * 100.0

    # Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # ROI-specific metrics if ROI was successfully extracted
    roi_std = std_val
    roi_sharpness = sharpness
    if roi_224 is not None and roi_224.size > 0:
        roi_std = float(np.std(roi_224))
        roi_lap = cv2.Laplacian(roi_224, cv2.CV_64F)
        roi_sharpness = float(roi_lap.var())

    # Underexposure / Overexposure penalty
    mean_penalty = 0.0
    if mean_val < TARGET_PALM_MEAN_MIN:
        mean_penalty += (TARGET_PALM_MEAN_MIN - mean_val) * 0.8
    elif mean_val > TARGET_PALM_MEAN_MAX:
        mean_penalty += (mean_val - TARGET_PALM_MEAN_MAX) * 0.8

    # IR Saturation penalty
    sat_penalty = 0.0
    if sat_pct > MAX_IR_SATURATION_PCT:
        sat_penalty += (sat_pct - MAX_IR_SATURATION_PCT) * 25.0

    # Padding penalty
    pad_penalty = float(pad_pct) * 15.0

    # Composite Score formulation:
    # Emphasizes ROI contrast and sharpness while strictly penalizing blur, blowout, and underexposure
    composite_score = (
        roi_std * 1.5
        + min(roi_sharpness, 100.0) * 0.4
        - mean_penalty
        - sat_penalty
        - pad_penalty
    )

    return {
        "score": round(composite_score, 2),
        "mean": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "roi_contrast_std": round(roi_std, 2),
        "sharpness": round(sharpness, 2),
        "roi_sharpness": round(roi_sharpness, 2),
        "sat_pct": round(sat_pct, 2),
        "pad_pct": round(pad_pct, 4),
        "valid": bool(sat_pct <= 5.0 and std_val >= 8.0 and mean_val >= 25.0),
    }


# ---------------------------------------------------------------------------
# 4. Best-Frame Selection (Burst Evaluator)
# ---------------------------------------------------------------------------
def select_best_frame(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects the single best frame from a burst of real candidate captures.

    Each candidate dict must contain:
    - 'gray': np.ndarray (real captured grayscale frame)
    - 'clahe_roi': np.ndarray (224x224 vessel-enhanced ROI)
    - 'embedding': np.ndarray (512-D L2-normalized embedding)
    - 'quality': dict (ROI quality gate result)
    - 'score_dict': dict (from compute_frame_quality_score)
    - 'attempt_idx': int

    PROHIBITED:
    - No frame averaging.
    - No blending or synthetic hallucination.
    - No multi-embedding vector averaging.
    The returned item is a pure, real single frame.
    """
    if not candidates:
        raise ValueError("Candidates list is empty in select_best_frame.")

    # Sort descending by composite quality score
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c["score_dict"]["score"],
        reverse=True
    )
    return sorted_candidates[0]


# ---------------------------------------------------------------------------
# 5. Bounded Exposure & Gain Calibration (Picamera2 / Hardware)
# ---------------------------------------------------------------------------
def calculate_calibrated_exposure_and_gain(
    current_mean: float,
    current_sat_pct: float,
    current_exposure_us: int = DEFAULT_EXPOSURE_US,
    current_gain: float = DEFAULT_ANALOGUE_GAIN
) -> Tuple[int, float]:
    """
    Computes bounded, safe exposure and analogue gain adjustments based on real frame statistics.

    Rules:
    - Stays strictly within [8000, 30000] us to prevent motion blur and frame rate drop (<30 fps).
    - Stays strictly within [1.0, 3.0] gain to prevent high thermal/sensor noise.
    - Adjusts incrementally towards the target mean band [75, 145].
    """
    exp_min, exp_max = EXPOSURE_SEARCH_BOUNDS_US
    gain_min, gain_max = GAIN_SEARCH_BOUNDS

    new_exp = current_exposure_us
    new_gain = current_gain

    # Case 1: IR blowout / saturation detected -> reduce exposure/gain
    if current_sat_pct > MAX_IR_SATURATION_PCT or current_mean > TARGET_PALM_MEAN_MAX:
        # Step down exposure first
        new_exp = max(exp_min, int(current_exposure_us * 0.75))
        if new_exp == exp_min:
            new_gain = max(gain_min, round(current_gain * 0.8, 1))
        return new_exp, new_gain

    # Case 2: Underexposure -> increase exposure/gain
    if current_mean < TARGET_PALM_MEAN_MIN:
        ratio = TARGET_PALM_MEAN_MIN / max(current_mean, 10.0)
        # Cap step size to 1.6x per calibration step
        step_factor = min(ratio, 1.6)
        target_exp = int(current_exposure_us * step_factor)

        if target_exp <= exp_max:
            new_exp = target_exp
        else:
            new_exp = exp_max
            # If exposure is already maxed, increase analogue gain
            new_gain = min(gain_max, round(current_gain * (target_exp / float(exp_max)), 1))

        return new_exp, new_gain

    # Case 3: Already within optimal band
    return current_exposure_us, current_gain


# ---------------------------------------------------------------------------
# 6. Operator Camera Debug Output (Task 9)
# ---------------------------------------------------------------------------
def save_operator_debug_dump(
    raw_frame: np.ndarray,
    processed_gray: np.ndarray,
    landmarks_overlay: Optional[np.ndarray],
    roi_raw: Optional[np.ndarray],
    roi_enhanced: Optional[np.ndarray],
    diagnostics: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Saves internal operator debug frames and structured diagnostics JSON (Task 9).

    Produces:
      01_raw.png
      02_processed_gray.png
      03_landmarks.png
      04_roi_raw.png
      05_roi_enhanced.png
      diagnostics.json
    """
    import os
    import json
    try:
        from app.constants import DEBUG_FRAMES_DIR
    except ImportError:
        from constants import DEBUG_FRAMES_DIR

    target_dir = output_dir or DEBUG_FRAMES_DIR
    os.makedirs(target_dir, exist_ok=True)

    saved_paths = {}

    # 01_raw.png
    if raw_frame is not None:
        p01 = os.path.join(target_dir, "01_raw.png")
        cv2.imwrite(p01, raw_frame)
        saved_paths["01_raw"] = p01

    # 02_processed_gray.png
    if processed_gray is not None:
        p02 = os.path.join(target_dir, "02_processed_gray.png")
        cv2.imwrite(p02, processed_gray)
        saved_paths["02_processed_gray"] = p02

    # 03_landmarks.png
    if landmarks_overlay is not None:
        p03 = os.path.join(target_dir, "03_landmarks.png")
        cv2.imwrite(p03, landmarks_overlay)
        saved_paths["03_landmarks"] = p03

    # 04_roi_raw.png
    if roi_raw is not None:
        p04 = os.path.join(target_dir, "04_roi_raw.png")
        cv2.imwrite(p04, roi_raw)
        saved_paths["04_roi_raw"] = p04

    # 05_roi_enhanced.png
    if roi_enhanced is not None:
        p05 = os.path.join(target_dir, "05_roi_enhanced.png")
        cv2.imwrite(p05, roi_enhanced)
        saved_paths["05_roi_enhanced"] = p05

    # diagnostics.json & diagnostics.txt
    p_json = os.path.join(target_dir, "diagnostics.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    saved_paths["diagnostics_json"] = p_json

    p_txt = os.path.join(target_dir, "diagnostics.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write("=== OPERATOR CAMERA CAPTURE DIAGNOSTICS ===\n")
        for k, v in diagnostics.items():
            f.write(f"{k:<24}: {v}\n")
    saved_paths["diagnostics_txt"] = p_txt

    return saved_paths
