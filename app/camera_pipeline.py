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
        CANDIDATE_EXPOSURE_SWEEPS,
        NIR_EXTRACTION_METHOD,
        DISPLAY_PERCENTILE_LOW, DISPLAY_PERCENTILE_HIGH,
        DISPLAY_CLAHE_CLIP, DISPLAY_CLAHE_GRID,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        MIN_CONTRAST_STD, MAX_IR_SATURATION_PCT,
    )
except ImportError:
    from constants import (
        DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN,
        EXPOSURE_SEARCH_BOUNDS_US, GAIN_SEARCH_BOUNDS,
        EXPOSURE_SEARCH_STEPS_US, GAIN_SEARCH_STEPS,
        CANDIDATE_EXPOSURE_SWEEPS,
        NIR_EXTRACTION_METHOD,
        DISPLAY_PERCENTILE_LOW, DISPLAY_PERCENTILE_HIGH,
        DISPLAY_CLAHE_CLIP, DISPLAY_CLAHE_GRID,
        TARGET_PALM_MEAN_MIN, TARGET_PALM_MEAN_MAX,
        MIN_CONTRAST_STD, MAX_IR_SATURATION_PCT,
    )


# ---------------------------------------------------------------------------
# 1. Configurable NIR Channel Extraction
# ---------------------------------------------------------------------------
def extract_nir_channel(frame: np.ndarray, method: Optional[str] = None) -> np.ndarray:
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

    Supported methods:
      - 'weighted_nir': 0.50*R + 0.25*G + 0.25*B (default)
      - 'r_channel': Pure Red channel
      - 'g_channel': Pure Green channel
      - 'b_channel': Pure Blue channel
      - 'rec601_gray': Standard Rec.601 (0.299*R + 0.587*G + 0.114*B)
      - 'equal_nir': Equal weighting ((R + G + B) / 3.0)
    """
    if frame is None:
        raise ValueError("Input frame is None")

    if frame.ndim == 2:
        return frame

    mode = (method or NIR_EXTRACTION_METHOD).lower()

    # 4-channel image (e.g. XBGR8888 or BGRA from Picamera2/V4L2)
    if frame.shape[2] == 4:
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

    if mode == "r_channel":
        return r.clip(0, 255).astype(np.uint8)
    elif mode == "g_channel":
        return g.clip(0, 255).astype(np.uint8)
    elif mode == "b_channel":
        return b.clip(0, 255).astype(np.uint8)
    elif mode in ("rec601_gray", "grayscale"):
        return (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
    elif mode in ("equal_nir", "equal_weighted"):
        return ((r + g + b) / 3.0).clip(0, 255).astype(np.uint8)
    else:
        # Default: weighted_nir (0.50*R + 0.25*G + 0.25*B)
        return (0.50 * r + 0.25 * g + 0.25 * b).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Display-Only Enhancement Pipeline (Human-Readable UI)
# ---------------------------------------------------------------------------
def auto_calibrate_exposure(picam2, target_mean=115.0,
                             min_exp=500, max_exp=8000,
                             min_gain=1.0, max_gain=1.8,
                             max_iterations=12, tolerance=8.0):
    """
    Binary-search auto-calibration for overexposed LED setups.
    Finds the lowest exposure where palm mean lands in target range.
    Runs once at startup before the main server loop.
    Returns (best_exposure_us, best_gain).
    """
    import time
    low_exp  = min_exp
    high_exp = max_exp
    best_exp  = min_exp
    best_gain = min_gain

    for iteration in range(max_iterations):
        mid_exp = int((low_exp + high_exp) / 2)
        picam2.set_controls({
            "AeEnable":      False,
            "AwbEnable":     False,
            "ColourGains":   (1.0, 1.0),
            "ExposureTime":  mid_exp,
            "AnalogueGain":  min_gain,
        })
        time.sleep(0.18)   # allow sensor to settle

        try:
            frame = picam2.capture_array()
        except TypeError:
            frame = picam2.capture_array("main")
        if frame is None:
            break
        # Convert to BGR if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = frame[:, :, ::-1].copy()
        else:
            bgr = frame

        nir = (0.50 * bgr[:, :, 2].astype(np.float32)
             + 0.25 * bgr[:, :, 1].astype(np.float32)
             + 0.25 * bgr[:, :, 0].astype(np.float32))
        current_mean = float(np.mean(nir))

        print(f"[AutoCal] iter={iteration+1:2d}  exp={mid_exp:5d}µs  "
              f"gain={min_gain:.1f}  mean={current_mean:.1f}")

        if abs(current_mean - target_mean) <= tolerance:
            best_exp  = mid_exp
            best_gain = min_gain
            break
        elif current_mean > target_mean:
            high_exp = mid_exp       # too bright → reduce
        else:
            low_exp  = mid_exp       # too dark  → increase

        best_exp  = mid_exp
        best_gain = min_gain

    print(f"[AutoCal] Final: {best_exp}µs @ gain {best_gain:.1f}  "
          f"(target mean={target_mean})")
    return best_exp, best_gain


def create_display_frame(raw_bgr: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """
    Display pipeline for overexposed 850nm LED NIR setup.
    Pure percentile normalization + CLAHE. No homomorphic division.
    """
    if raw_bgr is None:
        return None

    if raw_bgr.ndim == 2:
        nir = raw_bgr.astype(np.float32)
    else:
        b = raw_bgr[:, :, 0].astype(np.float32)
        g = raw_bgr[:, :, 1].astype(np.float32)
        r = raw_bgr[:, :, 2].astype(np.float32)
        nir = 0.50 * r + 0.25 * g + 0.25 * b

    lo = np.percentile(nir, DISPLAY_PERCENTILE_LOW)
    hi = np.percentile(nir, DISPLAY_PERCENTILE_HIGH)
    if hi - lo < 15:
        hi = lo + 15
    nir_norm = np.clip(
        (nir - lo) / (hi - lo) * 255.0, 0, 255
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=DISPLAY_CLAHE_CLIP,
        tileGridSize=DISPLAY_CLAHE_GRID
    )
    enhanced = clahe.apply(nir_norm)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    h, w = display.shape[:2]
    bx1, by1 = int(w * 0.15), int(h * 0.05)
    bx2, by2 = int(w * 0.85), int(h * 0.95)
    corner_len = min(22, max(5, int(w * 0.1)))
    col, thick = (0, 220, 0), 2
    for cx, cy in [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]:
        dx = corner_len if cx == bx1 else -corner_len
        dy = corner_len if cy == by1 else -corner_len
        cv2.line(display, (cx, cy), (cx + dx, cy), col, thick)
        cv2.line(display, (cx, cy), (cx, cy + dy), col, thick)
    ty = (by1 - 8) if by1 >= 12 else (by1 + 15)
    cv2.putText(
        display,
        "Place palm here | 10-14cm",
        (bx1, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
    )
    return display


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
# 6. Empirical NIR Representation Comparison (Problem 3)
# ---------------------------------------------------------------------------
def compute_palm_region_mask(gray: np.ndarray) -> np.ndarray:
    """
    Computes a binary mask isolating the palm/hand region from background
    using Otsu thresholding with morphological closing.
    Guarantees a valid, non-empty mask.
    """
    if gray is None:
        raise ValueError("Input gray image is None")
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological closing to seal internal vascular pits and valleys
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if cv2.countNonZero(closed) < 100:
        return np.full_like(gray, 255)
    return closed


def compute_channel_metrics(
    img_gray: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes empirical statistical and structural quality metrics on an image,
    strictly restricted to the palm region when a mask is provided.

    Metrics calculated (Problem 3):
    - mean
    - standard deviation (contrast std)
    - P1, P5, P50, P95, P99
    - dynamic range (P99 - P1)
    - local contrast (mean of local block std deviations in 16x16 windows)
    - Laplacian sharpness (Laplacian variance)
    - Sobel edge variance (Sobel gradient magnitude variance)
    """
    if img_gray is None:
        raise ValueError("Input image is None")

    if mask is not None and cv2.countNonZero(mask) > 50:
        palm_pixels = img_gray[mask > 0].astype(np.float32)
    else:
        palm_pixels = img_gray.astype(np.float32).ravel()

    mean_val = float(np.mean(palm_pixels))
    std_val = float(np.std(palm_pixels))
    p1, p5, p50, p95, p99 = [float(x) for x in np.percentile(palm_pixels, [1, 5, 50, 95, 99])]
    dyn_range = float(p99 - p1)

    # Sharpness via Laplacian variance
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    if mask is not None and cv2.countNonZero(mask) > 50:
        sharpness = float(np.var(lap[mask > 0]))
    else:
        sharpness = float(lap.var())

    # Sobel edge variance
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    if mask is not None and cv2.countNonZero(mask) > 50:
        sobel_var = float(np.var(sobel_mag[mask > 0]))
    else:
        sobel_var = float(np.var(sobel_mag))

    # Local contrast: average standard deviation across 16x16 sliding windows
    # inside the palm region (captures subtle sub-dermal vascular contrast)
    h, w = img_gray.shape
    block_stds = []
    bs = 16
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            if mask is not None:
                m_blk = mask[y:y+bs, x:x+bs]
                if cv2.countNonZero(m_blk) < (bs * bs * 0.70):
                    continue
            blk = img_gray[y:y+bs, x:x+bs]
            block_stds.append(float(np.std(blk)))

    local_contrast = float(np.mean(block_stds)) if block_stds else std_val

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "p1": round(p1, 2),
        "p5": round(p5, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
        "dynamic_range": round(dyn_range, 2),
        "local_contrast": round(local_contrast, 2),
        "sharpness": round(sharpness, 2),
        "sobel_variance": round(sobel_var, 2),
    }


def compare_nir_representations(
    frame: np.ndarray,
    palm_mask: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Empirically generates and compares candidate NIR representations:
    1. channel_R (Red channel alone)
    2. channel_G (Green channel alone)
    3. channel_B (Blue channel alone)
    4. grayscale (standard Rec.601)
    5. nir_weighted (0.50R + 0.25G + 0.25B)
    6. equal_weighted ((R + G + B) / 3.0)

    Computes detailed statistics INSIDE THE PALM REGION, and optionally saves:
      debug_frames/channel_R.png
      debug_frames/channel_G.png
      debug_frames/channel_B.png
      debug_frames/grayscale.png
      debug_frames/nir_weighted.png
      debug_frames/comparison.json
    """
    import os
    import json
    try:
        from app.constants import DEBUG_FRAMES_DIR
    except ImportError:
        from constants import DEBUG_FRAMES_DIR

    if frame is None:
        raise ValueError("Input frame is None")

    # Generate candidate grayscale representations
    representations = {
        "channel_R": extract_nir_channel(frame, method="r_channel"),
        "channel_G": extract_nir_channel(frame, method="g_channel"),
        "channel_B": extract_nir_channel(frame, method="b_channel"),
        "grayscale": extract_nir_channel(frame, method="rec601_gray"),
        "nir_weighted": extract_nir_channel(frame, method="weighted_nir"),
        "equal_weighted": extract_nir_channel(frame, method="equal_nir"),
    }

    # Obtain palm mask if not provided
    base_gray = representations["nir_weighted"]
    mask = palm_mask if palm_mask is not None else compute_palm_region_mask(base_gray)

    results: Dict[str, Any] = {}
    for name, img in representations.items():
        results[name] = compute_channel_metrics(img, mask=mask)

    target_dir = output_dir or DEBUG_FRAMES_DIR
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        # Save images
        cv2.imwrite(os.path.join(target_dir, "channel_R.png"), representations["channel_R"])
        cv2.imwrite(os.path.join(target_dir, "channel_G.png"), representations["channel_G"])
        cv2.imwrite(os.path.join(target_dir, "channel_B.png"), representations["channel_B"])
        cv2.imwrite(os.path.join(target_dir, "grayscale.png"), representations["grayscale"])
        cv2.imwrite(os.path.join(target_dir, "nir_weighted.png"), representations["nir_weighted"])
        cv2.imwrite(os.path.join(target_dir, "equal_weighted.png"), representations["equal_weighted"])

        comp_json_path = os.path.join(target_dir, "comparison.json")
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# 7. Operator Camera Debug Output (Problem 7)
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
    Saves internal operator debug frames and structured diagnostics JSON (Problem 7).

    Produces:
      01_raw.png
      02_nir.png (and 02_processed_gray.png for backward-compatibility)
      03_landmarks.png
      04_roi_raw.png
      05_roi_enhanced.png
      diagnostics.json
      diagnostics.txt

    Guarantees exact diagnostics.json keys:
      exposure, gain, resolution, selected channel/representation,
      brightness, contrast, saturation, sharpness, landmark count,
      Pv1/Pv2, ROI bbox, ROI padding, ROI contrast, ROI sharpness.
    """
    import os
    import json
    try:
        from app.constants import DEBUG_FRAMES_DIR, NIR_EXTRACTION_METHOD, DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN
    except ImportError:
        from constants import DEBUG_FRAMES_DIR, NIR_EXTRACTION_METHOD, DEFAULT_EXPOSURE_US, DEFAULT_ANALOGUE_GAIN

    target_dir = output_dir or DEBUG_FRAMES_DIR
    os.makedirs(target_dir, exist_ok=True)

    saved_paths = {}

    # 01_raw.png
    if raw_frame is not None:
        p01 = os.path.join(target_dir, "01_raw.png")
        cv2.imwrite(p01, raw_frame)
        saved_paths["01_raw"] = p01

    # 02_nir.png (primary) & 02_processed_gray.png (alias)
    if processed_gray is not None:
        p02 = os.path.join(target_dir, "02_nir.png")
        p02_alias = os.path.join(target_dir, "02_processed_gray.png")
        cv2.imwrite(p02, processed_gray)
        cv2.imwrite(p02_alias, processed_gray)
        saved_paths["02_nir"] = p02
        saved_paths["02_processed_gray"] = p02_alias

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

    # Normalize diagnostics dictionary with exact Problem 7 fields
    diag_norm: Dict[str, Any] = dict(diagnostics)

    # Resolution
    res = diag_norm.get("resolution")
    if not res:
        if raw_frame is not None and hasattr(raw_frame, "shape"):
            res = f"{raw_frame.shape[1]}x{raw_frame.shape[0]}"
        elif processed_gray is not None and hasattr(processed_gray, "shape"):
            res = f"{processed_gray.shape[1]}x{processed_gray.shape[0]}"
        else:
            res = "640x480"
    diag_norm["resolution"] = res

    # Exposure & Gain
    exp = diag_norm.get("exposure", diag_norm.get("exposure_us", DEFAULT_EXPOSURE_US))
    gn = diag_norm.get("gain", diag_norm.get("analogue_gain", DEFAULT_ANALOGUE_GAIN))
    diag_norm["exposure"] = exp
    diag_norm["exposure_us"] = exp
    diag_norm["gain"] = gn
    diag_norm["analogue_gain"] = gn

    # Selected representation
    rep = diag_norm.get("selected channel/representation", diag_norm.get("selected_representation", NIR_EXTRACTION_METHOD))
    diag_norm["selected channel/representation"] = rep
    diag_norm["selected_representation"] = rep

    # Brightness, Contrast, Saturation, Sharpness
    b_val = diag_norm.get("brightness", diag_norm.get("mean", float(np.mean(processed_gray)) if processed_gray is not None else 0.0))
    c_val = diag_norm.get("contrast", diag_norm.get("contrast_std", float(np.std(processed_gray)) if processed_gray is not None else 0.0))
    s_val = diag_norm.get("saturation", diag_norm.get("saturation_pct", 0.0))
    sh_val = diag_norm.get("sharpness", float(cv2.Laplacian(processed_gray, cv2.CV_64F).var()) if processed_gray is not None else 0.0)

    diag_norm["brightness"] = round(float(b_val), 2)
    diag_norm["mean"] = round(float(b_val), 2)
    diag_norm["contrast"] = round(float(c_val), 2)
    diag_norm["contrast_std"] = round(float(c_val), 2)
    diag_norm["saturation"] = round(float(s_val), 2)
    diag_norm["saturation_pct"] = round(float(s_val), 2)
    diag_norm["sharpness"] = round(float(sh_val), 2)

    # Landmark count and Pv1/Pv2
    lm_count = diag_norm.get("landmark count", diag_norm.get("landmark_count", 21))
    diag_norm["landmark count"] = lm_count
    diag_norm["landmark_count"] = lm_count

    pv = diag_norm.get("Pv1/Pv2", diag_norm.get("pv1_pv2", None))
    diag_norm["Pv1/Pv2"] = pv
    diag_norm["pv1_pv2"] = pv

    # ROI metrics
    bbox = diag_norm.get("ROI bbox", diag_norm.get("roi_bbox", None))
    pad = diag_norm.get("ROI padding", diag_norm.get("roi_padding", diag_norm.get("roi_padding_pct", 0.0)))
    r_cont = diag_norm.get("ROI contrast", diag_norm.get("roi_contrast", diag_norm.get("roi_contrast_std", 0.0)))
    r_sharp = diag_norm.get("ROI sharpness", diag_norm.get("roi_sharpness", float(cv2.Laplacian(roi_raw, cv2.CV_64F).var()) if roi_raw is not None else 0.0))

    diag_norm["ROI bbox"] = bbox
    diag_norm["roi_bbox"] = bbox
    diag_norm["ROI padding"] = pad
    diag_norm["roi_padding"] = pad
    diag_norm["roi_padding_pct"] = pad
    diag_norm["ROI contrast"] = r_cont
    diag_norm["roi_contrast"] = r_cont
    diag_norm["roi_contrast_std"] = r_cont
    diag_norm["ROI sharpness"] = round(float(r_sharp), 2)
    diag_norm["roi_sharpness"] = round(float(r_sharp), 2)

    # Write diagnostics.json
    p_json = os.path.join(target_dir, "diagnostics.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(diag_norm, f, indent=2)
    saved_paths["diagnostics_json"] = p_json

    # Write human-readable diagnostics.txt
    p_txt = os.path.join(target_dir, "diagnostics.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write("=== OPERATOR CAMERA CAPTURE DIAGNOSTICS ===\n")
        for k, v in diag_norm.items():
            f.write(f"{k:<32}: {v}\n")
    saved_paths["diagnostics_txt"] = p_txt

    return saved_paths
