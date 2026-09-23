#!/usr/bin/env python3
r"""
tools/sweep_camera_exposure.py — Empirical Exposure & Analogue Gain Sweep Tool
=============================================================================
Addresses Problem 4: Treats 18,000 µs / Gain 1.8 as a CANDIDATE setting,
not ground truth. Performs a bounded sweep across candidate settings:
  (10000, 1.2), (14000, 1.5), (18000, 1.8), (22000, 2.0), (26000, 2.2)

For each candidate setting:
  - Mean intensity
  - Contrast standard deviation
  - IR Saturation percentage (>=250)
  - Laplacian sharpness
  - Palm-region contrast (std dev restricted to segmented palm body)
  - Saves: debug_frames/sweep_exp_{exp}_gain_{gain}.png

Evaluates settings to maximize useful palm vascular contrast while avoiding:
  - White clipping / IR burnout
  - Severe motion blur
  - Excessive high-gain sensor/thermal noise
  - Washed-out or underexposed palm

Usage:
  python3 tools/sweep_camera_exposure.py                  # On Raspberry Pi (live Picamera2)
  python3 tools/sweep_camera_exposure.py --synthetic      # Laptop simulation mode
  python3 tools/sweep_camera_exposure.py --image path.png # Image-based simulation
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from app.constants import (
    DEBUG_FRAMES_DIR,
    TARGET_PALM_MEAN_MIN,
    TARGET_PALM_MEAN_MAX,
    MAX_IR_SATURATION_PCT,
)
from app.camera_pipeline import (
    extract_nir_channel,
    compute_palm_region_mask,
    compute_channel_metrics,
)

SWEEP_PRESETS = [
    (1000, 1.0),
    (2000, 1.0),
    (3000, 1.0),
    (4000, 1.0),
    (5000, 1.0),
    (6000, 1.0),
    (8000, 1.0),
    (10000, 1.0),
    (12000, 1.0),
]


def capture_candidate_picamera2(picam2, exposure_us: int, gain: float) -> np.ndarray:
    """Captures a frame using specific exposure and gain settings on Picamera2."""
    picam2.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ColourGains": (1.0, 1.0),
        "ExposureTime": exposure_us,
        "AnalogueGain": gain,
    })
    # Settle
    for _ in range(2):
        picam2.capture_array("main")
    frame_rgb = picam2.capture_array("main")
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def simulate_candidate_frame(base_frame: np.ndarray, exp_us: int, gain: float) -> np.ndarray:
    """
    Simulates optical response under scaled exposure and gain for laptop testing.
    Includes sensor scaling, non-linear saturation, and Poisson-like noise.
    """
    scale = (exp_us / 6000.0) * (gain / 1.0)
    sim = base_frame.astype(np.float32) * scale
    # Add mild noise proportional to gain
    noise = np.random.normal(0, gain * 1.5, sim.shape)
    sim_noisy = np.clip(sim + noise, 0, 255).astype(np.uint8)
    return sim_noisy


def run_exposure_sweep(args):
    target_dir = args.output_dir or DEBUG_FRAMES_DIR
    os.makedirs(target_dir, exist_ok=True)

    picam2 = None
    is_real_pi = False

    if not args.synthetic and not args.image:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            cfg = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            picam2.configure(cfg)
            picam2.start()
            is_real_pi = True
            print("[+] Picamera2 successfully initialized for hardware sweep.")
        except Exception as e:
            print(f"[!] Real Picamera2 not available ({e}). Running in synthetic simulation mode.")
            picam2 = None

    base_frame = None
    if args.image:
        base_frame = cv2.imread(args.image)
        if base_frame is None:
            print(f"[!] Failed to load image: {args.image}")
            return 1
    elif not is_real_pi:
        # Create base synthetic frame
        base_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        cv2.ellipse(base_frame, (320, 260), (95, 120), 0, 0, 360, (140, 115, 160), -1)
        for fx in [245, 290, 345, 395]:
            cv2.ellipse(base_frame, (fx, 130), (18, 55), 0, 0, 360, (130, 105, 150), -1)
        for pts in [
            [(320, 210), (310, 260), (330, 310)],
            [(280, 220), (290, 270), (280, 320)],
            [(350, 220), (340, 280), (360, 330)],
        ]:
            cv2.polylines(base_frame, [np.array(pts, dtype=np.int32)], False, (80, 65, 95), 3)

    sweep_results = []
    print("\n" + "=" * 95)
    print(f"  {'Setting (Exp µs, Gain)':<24} | {'Mean':<6} | {'Std':<6} | {'Palm Cont':<9} | {'Sat %':<6} | {'Sharpness':<9} | {'Verdict'}")
    print("-" * 95)

    best_setting = None
    best_score = -999.0

    try:
        for exp_us, gain in SWEEP_PRESETS:
            if is_real_pi and picam2 is not None:
                frame_bgr = capture_candidate_picamera2(picam2, exp_us, gain)
            else:
                frame_bgr = simulate_candidate_frame(base_frame, exp_us, gain)

            nir_gray = extract_nir_channel(frame_bgr)
            mask = compute_palm_region_mask(nir_gray)
            metrics = compute_channel_metrics(nir_gray, mask=mask)

            mean_val = metrics["mean"]
            std_val = metrics["std"]
            palm_cont = metrics["std"]
            sharpness = metrics["sharpness"]

            sat_px = int(np.count_nonzero(nir_gray >= 250))
            sat_pct = round((sat_px / float(nir_gray.size)) * 100.0, 2)

            # Quality verdict & scoring:
            # Rewards high palm contrast, penalizes saturation > 2% and extreme mean deviations
            mean_penalty = 0.0
            if mean_val < TARGET_PALM_MEAN_MIN:
                mean_penalty = (TARGET_PALM_MEAN_MIN - mean_val) * 0.7
            elif mean_val > TARGET_PALM_MEAN_MAX:
                mean_penalty = (mean_val - TARGET_PALM_MEAN_MAX) * 0.7

            sat_penalty = 0.0
            if sat_pct > MAX_IR_SATURATION_PCT:
                sat_penalty = (sat_pct - MAX_IR_SATURATION_PCT) * 20.0

            score = palm_cont * 1.5 + min(sharpness, 100.0) * 0.3 - mean_penalty - sat_penalty

            verdict = "PASS"
            if sat_pct > 3.0:
                verdict = "WARN_SATURATED"
            elif mean_val < 50.0:
                verdict = "WARN_UNDEREXPOSED"
            elif mean_val > 160.0:
                verdict = "WARN_OVEREXPOSED"

            item = {
                "exposure_us": exp_us,
                "analogue_gain": gain,
                "mean": mean_val,
                "contrast_std": std_val,
                "palm_contrast_std": palm_cont,
                "saturation_pct": sat_pct,
                "sharpness": sharpness,
                "score": round(score, 2),
                "verdict": verdict,
                "is_candidate": bool(exp_us == 6000 and gain == 1.0),
            }
            sweep_results.append(item)

            setting_label = f"{exp_us} µs / {gain:.1f}x"
            if item["is_candidate"]:
                setting_label += " [*]"
            print(f"  {setting_label:<24} | {mean_val:<6.1f} | {std_val:<6.1f} | {palm_cont:<9.1f} | {sat_pct:<6.1f} | {sharpness:<9.1f} | {verdict}")

            # Save diagnostic frame for this candidate setting
            fname = f"sweep_exp_{exp_us}_gain_{gain:.1f}.png"
            cv2.imwrite(os.path.join(target_dir, fname), frame_bgr)

            if score > best_score and verdict == "PASS":
                best_score = score
                best_setting = item

    finally:
        if picam2 is not None:
            picam2.stop()

    print("=" * 95)
    print("  [*] Current candidate setting: 6,000 µs @ Gain 1.0\n")

    summary = {
        "hardware_source": "REAL_PICAMERA2" if is_real_pi else ("IMAGE_SIMULATION" if args.image else "SYNTHETIC_SIMULATION"),
        "candidates_tested": sweep_results,
        "best_measured_setting": best_setting,
        "note": "Physical validation on Raspberry Pi must verify these results under actual IR LED illumination."
    }

    out_json = os.path.join(target_dir, "exposure_sweep_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Saved exposure sweep results to: {out_json}")
    if best_setting:
        print(f"[+] Recommended candidate from this run: {best_setting['exposure_us']} µs @ Gain {best_setting['analogue_gain']} (Score: {best_setting['score']})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep safe camera exposure and gain combinations")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic simulation mode")
    parser.add_argument("--image", type=str, default="", help="Path to base image to simulate sweeps on")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for frames and JSON")
    args = parser.parse_args()

    sys.exit(run_exposure_sweep(args))
