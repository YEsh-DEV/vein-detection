#!/usr/bin/env python3
"""
tests/test_camera_pipeline.py
-----------------------------
Comprehensive automated unit tests for:
1. Optimal NIR channel extraction (Task 2 & 3).
2. Bounded exposure and analogue gain calibration (Task 4).
3. Best-frame burst selection (Task 5).
4. Display-only enhancement vs model input pipeline separation (Task 7).
5. Operator camera debug dump (Task 9).
6. Zero hallucination / no synthetic blending validation (Task 10).
"""

import os
import shutil
import tempfile
import unittest
import json
import cv2
import numpy as np

from app.constants import (
    DEFAULT_EXPOSURE_US,
    DEFAULT_ANALOGUE_GAIN,
    EXPOSURE_SEARCH_BOUNDS_US,
    GAIN_SEARCH_BOUNDS,
    TARGET_PALM_MEAN_MIN,
    TARGET_PALM_MEAN_MAX,
)
from app.camera_pipeline import (
    extract_nir_channel,
    create_display_frame,
    compute_frame_quality_score,
    select_best_frame,
    calculate_calibrated_exposure_and_gain,
    save_operator_debug_dump,
)


class TestCameraPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_camera_pipeline_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_optimal_nir_channel_extraction(self):
        """
        Validates that extract_nir_channel correctly extracts NIR-weighted luminance
        (0.50*R + 0.25*G + 0.25*B) from 3-channel and 4-channel arrays,
        giving higher weight to the NIR-transmissive Red channel.
        """
        # Test 1: Grayscale 2D array passes through unchanged
        gray_in = np.full((100, 100), 120, dtype=np.uint8)
        gray_out = extract_nir_channel(gray_in)
        np.testing.assert_array_equal(gray_in, gray_out)

        # Test 2: 3-channel BGR where Red is high (NIR absorption signal)
        # B=50, G=50, R=200
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 0] = 50   # Blue
        bgr[:, :, 1] = 50   # Green
        bgr[:, :, 2] = 200  # Red
        nir_out = extract_nir_channel(bgr)

        # Expected: 0.50*200 + 0.25*50 + 0.25*50 = 100 + 12.5 + 12.5 = 125
        self.assertEqual(nir_out.shape, (100, 100))
        self.assertAlmostEqual(float(nir_out.mean()), 125.0, delta=1.0)

        # Compare with OpenCV standard BGR2GRAY: 0.299*200 + 0.587*50 + 0.114*50 = 59.8 + 29.35 + 5.7 = 94.85
        std_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.assertGreater(float(nir_out.mean()), float(std_gray.mean()),
                           "NIR extraction must weight the Red channel higher than standard daylight Rec.601 BGR2GRAY")

        # Test 3: 4-channel XBGR8888 / BGRA
        xbgr = np.zeros((100, 100, 4), dtype=np.uint8)
        xbgr[:, :, :3] = bgr
        xbgr[:, :, 3] = 255
        nir_4ch = extract_nir_channel(xbgr)
        np.testing.assert_array_equal(nir_out, nir_4ch)

    def test_02_bounded_exposure_and_gain_calibration(self):
        """
        Validates bounded search behavior for underexposed and overexposed frames.
        Must strictly remain within safe hardware limits [8000, 22000] us and [1.0, 2.5] gain.
        """
        exp_min, exp_max = EXPOSURE_SEARCH_BOUNDS_US
        gain_min, gain_max = GAIN_SEARCH_BOUNDS

        # Scenario A: Real Pi test condition: mean=37.85 (underexposed), sat=0%, exp=5000, gain=1.0
        new_exp, new_gain = calculate_calibrated_exposure_and_gain(
            current_mean=37.85,
            current_sat_pct=0.0,
            current_exposure_us=5000,
            current_gain=1.0,
        )
        self.assertGreater(new_exp, 5000, "Exposure must increase when underexposed")
        self.assertGreaterEqual(new_exp, exp_min)
        self.assertLessEqual(new_exp, exp_max)
        self.assertGreaterEqual(new_gain, gain_min)
        self.assertLessEqual(new_gain, gain_max)

        # Scenario B: Saturated frame: mean=185, sat=8.5%
        new_exp, new_gain = calculate_calibrated_exposure_and_gain(
            current_mean=185.0,
            current_sat_pct=8.5,
            current_exposure_us=25000,
            current_gain=2.5,
        )
        self.assertLess(new_exp, 25000, "Exposure must decrease when saturated")
        self.assertGreaterEqual(new_exp, exp_min)
        self.assertLessEqual(new_gain, 2.5)

        # Scenario C: Optimal frame: mean=110, sat=0.5%
        new_exp, new_gain = calculate_calibrated_exposure_and_gain(
            current_mean=110.0,
            current_sat_pct=0.5,
            current_exposure_us=14000,
            current_gain=1.5,
        )
        self.assertEqual(new_exp, 14000, "Optimal exposure should remain stable")
        self.assertEqual(new_gain, 1.5, "Optimal gain should remain stable")

    def test_03_display_enhancement_removes_purple_and_separates_from_model(self):
        """
        Validates that create_display_frame transforms raw frames into a 3-channel
        uint8 BGR image with no channel blowout (no channel mean above 200),
        and that the display transform does NOT bleed into or mutate the raw frame.
        """
        # Create a purple frame: High Blue (210), Low Green (120), High Red (190)
        purple_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        purple_frame[:, :, 0] = 210  # B
        purple_frame[:, :, 1] = 120  # G
        purple_frame[:, :, 2] = 190  # R

        disp_frame = create_display_frame(purple_frame)

        # Output must be a 3-channel uint8 BGR image
        self.assertIsNotNone(disp_frame)
        self.assertEqual(disp_frame.shape, (200, 200, 3))
        self.assertEqual(disp_frame.dtype, np.uint8)

        # Check no channel blowout: no channel mean above 200
        for ch_idx in range(3):
            ch_mean = float(disp_frame[:, :, ch_idx].mean())
            self.assertLess(ch_mean, 200.0, f"Channel {ch_idx} mean ({ch_mean}) must not blow out above 200")

        # Crucial separation: Verify original raw frame was NOT mutated in-place
        self.assertEqual(purple_frame[0, 0, 0], 210)
        self.assertEqual(purple_frame[0, 0, 1], 120)
        self.assertEqual(purple_frame[0, 0, 2], 190)

        # Balanced mid-gray palm: Ensure output is NOT blown out above 200
        palm_patch = np.full((100, 100, 3), 160, dtype=np.uint8)
        disp_palm = create_display_frame(palm_patch)
        self.assertEqual(disp_palm.dtype, np.uint8)
        self.assertLess(float(disp_palm.mean()), 200.0, "Display frame must not blow palm out above 200")

    def test_04_best_frame_selection_prefers_sharp_contrasted_frame(self):
        """
        Validates that select_best_frame chooses the candidate with superior contrast
        and sharpness without any averaging or synthetic blending.
        """
        # Candidate 1: Blurry / low contrast frame
        frame1 = np.full((100, 100), 100, dtype=np.uint8)
        cv2.circle(frame1, (50, 50), 30, 110, -1)
        roi1 = frame1[20:80, 20:80]
        score1 = compute_frame_quality_score(frame1, roi_224=roi1, pad_pct=0.10)

        # Candidate 2: Crisp / high contrast frame
        frame2 = np.full((100, 100), 60, dtype=np.uint8)
        cv2.circle(frame2, (50, 50), 30, 140, -1)
        # Add sharp high-frequency edges
        for y in range(30, 70, 4):
            cv2.line(frame2, (30, y), (70, y), 80, 1)
        roi2 = frame2[20:80, 20:80]
        score2 = compute_frame_quality_score(frame2, roi_224=roi2, pad_pct=0.02)

        candidates = [
            {
                "gray": frame1,
                "clahe_roi": roi1,
                "embedding": np.ones(512),
                "quality": {"is_valid": True, "pad_pct": 0.10},
                "score_dict": score1,
                "attempt_idx": 1,
            },
            {
                "gray": frame2,
                "clahe_roi": roi2,
                "embedding": np.ones(512) * 2.0,
                "quality": {"is_valid": True, "pad_pct": 0.02},
                "score_dict": score2,
                "attempt_idx": 2,
            }
        ]

        self.assertGreater(score2["score"], score1["score"],
                           "Candidate 2 with higher contrast and sharpness must score higher")

        best = select_best_frame(candidates)
        self.assertEqual(best["attempt_idx"], 2, "Candidate 2 must be selected as best frame")
        # Ensure returned frame is real unblended frame2
        np.testing.assert_array_equal(best["gray"], frame2)

    def test_05_operator_debug_dump_saves_all_required_artifacts(self):
        """
        Validates Problem 7: saving 01_raw.png, 02_nir.png, 03_landmarks.png,
        04_roi_raw.png, 05_roi_enhanced.png, and diagnostics.json with all required fields:
          exposure, gain, resolution, selected channel/representation, brightness,
          contrast, saturation, sharpness, landmark count, Pv1/Pv2, ROI bbox,
          ROI padding, ROI contrast, ROI sharpness.
        """
        raw = np.full((100, 100, 3), 100, dtype=np.uint8)
        gray = np.full((100, 100), 100, dtype=np.uint8)
        overlay = np.full((100, 100, 3), 120, dtype=np.uint8)
        roi_raw = np.full((50, 50), 90, dtype=np.uint8)
        roi_enh = np.full((50, 50), 110, dtype=np.uint8)

        diag_data = {
            "resolution": "640x480",
            "exposure_us": 18000,
            "analogue_gain": 1.8,
            "mean": 112.5,
            "contrast_std": 24.3,
            "dynamic_range": [25, 230],
            "saturation_pct": 0.4,
            "sharpness": 55.2,
            "landmark_count": 21,
            "pv1_pv2": [[100, 200], [150, 210]],
            "roi_bbox": [100, 120, 224, 224],
            "roi_padding_pct": 0.05,
            "roi_contrast_std": 28.1,
            "roi_sharpness": 45.0,
            "selected_frame_score": 64.5,
        }

        saved = save_operator_debug_dump(
            raw_frame=raw,
            processed_gray=gray,
            landmarks_overlay=overlay,
            roi_raw=roi_raw,
            roi_enhanced=roi_enh,
            diagnostics=diag_data,
            output_dir=self.temp_dir,
        )

        # Check that both 02_nir.png and 02_processed_gray.png exist
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "01_raw.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "02_nir.png")), "02_nir.png must exist")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "02_processed_gray.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "03_landmarks.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "04_roi_raw.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "05_roi_enhanced.png")))

        # Check diagnostics JSON
        json_path = os.path.join(self.temp_dir, "diagnostics.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        # Check exact Problem 7 fields
        for required_key in [
            "exposure", "gain", "resolution", "selected channel/representation",
            "brightness", "contrast", "saturation", "sharpness", "landmark count",
            "Pv1/Pv2", "ROI bbox", "ROI padding", "ROI contrast", "ROI sharpness"
        ]:
            self.assertIn(required_key, loaded_json, f"Missing required Problem 7 diagnostic field: {required_key}")

    def test_06_configurable_nir_channels_and_empirical_comparison(self):
        """
        Validates Problem 3: empirical comparison of NIR representations (R, G, B, Rec.601,
        NIR weighted, Equal weighted) inside the palm region.
        """
        from app.camera_pipeline import compare_nir_representations

        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :, 0] = 50   # Blue
        test_frame[:, :, 1] = 80   # Green
        test_frame[:, :, 2] = 160  # Red

        # Test configurable extraction modes
        r_out = extract_nir_channel(test_frame, method="r_channel")
        self.assertEqual(int(r_out.mean()), 160)
        g_out = extract_nir_channel(test_frame, method="g_channel")
        self.assertEqual(int(g_out.mean()), 80)
        b_out = extract_nir_channel(test_frame, method="b_channel")
        self.assertEqual(int(b_out.mean()), 50)
        eq_out = extract_nir_channel(test_frame, method="equal_nir")
        self.assertEqual(int(eq_out.mean()), (160 + 80 + 50) // 3)

        # Test empirical comparison execution
        comp = compare_nir_representations(test_frame, output_dir=self.temp_dir)
        self.assertIn("channel_R", comp)
        self.assertIn("channel_G", comp)
        self.assertIn("channel_B", comp)
        self.assertIn("grayscale", comp)
        self.assertIn("nir_weighted", comp)
        self.assertIn("equal_weighted", comp)

        # Verify all Problem 3 required statistics exist for each channel
        for ch_name, stats in comp.items():
            for stat_key in ["mean", "std", "p1", "p5", "p50", "p95", "p99", "dynamic_range", "local_contrast", "sharpness", "sobel_variance"]:
                self.assertIn(stat_key, stats, f"Stat '{stat_key}' missing from '{ch_name}'")

        # Verify output files
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "channel_R.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "channel_G.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "channel_B.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "grayscale.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "nir_weighted.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "comparison.json")))

    def test_07_candidate_exposure_sweep(self):
        """
        Validates Problem 4: bounded candidate exposure and analogue gain sweep.
        """
        import argparse
        from tools.sweep_camera_exposure import run_exposure_sweep

        args = argparse.Namespace(synthetic=True, image="", output_dir=self.temp_dir)
        exit_code = run_exposure_sweep(args)
        self.assertEqual(exit_code, 0)

        # Verify sweep output JSON
        sweep_json_path = os.path.join(self.temp_dir, "exposure_sweep_results.json")
        self.assertTrue(os.path.isfile(sweep_json_path))
        with open(sweep_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("candidates_tested", data)
        self.assertEqual(len(data["candidates_tested"]), 7)
        for cand in data["candidates_tested"]:
            self.assertIn("exposure_us", cand)
            self.assertIn("analogue_gain", cand)
            self.assertIn("palm_contrast_std", cand)
            self.assertIn("saturation_pct", cand)


if __name__ == "__main__":
    unittest.main()
