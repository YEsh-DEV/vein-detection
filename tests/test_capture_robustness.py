#!/usr/bin/env python3
"""
tests/test_capture_robustness.py
---------------------------------
Unit and integration tests for Demo-Hardening Capture Robustness:
  1. Structured CaptureError domain exception & to_dict() contract
  2. All 12 canonical error codes defined and prioritized correctly
  3. Pre-landmark positioning heuristic error classification
  4. Multi-frame burst capture logic (immediate success on first valid frame)
  5. Multi-frame burst capture fallback (aggregates and surfaces highest-priority diagnostic)
  6. Capture diagnostics JSONL logging format and telemetry fields
  7. API endpoint error contracts for /api/enroll/sample and /api/scan
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import app.constants as constants
from app.capture_errors import (
    CaptureError,
    ERROR_PRIORITY,
    CODE_HAND_TOO_CLOSE,
    CODE_HAND_TOO_FAR,
    CODE_HAND_OUTSIDE_FRAME,
    CODE_MEDIAPIPE_NO_LANDMARKS,
    CODE_INVALID_LANDMARKS,
    CODE_VALLEY_EXTRACTION_FAILED,
    CODE_ROI_EXTRACTION_FAILED,
    CODE_QUALITY_LOW_CONTRAST,
    CODE_QUALITY_EXCESSIVE_PADDING,
    CODE_MODEL_NOT_LOADED,
    CODE_CAMERA_ERROR,
    CODE_UNKNOWN_PIPELINE_ERROR,
)
from app.mediapipe_img import (
    diagnose_hand_positioning,
    detect_hand_landmarks_with_diagnostics,
    compute_roi_quality,
    extract_valleys_from_landmarks,
    segment_hand,
    extract_ma2017_scaled_roi,
)
import app.server as server
import app.db_manager as db_manager


class TestCaptureRobustness(unittest.TestCase):
    """Verifies structured capture error taxonomy and multi-frame burst capture."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_log = os.path.join(self.temp_dir.name, "test_capture_diagnostics.jsonl")
        self.orig_log = constants.CAPTURE_DIAGNOSTICS_LOG
        constants.CAPTURE_DIAGNOSTICS_LOG = self.test_log
        server.CAPTURE_DIAGNOSTICS_LOG = self.test_log

        # Isolate database for tests calling server endpoints
        self.orig_db_path = db_manager.DB_PATH
        self.test_db_path = os.path.join(self.temp_dir.name, "test_capture_db.db")
        db_manager.DB_PATH = self.test_db_path
        db_manager.init_db()
        server.engine = server.SearchEngine(engine_version="v2")

    def tearDown(self):
        db_manager.DB_PATH = self.orig_db_path
        constants.CAPTURE_DIAGNOSTICS_LOG = self.orig_log
        server.CAPTURE_DIAGNOSTICS_LOG = self.orig_log
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Error Taxonomy & Exception Contract
    # -----------------------------------------------------------------------
    def test_01_all_12_canonical_error_codes_defined(self):
        """Validates that all 12 canonical error codes are present and unique."""
        expected_codes = {
            "HAND_TOO_CLOSE",
            "HAND_TOO_FAR",
            "HAND_OUTSIDE_FRAME",
            "MEDIAPIPE_NO_LANDMARKS",
            "INVALID_LANDMARKS",
            "VALLEY_EXTRACTION_FAILED",
            "ROI_EXTRACTION_FAILED",
            "QUALITY_LOW_CONTRAST",
            "QUALITY_EXCESSIVE_PADDING",
            "MODEL_NOT_LOADED",
            "CAMERA_ERROR",
            "UNKNOWN_PIPELINE_ERROR",
        }
        actual_codes = {
            CODE_HAND_TOO_CLOSE,
            CODE_HAND_TOO_FAR,
            CODE_HAND_OUTSIDE_FRAME,
            CODE_MEDIAPIPE_NO_LANDMARKS,
            CODE_INVALID_LANDMARKS,
            CODE_VALLEY_EXTRACTION_FAILED,
            CODE_ROI_EXTRACTION_FAILED,
            CODE_QUALITY_LOW_CONTRAST,
            CODE_QUALITY_EXCESSIVE_PADDING,
            CODE_MODEL_NOT_LOADED,
            CODE_CAMERA_ERROR,
            CODE_UNKNOWN_PIPELINE_ERROR,
        }
        self.assertEqual(expected_codes, actual_codes)
        self.assertEqual(len(actual_codes), 12)

    def test_02_capture_error_is_value_error_subclass(self):
        """Verifies CaptureError inherits from ValueError for backward compatibility."""
        err = CaptureError(
            error_code=CODE_HAND_TOO_CLOSE,
            instruction="Hand is too close — move hand farther (~10-15cm) from lens.",
            stage="positioning",
            diagnostics={"occupancy_pct": 58.2, "border_touches": 3}
        )
        self.assertIsInstance(err, ValueError)
        self.assertEqual(str(err), "Hand is too close — move hand farther (~10-15cm) from lens.")

        # Test to_dict() contract
        d = err.to_dict()
        self.assertEqual(d["error_code"], CODE_HAND_TOO_CLOSE)
        self.assertEqual(d["reason"], CODE_HAND_TOO_CLOSE)
        self.assertEqual(d["stage"], "positioning")
        self.assertIn("farther", d["instruction"])
        self.assertEqual(d["detail"], d["instruction"])  # detail matches instruction for frontend
        self.assertEqual(d["diagnostics"]["occupancy_pct"], 58.2)

    def test_03_error_priority_ranking(self):
        """Verifies that Quality and Positioning errors have higher priority than generic landmark drops."""
        err_quality = CaptureError(CODE_QUALITY_EXCESSIVE_PADDING, "excessive padding")
        err_position = CaptureError(CODE_HAND_TOO_CLOSE, "hand too close")
        err_mediapipe = CaptureError(CODE_MEDIAPIPE_NO_LANDMARKS, "no landmarks")
        err_unknown = CaptureError(CODE_UNKNOWN_PIPELINE_ERROR, "unknown error")

        self.assertGreater(err_quality.priority, err_position.priority)
        self.assertGreater(err_position.priority, err_mediapipe.priority)
        self.assertGreater(err_mediapipe.priority, err_unknown.priority)

    # -----------------------------------------------------------------------
    # 2. Heuristic Positioning & Quality Gate Error Mapping
    # -----------------------------------------------------------------------
    def test_04_positioning_heuristic_too_close_detection(self):
        """Verifies that an image dominated by high-intensity palm triggers HAND_TOO_CLOSE."""
        # Frame of 480x640: palm filling ~65% of frame with border touches
        img = np.zeros((480, 640), dtype=np.uint8)
        img[0:450, 50:640] = 180  # bright palm occupying >60% and touching borders
        diag = diagnose_hand_positioning(img)
        self.assertEqual(diag["reason"], CODE_HAND_TOO_CLOSE)
        self.assertIn("farther", diag["instruction"].lower())

    def test_05_positioning_heuristic_empty_frame_detection(self):
        """Verifies that a dark/empty frame triggers HAND_OUTSIDE_FRAME."""
        img = np.full((480, 640), 5, dtype=np.uint8)
        diag = diagnose_hand_positioning(img)
        self.assertEqual(diag["reason"], CODE_HAND_OUTSIDE_FRAME)
        self.assertIn("no hand", diag["instruction"].lower())

    def test_06_compute_roi_quality_error_codes(self):
        """Verifies that compute_roi_quality returns explicit error codes on failures."""
        # Low contrast check
        flat_roi = np.full((224, 224), 100, dtype=np.uint8)
        q_low_contrast = compute_roi_quality(flat_roi, (50, 50, 200, 200), (480, 640))
        self.assertFalse(q_low_contrast["is_valid"])
        self.assertEqual(q_low_contrast["error_code"], CODE_QUALITY_LOW_CONTRAST)

        # High padding check
        noisy_roi = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        # Bounding box extending 100px beyond 480x640 frame
        q_padding = compute_roi_quality(noisy_roi, (-100, -100, 200, 200), (480, 640))
        self.assertFalse(q_padding["is_valid"])
        self.assertEqual(q_padding["error_code"], CODE_QUALITY_EXCESSIVE_PADDING)

    # -----------------------------------------------------------------------
    # 3. Multi-Frame Burst Capture Mechanism
    # -----------------------------------------------------------------------
    def test_07_burst_capture_succeeds_on_second_frame(self):
        """
        Simulates live capture where frame 1 experiences transient MediaPipe landmark drop,
        but frame 2 passes all quality gates. Burst capture must immediately succeed on frame 2.
        """
        # Mock frames
        frame_bad = np.zeros((480, 640), dtype=np.uint8)
        frame_good = np.ones((480, 640), dtype=np.uint8) * 128

        frames = [frame_bad, frame_good]
        frame_idx = 0

        def mock_capture():
            nonlocal frame_idx
            f = frames[min(frame_idx, len(frames) - 1)]
            frame_idx += 1
            return f

        dummy_roi = np.ones((224, 224), dtype=np.uint8) * 128
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)
        dummy_quality = {"is_valid": True, "pad_pct": 0.05, "contrast_std": 25.0}

        def mock_process(gray):
            if np.all(gray == 0):
                raise CaptureError(
                    error_code=CODE_MEDIAPIPE_NO_LANDMARKS,
                    instruction="Hand landmarks not detected. Hold palm flat ~10-15cm above camera.",
                    stage="mediapipe"
                )
            return dummy_roi, dummy_emb, dummy_quality

        with patch.object(server, "CAMERA_AVAILABLE", True), \
             patch.object(server, "capture_frame_gray", side_effect=mock_capture), \
             patch.object(server, "process_enrollment_sample", side_effect=mock_process):

            gray, roi, emb, quality, attempt = server.capture_burst_and_process_enrollment("test_user", max_frames=5, frame_interval_s=0.01)

            self.assertEqual(attempt, 2, "Burst capture should have succeeded on attempt #2")
            self.assertIsNotNone(roi)
            self.assertIsNotNone(emb)
            self.assertTrue(quality["is_valid"])

    def test_08_burst_capture_all_fail_surfaces_highest_priority_error(self):
        """
        When all 5 frames fail, burst capture must select the highest diagnostic priority error
        (e.g., HAND_TOO_CLOSE over generic MEDIAPIPE_NO_LANDMARKS) and log the failure.
        """
        errors = [
            CaptureError(CODE_MEDIAPIPE_NO_LANDMARKS, "No landmarks", stage="mediapipe"),
            CaptureError(CODE_HAND_TOO_CLOSE, "Hand is too close — move hand farther (~10-15cm) from lens.", stage="positioning"),
            CaptureError(CODE_MEDIAPIPE_NO_LANDMARKS, "No landmarks", stage="mediapipe"),
            CaptureError(CODE_MEDIAPIPE_NO_LANDMARKS, "No landmarks", stage="mediapipe"),
            CaptureError(CODE_MEDIAPIPE_NO_LANDMARKS, "No landmarks", stage="mediapipe"),
        ]
        err_idx = 0

        def mock_process(gray):
            nonlocal err_idx
            e = errors[min(err_idx, len(errors) - 1)]
            err_idx += 1
            raise e

        with patch.object(server, "CAMERA_AVAILABLE", True), \
             patch.object(server, "capture_frame_gray", return_value=np.zeros((480, 640), dtype=np.uint8)), \
             patch.object(server, "process_enrollment_sample", side_effect=mock_process):

            with self.assertRaises(CaptureError) as ctx:
                server.capture_burst_and_process_enrollment("test_user", max_frames=5, frame_interval_s=0.01)

            # Assert that the prioritized error was HAND_TOO_CLOSE
            self.assertEqual(ctx.exception.error_code, CODE_HAND_TOO_CLOSE)
            self.assertIn("farther", ctx.exception.instruction)

            # Assert diagnostic log file was written
            self.assertTrue(os.path.exists(self.test_log))
            with open(self.test_log, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertGreaterEqual(len(lines), 1)
                log_entry = json.loads(lines[-1])
                self.assertEqual(log_entry["error_code"], CODE_HAND_TOO_CLOSE)
                self.assertEqual(log_entry["attempts_total"], 5)
                self.assertEqual(len(log_entry["attempt_details"]), 5)

    # -----------------------------------------------------------------------
    # 4. Diagnostic Logging Format Verification (Task 4)
    # -----------------------------------------------------------------------
    def test_09_capture_diagnostic_log_schema(self):
        """Verifies that log_capture_failure_diagnostic writes all required fields."""
        diag_err = CaptureError(
            error_code=CODE_QUALITY_LOW_CONTRAST,
            instruction="Quality gate rejected sample: Low vessel contrast. Ensure proper illumination.",
            stage="quality_gate",
            diagnostics={
                "occupancy_pct": 34.5,
                "border_touches": 0,
                "contrast_std": 8.2,
                "pad_pct": 0.04,
                "landmarks_count": 21,
            }
        )
        server.log_capture_failure_diagnostic(diag_err, attempts=3, attempt_details=[diag_err.to_dict()])

        self.assertTrue(os.path.exists(self.test_log))
        with open(self.test_log, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        # Verify all Task 4 fields:
        self.assertIn("timestamp", entry)
        self.assertEqual(entry["stage"], "quality_gate")
        self.assertEqual(entry["error_code"], CODE_QUALITY_LOW_CONTRAST)
        self.assertEqual(entry["occupancy"], 34.5)
        self.assertEqual(entry["border_touches"], 0)
        self.assertEqual(entry["contrast"], 8.2)
        self.assertEqual(entry["padding"], 0.04)
        self.assertTrue(entry["mediapipe_landmarks"])
        self.assertEqual(entry["roi_status"], "rejected_quality")
        self.assertEqual(entry["attempts_total"], 3)

    # -----------------------------------------------------------------------
    # 5. API Error Response Contracts
    # -----------------------------------------------------------------------
    def test_10_enroll_sample_failure_returns_structured_error(self):
        """Verifies /api/enroll/sample returns HTTP 400 with structured JSON on capture failure."""
        try:
            from starlette.testclient import TestClient
        except (ImportError, RuntimeError):
            raise unittest.SkipTest("TestClient not available in this environment")

        sample_err = CaptureError(
            error_code=CODE_HAND_TOO_CLOSE,
            instruction="Hand is too close — move hand farther (~10-15cm) from lens.",
            stage="positioning",
            diagnostics={"occupancy_pct": 58.0, "border_touches": 3}
        )

        with patch.object(server, "MODEL_LOADED", True), \
             patch.object(server, "capture_burst_and_process_enrollment", side_effect=sample_err):

            client = TestClient(server.app, raise_server_exceptions=False)
            res = client.post("/api/enroll/sample", json={"username": "test_user"})

            self.assertEqual(res.status_code, 400)
            data = res.json()
            self.assertEqual(data["error_code"], CODE_HAND_TOO_CLOSE)
            self.assertEqual(data["reason"], CODE_HAND_TOO_CLOSE)
            self.assertEqual(data["stage"], "positioning")
            self.assertIn("farther", data["detail"])
            self.assertIsInstance(data["detail"], str)

    def test_11_scan_failure_returns_structured_error(self):
        """Verifies /api/scan returns HTTP 400 with structured JSON on capture failure."""
        try:
            from starlette.testclient import TestClient
        except (ImportError, RuntimeError):
            raise unittest.SkipTest("TestClient not available in this environment")

        scan_err = CaptureError(
            error_code=CODE_MEDIAPIPE_NO_LANDMARKS,
            instruction="Hand landmarks not detected. Hold palm flat ~10-15cm above camera.",
            stage="mediapipe"
        )

        with patch.object(server, "MODEL_LOADED", True), \
             patch.object(server, "CAMERA_AVAILABLE", True), \
             patch.object(server, "capture_frame_gray", return_value=np.zeros((480, 640), dtype=np.uint8)), \
             patch.object(server, "process_image_with_timing", side_effect=scan_err):

            client = TestClient(server.app, raise_server_exceptions=False)
            res = client.post("/api/scan")

            self.assertEqual(res.status_code, 400)
            data = res.json()
            self.assertEqual(data["error_code"], CODE_MEDIAPIPE_NO_LANDMARKS)
            self.assertEqual(data["stage"], "mediapipe")
            self.assertIsInstance(data["detail"], str)

    # -----------------------------------------------------------------------
    # 6. Pipeline Signature & Contract Verification
    # -----------------------------------------------------------------------
    def test_12_pipeline_signature_contracts(self):
        """
        Guarantees that pipeline functions enforce their exact production signatures:
        1. extract_valleys_from_landmarks(landmarks_px: list) takes exactly 1 positional arg.
           Passing 2 arguments must raise TypeError.
        2. extract_ma2017_scaled_roi returns a 3-tuple (roi_normalized, bbox, rotated_gray).
        3. compute_roi_quality takes exactly 3 positional args (roi_224, bbox, frame_shape).
        """
        # Synthetic 21-point palm skeleton
        synthetic_landmarks = [
            (320, 420),  # L0: Wrist
            (280, 360), (260, 320), (250, 280), (240, 250),  # L1-L4: Thumb
            (290, 290), (280, 240), (275, 200), (270, 160),  # L5-L8: Index (L5=MCP)
            (320, 285), (320, 230), (320, 190), (320, 150),  # L9-L12: Middle (L9=MCP)
            (350, 290), (355, 240), (360, 200), (365, 165),  # L13-L16: Ring (L13=MCP)
            (380, 305), (390, 260), (395, 225), (400, 190),  # L17-L20: Pinky (L17=MCP)
        ]

        # 1. extract_valleys_from_landmarks takes 1 positional argument
        pv1, pv2 = extract_valleys_from_landmarks(synthetic_landmarks)
        self.assertIsInstance(pv1, tuple)
        self.assertIsInstance(pv2, tuple)
        self.assertEqual(len(pv1), 2)
        self.assertEqual(len(pv2), 2)

        # Calling with 2 positional arguments MUST raise TypeError
        with self.assertRaises(TypeError):
            extract_valleys_from_landmarks(synthetic_landmarks, (480, 640))

        # 2. extract_ma2017_scaled_roi
        gray = np.ones((480, 640), dtype=np.uint8) * 128
        hand_mask = segment_hand(gray)
        roi_res = extract_ma2017_scaled_roi(
            gray, pv1, pv2, hand_mask,
            target_size=224, scale_factor=1.6, offset_factor=0.35,
            landmarks_px=synthetic_landmarks
        )
        self.assertIsInstance(roi_res, tuple)
        self.assertEqual(len(roi_res), 3, "extract_ma2017_scaled_roi must return a 3-tuple (roi, bbox, rotated)")
        roi_224, bbox, rot = roi_res
        self.assertEqual(roi_224.shape, (224, 224))
        self.assertEqual(len(bbox), 4)

        # 3. compute_roi_quality takes 3 positional arguments
        quality = compute_roi_quality(roi_224, bbox, gray.shape)
        self.assertIn("is_valid", quality)
        self.assertIn("pad_pct", quality)
        self.assertIn("contrast_std", quality)

        with self.assertRaises(TypeError):
            compute_roi_quality(roi_224)

    def test_13_server_pipeline_contract_consistency(self):
        """
        Verifies that app/server.py capture pipeline executes without any TypeError
        using the production contract for both scan and enroll.
        """
        synthetic_landmarks = [
            (320, 420),
            (280, 360), (260, 320), (250, 280), (240, 250),
            (290, 290), (280, 240), (275, 200), (270, 160),
            (320, 285), (320, 230), (320, 190), (320, 150),
            (350, 290), (355, 240), (360, 200), (365, 165),
            (380, 305), (390, 260), (395, 225), (400, 190),
        ]

        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)

        gray = np.random.randint(40, 200, (480, 640), dtype=np.uint8)

        # Test process_image_with_timing (used by /api/scan)
        with patch("app.server.detect_hand_landmarks", return_value=synthetic_landmarks), \
             patch("app.server.extract_embedding", return_value=dummy_emb):
            clahe_roi, emb, timing = server.process_image_with_timing(gray)
            self.assertEqual(clahe_roi.shape, (224, 224))
            self.assertEqual(len(emb), 512)
            self.assertIn("landmark_ms", timing)

        # Test process_enrollment_sample (used by /api/enroll/sample)
        mock_diag = {
            "success": True,
            "landmarks": synthetic_landmarks,
            "reason": "OK",
            "instruction": "OK",
            "diagnostics": {"occupancy_pct": 35.0, "border_touches": 0}
        }
        with patch("app.server.detect_hand_landmarks_with_diagnostics", return_value=mock_diag), \
             patch("app.server.extract_embedding", return_value=dummy_emb):
            clahe_roi, emb, quality = server.process_enrollment_sample(gray)
            self.assertEqual(clahe_roi.shape, (224, 224))
            self.assertEqual(len(emb), 512)
            self.assertIn("is_valid", quality)


if __name__ == "__main__":
    unittest.main(verbosity=2)
