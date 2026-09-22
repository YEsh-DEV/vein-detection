#!/usr/bin/env python3
"""
tests/test_stage3_integration.py
---------------------------------
Comprehensive offline end-to-end integration test suite for Stage 3.
Verifies the complete AMPVNet ONNX embedding, SQLite storage, and RAM cosine
matching pipeline against all 10 required failure modes and test cases:

  1. Same-person genuine matches
  2. Different-person impostor matches
  3. 1-template enrollment
  4. 3-template enrollment
  5. Multiple users
  6. Empty database
  7. Duplicate enrollment
  8. Low-quality / degenerate ROI
  9. Missing model file
 10. Corrupt embedding / invalid dimensions

Zero crashes allowed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

# Project root path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import app.constants as constants
import app.db_manager as db_manager
from app.ampvnet_inference import AMPVNetInference, extract_embedding, cosine_similarity
from app.search_engine import SearchEngine


class TestStage3Integration(unittest.TestCase):
    """Offline End-to-End integration test suite for AMPVNet v2 biometric pipeline."""

    @classmethod
    def setUpClass(cls):
        # Locate real test dataset ROIs (held-out subject-disjoint test split)
        cls.test_split_dir = _PROJECT_ROOT / "training/data_processed/own_splits/test"
        cls.subjects = {}
        for subj_dir in sorted(cls.test_split_dir.glob("*")):
            if subj_dir.is_dir():
                rois = sorted(list(subj_dir.glob("*.png")))
                if len(rois) >= 2:
                    cls.subjects[subj_dir.name] = rois

        cls.engine_onnx = AMPVNetInference()
        assert cls.engine_onnx.model_loaded, f"AMPVNet model failed to load: {cls.engine_onnx.error_detail}"

    def setUp(self):
        # Create an isolated temporary SQLite database for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, "test_palm_vein.db")

        # Patch db_manager DB_PATH
        self._orig_db_path = db_manager.DB_PATH
        db_manager.DB_PATH = self.test_db_path
        db_manager.init_db()

        # Initialize fresh search engine connected to test DB
        self.search_engine = SearchEngine(engine_version="v2")

    def tearDown(self):
        db_manager.DB_PATH = self._orig_db_path
        self.temp_dir.cleanup()

    def test_01_same_person_genuine_matches(self):
        """Test 1: Genuine match of the same unseen person across separate sessions."""
        # Use subject '119' or '039' with high genuine consistency
        subj_name = "119" if "119" in self.subjects else list(self.subjects.keys())[0]
        rois = self.subjects[subj_name]
        enroll_img = cv2.imread(str(rois[0]), cv2.IMREAD_GRAYSCALE)
        probe_img = cv2.imread(str(rois[1]), cv2.IMREAD_GRAYSCALE)

        emb_enroll = self.engine_onnx.extract_embedding(enroll_img)
        emb_probe = self.engine_onnx.extract_embedding(probe_img)

        # Enroll user
        username = f"user_{subj_name}"
        db_manager.enroll_user(username, [emb_enroll])
        self.search_engine.refresh_cache()

        # Probe search
        diag = self.search_engine.identify_with_diagnostics(emb_probe)

        self.assertTrue(diag["accepted"], f"Genuine probe should be accepted. Score: {diag['score']}")
        self.assertEqual(diag["username"], username)
        self.assertGreaterEqual(diag["score"], constants.MATCH_THRESHOLD)
        self.assertGreaterEqual(len(diag["ranked_candidates"]), 1)
        self.assertEqual(diag["ranked_candidates"][0]["username"], username)

    def test_02_different_person_impostor_matches(self):
        """Test 2: Impostor match between different persons must score below or reject."""
        subj_a, subj_b = "055", "039"
        if subj_a not in self.subjects or subj_b not in self.subjects:
            subj_names = list(self.subjects.keys())[:2]
            subj_a, subj_b = subj_names[0], subj_names[1]

        img_a = cv2.imread(str(self.subjects[subj_a][0]), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(self.subjects[subj_b][0]), cv2.IMREAD_GRAYSCALE)

        emb_a = self.engine_onnx.extract_embedding(img_a)
        emb_b = self.engine_onnx.extract_embedding(img_b)

        # Enroll Subject A
        db_manager.enroll_user("user_a", [emb_a])
        self.search_engine.refresh_cache()

        # Probe with Subject B (impostor)
        diag = self.search_engine.identify_with_diagnostics(emb_b)

        sim_ab = cosine_similarity(emb_a, emb_b)
        self.assertAlmostEqual(diag["score"], round(sim_ab, 4), places=3)
        # Impostor should score below MATCH_THRESHOLD and be rejected
        self.assertFalse(diag["accepted"])
        self.assertIsNone(diag["username"])
        print(f"\n[Test 2] Impostor similarity A-vs-B: {sim_ab:.4f} (Threshold: {constants.MATCH_THRESHOLD})")

    def test_03_one_template_enrollment(self):
        """Test 3: 1-template enrollment stores single template and allows correct probe recognition."""
        subj_name = "119" if "119" in self.subjects else list(self.subjects.keys())[0]
        rois = self.subjects[subj_name]
        img_enroll = cv2.imread(str(rois[0]), cv2.IMREAD_GRAYSCALE)
        img_probe = cv2.imread(str(rois[1]), cv2.IMREAD_GRAYSCALE)

        emb_enroll = self.engine_onnx.extract_embedding(img_enroll)
        emb_probe = self.engine_onnx.extract_embedding(img_probe)

        uid = db_manager.enroll_user("single_template_user", [emb_enroll])
        self.search_engine.refresh_cache()

        all_embs = db_manager.get_all_embeddings()
        self.assertEqual(len(all_embs["template_ids"]), 1)
        self.assertEqual(all_embs["matrix"].shape, (1, 512))

        diag = self.search_engine.identify_with_diagnostics(emb_probe)
        self.assertTrue(diag["accepted"])
        self.assertEqual(diag["username"], "single_template_user")

    def test_04_three_template_enrollment_and_aggregation(self):
        """Test 4: 3-template enrollment verifies mean aggregation T = normalize(sum(e_i))."""
        subj_name = "119" if "119" in self.subjects else list(self.subjects.keys())[0]
        rois = self.subjects[subj_name][:3]
        embs = [self.engine_onnx.extract_embedding(cv2.imread(str(r), cv2.IMREAD_GRAYSCALE)) for r in rois]

        # Calculate manual expected normalized mean
        sum_vec = np.sum(embs, axis=0)
        expected_mean = sum_vec / np.linalg.norm(sum_vec)

        # Verify db_manager.aggregate_embeddings matches
        agg = db_manager.aggregate_embeddings(embs)
        np.testing.assert_allclose(agg, expected_mean, atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(agg)), 1.0, places=5)

        # Enroll user with 3 samples
        uid = db_manager.enroll_user("multi_template_user", embs)
        self.search_engine.refresh_cache()

        # Should store 3 constituent templates
        all_embs = db_manager.get_all_embeddings()
        self.assertEqual(len(all_embs["template_ids"]), 3)

        # Identify using constituent sample
        diag = self.search_engine.identify_with_diagnostics(embs[0])
        self.assertTrue(diag["accepted"])
        self.assertEqual(diag["username"], "multi_template_user")

    def test_05_multiple_users_ranking(self):
        """Test 5: Multiple enrolled users ranking correctly with highest similarity first."""
        target_subjs = [s for s in ["119", "039", "055"] if s in self.subjects]
        enrolled_users = []
        for subj in target_subjs:
            rois = self.subjects[subj]
            uname = f"enrolled_{subj}"
            img = cv2.imread(str(rois[0]), cv2.IMREAD_GRAYSCALE)
            emb = self.engine_onnx.extract_embedding(img)
            db_manager.enroll_user(uname, [emb])
            enrolled_users.append((uname, subj, rois))

        self.search_engine.refresh_cache()

        # Test probe for user '119' (idx 0)
        target_uname, target_subj, target_rois = enrolled_users[0]
        probe_img = cv2.imread(str(target_rois[1]), cv2.IMREAD_GRAYSCALE)
        probe_emb = self.engine_onnx.extract_embedding(probe_img)

        diag = self.search_engine.identify_with_diagnostics(probe_emb)
        self.assertEqual(diag["username"], target_uname)
        self.assertEqual(len(diag["ranked_candidates"]), len(enrolled_users))

        # Check ranking order is strictly descending
        scores = [c["score"] for c in diag["ranked_candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_06_empty_database_handling(self):
        """Test 6: Empty database probe returns rejection without crash or exception."""
        dummy_probe = np.random.randn(512).astype(np.float32)
        dummy_probe /= np.linalg.norm(dummy_probe)

        # Ensure DB is empty
        all_embs = db_manager.get_all_embeddings()
        self.assertEqual(len(all_embs["template_ids"]), 0)

        diag = self.search_engine.identify_with_diagnostics(dummy_probe)
        self.assertFalse(diag["accepted"])
        self.assertIsNone(diag["username"])
        self.assertEqual(diag["score"], -1.0)
        self.assertEqual(diag["ranked_candidates"], [])

    def test_07_duplicate_enrollment_rejected(self):
        """Test 7: Attempting to enroll existing username raises ValueError."""
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)

        db_manager.enroll_user("duplicate_user", [dummy_emb])
        with self.assertRaises(ValueError) as ctx:
            db_manager.enroll_user("duplicate_user", [dummy_emb])
        self.assertIn("already enrolled", str(ctx.exception))

    def test_08_low_quality_and_degenerate_rois(self):
        """Test 8: Degenerate ROIs (pure black, pure white, pure noise) handled without crash."""
        black_roi = np.zeros((224, 224), dtype=np.uint8)
        white_roi = np.full((224, 224), 255, dtype=np.uint8)
        noise_roi = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

        for name, roi in [("black", black_roi), ("white", white_roi), ("noise", noise_roi)]:
            emb = self.engine_onnx.extract_embedding(roi)
            self.assertEqual(emb.shape, (512,))
            self.assertEqual(emb.dtype, np.float32)
            norm = np.linalg.norm(emb)
            self.assertAlmostEqual(float(norm), 1.0, places=4, msg=f"Degenerate {name} ROI must produce normalized embedding")

    def test_09_missing_model_file_handling(self):
        """Test 9: Missing model file initializes gracefully and raises clear RuntimeError on call."""
        bogus_engine = AMPVNetInference(model_path="/nonexistent/path/ampvnet.onnx")
        self.assertFalse(bogus_engine.model_loaded)
        self.assertIn("No ONNX model found", bogus_engine.error_detail)

        with self.assertRaises(RuntimeError) as ctx:
            bogus_engine.extract_embedding(np.zeros((224, 224), dtype=np.uint8))
        self.assertIn("not loaded", str(ctx.exception))

    def test_10_corrupt_embedding_handling(self):
        """Test 10: Corrupt embedding dimensions, NaNs, and wrong byte sizes are rejected."""
        # 1. Invalid length embedding for storage
        corrupt_emb_256 = np.random.randn(256).astype(np.float32)
        with self.assertRaises(ValueError):
            db_manager.enroll_user("corrupt_user", [corrupt_emb_256])

        # 2. Corrupt BLOB size for load_embedding
        with self.assertRaises(ValueError):
            db_manager.load_embedding(b"\x00" * 1024)

        # 3. Probe with NaN / inf handled defensively in search engine
        nan_probe = np.full(512, np.nan, dtype=np.float32)
        # Search engine identify with NaN probe shouldn't crash
        try:
            diag = self.search_engine.identify_with_diagnostics(nan_probe)
            self.assertFalse(diag["accepted"])
        except Exception as e:
            self.fail(f"Search engine crashed on NaN probe: {e}")


if __name__ == "__main__":
    unittest.main()
