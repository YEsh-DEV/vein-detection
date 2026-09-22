#!/usr/bin/env python3
"""
tests/test_offline.py
---------------------
Offline unit and regression tests for the Palm Vein Biometrics System (v2 CNN Pipeline).
Validates:
- D1: Re-enrollment after soft-delete works without UNIQUE constraint crashes
- S1: Empty template candidate set does not crash identify()
- S2: get_templates_by_ids() 3-tuple contract with partial/missing ID queries
- Audit Logging: Successful authentication logs resolved non-NULL integer user_id
- CNN Extractor: Cosine similarity and normalization bounds
- SearchEngine: Best-match MAX aggregation policy
- HTTP 503 Guard: /api/scan and /api/enroll/sample return HTTP 503 when model is absent
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_random_embedding(seed: int = None) -> np.ndarray:
    """Generates a synthetic 512-dim L2-normalized float32 vector."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        v = rng.randn(512).astype(np.float32)
    else:
        v = np.random.randn(512).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32)


class TestPalmVeinV2Offline(unittest.TestCase):

    def setUp(self):
        """Create a temporary SQLite database for isolated test execution."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, "test_palm_vein.db")

        # Patch DB_PATH in constants and db_manager
        self.db_patcher1 = patch("app.constants.DB_PATH", self.test_db_path)
        self.db_patcher2 = patch("app.db_manager.DB_PATH", self.test_db_path)
        self.db_patcher1.start()
        self.db_patcher2.start()

        from app.db_manager import init_db
        init_db()

    def tearDown(self):
        self.db_patcher1.stop()
        self.db_patcher2.stop()
        self.temp_dir.cleanup()

    # ──────────────────────────────────────────────────────────────────────────
    # S1 Bug Class: Empty Database / Candidate Set Guard
    # ──────────────────────────────────────────────────────────────────────────
    def test_s1_empty_database_identify_does_not_crash(self):
        """
        S1 Bug Regression Test: When zero templates exist in the database,
        identify() and identify_with_diagnostics() must return cleanly without
        calling argmax on an empty array.
        """
        from app.search_engine import SearchEngine

        engine = SearchEngine()
        probe = make_random_embedding()

        # identify() contract: returns (None, -1.0, None)
        uname, score, uid = engine.identify(probe)
        self.assertIsNone(uname, "Username must be None on empty database")
        self.assertEqual(score, -1.0, "Score must be -1.0 sentinel on empty database")
        self.assertIsNone(uid, "User ID must be None on empty database")

        # identify_with_diagnostics() contract
        diag = engine.identify_with_diagnostics(probe)
        self.assertIsNone(diag['username'])
        self.assertEqual(diag['score'], -1.0)
        self.assertIsNone(diag['user_id'])
        self.assertFalse(diag['accepted'])
        self.assertEqual(diag['ranked_candidates'], [])
        self.assertNotIn('l1_filtered_out', diag, "l1_filtered_out must be cleanly removed in v2")

    # ──────────────────────────────────────────────────────────────────────────
    # S2 Bug Class: 3-Tuple Contract with Partial / Missing IDs
    # ──────────────────────────────────────────────────────────────────────────
    def test_s2_get_templates_by_ids_contract_partial_missing(self):
        """
        S2 Bug Regression Test:
        get_templates_by_ids(template_ids) must return (template_id, user_id, template_dict)
        tuples where template_dict = {"embedding": np.ndarray(512,)}.
        If 5 IDs are queried and only 3 exist, it must return exactly 3 tuples,
        strictly matched to their true user_ids without length mismatch or index corruption.
        """
        from app.db_manager import enroll_user, get_templates_by_ids

        # Enroll user 1 with 2 samples
        u1_embs = [make_random_embedding(1), make_random_embedding(2)]
        uid1 = enroll_user("alice", u1_embs)

        # Enroll user 2 with 2 samples
        u2_embs = [make_random_embedding(3), make_random_embedding(4)]
        uid2 = enroll_user("bob", u2_embs)

        # Total 4 templates exist with IDs 1, 2, 3, 4
        # Query with non-existent IDs (e.g. 999, 888) interleaved
        query_ids = [1, 999, 3, 888, 4]
        results = get_templates_by_ids(query_ids)

        self.assertEqual(len(results), 3, "Must return exactly 3 tuples for existing IDs")

        # Check structure of each tuple
        expected_pairs = {1: uid1, 3: uid2, 4: uid2}
        for tid, uid, tdict in results:
            self.assertIn(tid, expected_pairs, f"Unexpected template_id {tid}")
            self.assertEqual(uid, expected_pairs[tid], f"Template {tid} paired with wrong user_id {uid}")
            self.assertIsInstance(tdict, dict)
            self.assertIn("embedding", tdict)
            self.assertIsInstance(tdict["embedding"], np.ndarray)
            self.assertEqual(tdict["embedding"].shape, (512,))
            self.assertEqual(tdict["embedding"].dtype, np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # D1 Bug Class: Soft-Delete Re-Enrollment Hygiene
    # ──────────────────────────────────────────────────────────────────────────
    def test_d1_re_enrollment_after_soft_delete_succeeds(self):
        """
        D1 Bug Regression Test: When a user is soft-deleted (active=0),
        re-enrolling with the same username must cleanly purge the stale records
        and re-insert without SQLite UNIQUE constraint errors.
        """
        from app.db_manager import enroll_user, delete_user, user_exists, list_users

        embs_v1 = [make_random_embedding(10), make_random_embedding(11), make_random_embedding(12)]
        uid_v1 = enroll_user("charlie", embs_v1)
        self.assertTrue(user_exists("charlie"))

        # Soft delete
        delete_user("charlie")
        self.assertFalse(user_exists("charlie"))

        # Re-enroll with new embeddings
        embs_v2 = [make_random_embedding(20), make_random_embedding(21), make_random_embedding(22)]
        uid_v2 = enroll_user("charlie", embs_v2)

        self.assertTrue(user_exists("charlie"))
        users = list_users()
        charlie_entry = [u for u in users if u['username'] == 'charlie']
        self.assertEqual(len(charlie_entry), 1)
        self.assertEqual(charlie_entry[0]['sample_count'], 3)

    # ──────────────────────────────────────────────────────────────────────────
    # Audit Logging: Resolved User ID Integrity
    # ──────────────────────────────────────────────────────────────────────────
    def test_audit_logging_writes_resolved_user_id_on_match(self):
        """
        Audit Log Requirement: On genuine match (accepted=True), access_log.user_id
        must be the resolved integer primary key of the matched user, never NULL.
        """
        import sqlite3
        from app.db_manager import enroll_user, log_access, get_user_id

        embs = [make_random_embedding(30), make_random_embedding(31), make_random_embedding(32)]
        uid = enroll_user("david", embs)

        # Log genuine accepted match
        log_access(user_id=uid, score=0.88, accepted=True)

        # Log rejected attempt
        log_access(user_id=None, score=0.21, accepted=False)

        with sqlite3.connect(self.test_db_path) as conn:
            rows = conn.execute("SELECT user_id, score, accepted FROM access_log ORDER BY id").fetchall()

        self.assertEqual(len(rows), 2)
        # Row 1: Genuine match with resolved user_id
        self.assertEqual(rows[0][0], uid, "Accepted match must store resolved integer user_id")
        self.assertAlmostEqual(rows[0][1], 0.88, places=2)
        self.assertEqual(rows[0][2], 1)

        # Row 2: Rejected scan with NULL user_id
        self.assertIsNone(rows[1][0], "Rejected match stores NULL user_id")
        self.assertAlmostEqual(rows[1][1], 0.21, places=2)
        self.assertEqual(rows[1][2], 0)

    # ──────────────────────────────────────────────────────────────────────────
    # SearchEngine: Best-Match MAX Policy & Cosine Ranking
    # ──────────────────────────────────────────────────────────────────────────
    def test_search_engine_max_policy_and_ranking(self):
        """
        SearchEngine Matching Logic Test:
        - Must aggregate scores per user using MAX (best-match), not MIN.
        - Higher cosine similarity = better match.
        - Candidate ranking sorted descending.
        """
        from app.db_manager import enroll_user
        from app.search_engine import SearchEngine

        probe = make_random_embedding(100)

        # User 1: One low match (0.2), one high match (0.85)
        # Construct template with specific cosine similarity to probe
        ortho1 = make_random_embedding(101)
        ortho1 = ortho1 - np.dot(ortho1, probe) * probe
        ortho1 = ortho1 / np.linalg.norm(ortho1)

        # Create embedding with exact dot product 0.85 with probe
        t_high = 0.85 * probe + np.sqrt(1 - 0.85**2) * ortho1
        t_low = 0.20 * probe + np.sqrt(1 - 0.20**2) * ortho1

        uid_eva = enroll_user("eva", [t_low, t_high])

        # User 2: Moderate matches (0.60, 0.65)
        t_med1 = 0.60 * probe + np.sqrt(1 - 0.60**2) * ortho1
        t_med2 = 0.65 * probe + np.sqrt(1 - 0.65**2) * ortho1
        uid_frank = enroll_user("frank", [t_med1, t_med2])

        engine = SearchEngine()
        diag = engine.identify_with_diagnostics(probe)

        # User Eva's best template is 0.85, Frank's is 0.65
        # Under MAX policy, Eva wins with ~0.85
        self.assertTrue(diag['accepted'])
        self.assertEqual(diag['username'], "eva")
        self.assertEqual(diag['user_id'], uid_eva)
        self.assertAlmostEqual(diag['score'], 0.85, places=2)

        # Ranked candidates should have Eva first, Frank second
        ranked = diag['ranked_candidates']
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]['username'], "eva")
        self.assertAlmostEqual(ranked[0]['score'], 0.85, places=2)
        self.assertEqual(ranked[1]['username'], "frank")
        self.assertAlmostEqual(ranked[1]['score'], 0.65, places=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Cosine Similarity Utility Test
    # ──────────────────────────────────────────────────────────────────────────
    def test_cosine_similarity_bounds_and_values(self):
        """Validates cosine_similarity dot product, unit vector handling, and clipping."""
        from app.cnn_extractor import cosine_similarity

        v1 = make_random_embedding(1)
        # Identical vectors -> 1.0
        self.assertAlmostEqual(cosine_similarity(v1, v1), 1.0, places=5)

        # Opposite vectors -> -1.0
        self.assertAlmostEqual(cosine_similarity(v1, -v1), -1.0, places=5)

        # Orthogonal vector -> 0.0
        v2 = make_random_embedding(2)
        v_ortho = v2 - np.dot(v2, v1) * v1
        v_ortho = v_ortho / np.linalg.norm(v_ortho)
        self.assertAlmostEqual(cosine_similarity(v1, v_ortho), 0.0, places=5)

    # ──────────────────────────────────────────────────────────────────────────
    # HTTP 503 Guard: Absent Model Returns 503, Not Unhandled Crash
    # ──────────────────────────────────────────────────────────────────────────
    def test_missing_model_returns_http_503(self):
        """
        Verifies that when MODEL_LOADED is False (e.g. models/ampvnet.onnx is absent),
        FastAPI endpoints /api/scan and /api/enroll/sample return HTTP 503 with an informative
        error message rather than crashing with an unhandled exception.
        """
        from fastapi.testclient import TestClient
        import app.server as server_module

        # Ensure MODEL_LOADED is False for this test
        with patch.object(server_module, "MODEL_LOADED", False):
            client = TestClient(server_module.app, raise_server_exceptions=False)

            # Test 1: /api/enroll/sample returns 503
            res_enroll = client.post("/api/enroll/sample", json={"username": "grace"})
            self.assertEqual(
                res_enroll.status_code, 503,
                f"Expected HTTP 503 for enroll/sample when model not loaded, got {res_enroll.status_code}"
            )
            self.assertIn("CNN model not loaded", res_enroll.text)

            # Test 2: /api/scan returns 503
            res_scan = client.post("/api/scan")
            self.assertEqual(
                res_scan.status_code, 503,
                f"Expected HTTP 503 for scan when model not loaded, got {res_scan.status_code}"
            )
            self.assertIn("CNN model not loaded", res_scan.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
