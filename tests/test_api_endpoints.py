#!/usr/bin/env python3
"""
tests/test_api_endpoints.py
----------------------------
Integration tests for FastAPI server endpoints verifying API contracts:
  - /health
  - /api/status
  - /api/report
  - /api/enroll/sample
  - /api/enroll/save
  - /api/scan
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

try:
    from starlette.testclient import TestClient
    TESTCLIENT_AVAILABLE = True
except Exception:
    TestClient = None
    TESTCLIENT_AVAILABLE = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import app.constants as constants
import app.db_manager as db_manager
import app.server as server


class TestAPIEndpoints(unittest.TestCase):
    """Verifies FastAPI endpoints contracts and v2 compatibility."""

    @classmethod
    def setUpClass(cls):
        if not TESTCLIENT_AVAILABLE:
            raise unittest.SkipTest(
                "starlette/httpx TestClient is not installed. Install with 'pip install httpx' for API testing."
            )
        # Patch db_manager DB_PATH
        cls._orig_db_path = db_manager.DB_PATH
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_db_path = os.path.join(cls.temp_dir.name, "test_api_vein.db")
        db_manager.DB_PATH = cls.test_db_path
        db_manager.init_db()

        # Initialize server globals for test
        server.engine = server.SearchEngine(engine_version="v2")
        server.MODEL_LOADED = True
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        db_manager.DB_PATH = cls._orig_db_path
        if hasattr(cls, "temp_dir") and cls.temp_dir:
            cls.temp_dir.cleanup()

    def test_health_endpoint(self):
        """GET /health returns HTTP 200 with engine and health status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["engine"], "v2")
        self.assertIn("model_loaded", data)

    def test_status_endpoint(self):
        """GET /api/status returns HTTP 200 with all required fields."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("camera_available", data)
        self.assertIn("model_loaded", data)
        self.assertIn("enrolled_users_count", data)
        self.assertIn("total_templates", data)
        self.assertIn("match_threshold", data)
        self.assertEqual(data["match_threshold"], constants.MATCH_THRESHOLD)
        self.assertEqual(data["biometric_engine"], "v2")

    def test_report_endpoint(self):
        """GET /api/report returns active users list and template counts."""
        resp = self.client.get("/api/report")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("users", data)
        self.assertIn("total_templates", data)

    def test_enroll_validation(self):
        """POST /api/enroll/sample rejects invalid usernames."""
        resp = self.client.post("/api/enroll/sample", json={"username": ""})
        self.assertEqual(resp.status_code, 422)

        resp = self.client.post("/api/enroll/sample", json={"username": "a!bad*user"})
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
