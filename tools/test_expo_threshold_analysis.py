#!/usr/bin/env python3
"""
tools/test_expo_threshold_analysis.py
-------------------------------------
Unit and integration tests for tools/expo_threshold_analysis.py.
Runnable with plain python:
    python tools/test_expo_threshold_analysis.py
or with unittest / pytest:
    python -m unittest discover -s tools -p "test_*.py"
"""

import json
import math
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

# Import directly from tools.expo_threshold_analysis or local path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.expo_threshold_analysis import (
    parse_utc_datetime,
    mean,
    sample_std,
    percentile_linear,
    binom_pmf,
    binom_cdf,
    clopper_pearson,
    rule_of_three,
    ScanRecord,
    parse_and_clean_scans,
    apply_labels,
    build_evaluation_sets,
    evaluate_threshold_grid,
    recommend_thresholds,
    analyze_margin_rule,
    per_user_check,
    format_console_report,
)


class TestStatisticalFunctions(unittest.TestCase):
    def test_hand_computed_summary_statistics(self):
        """
        Verify mean, sample std (ddof=1), and percentiles against exact hand calculations.
        Data = [10.0, 20.0, 30.0, 40.0, 50.0]
        Mean = 30.0
        Sample std = sqrt(250.0) ~= 15.8113883
        5th percentile: q = 0.05 * 4 = 0.20 => 10 + 0.2*(20 - 10) = 12.0
        Median (50th): 30.0
        95th percentile: q = 0.95 * 4 = 3.80 => 40 + 0.8*(50 - 40) = 48.0
        """
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(mean(data), 30.0, places=6)
        self.assertAlmostEqual(sample_std(data), math.sqrt(250.0), places=6)
        self.assertAlmostEqual(percentile_linear(data, 5.0), 12.0, places=6)
        self.assertAlmostEqual(percentile_linear(data, 50.0), 30.0, places=6)
        self.assertAlmostEqual(percentile_linear(data, 95.0), 48.0, places=6)
        self.assertAlmostEqual(percentile_linear(data, 0.0), 10.0, places=6)
        self.assertAlmostEqual(percentile_linear(data, 100.0), 50.0, places=6)

    def test_clopper_pearson_exact_bounds(self):
        """
        Assert Clopper-Pearson confidence intervals match required values:
        - For k=0, n=30: gives an upper bound close to 0.0950
        - For k=3, n=30: gives about (0.021, 0.265)
        """
        low_0, high_0 = clopper_pearson(0, 30, alpha=0.05)
        self.assertEqual(low_0, 0.0)
        # 1 - 0.05^(1/30) = 0.0950338...
        self.assertAlmostEqual(high_0, 0.0950, places=3)

        low_3, high_3 = clopper_pearson(3, 30, alpha=0.05)
        self.assertAlmostEqual(low_3, 0.021, places=3)
        self.assertAlmostEqual(high_3, 0.265, places=3)

        # Rule of three
        self.assertAlmostEqual(rule_of_three(30), 0.100, places=3)


class TestThresholdEvaluationAndBoundary(unittest.TestCase):
    def test_far_tar_hand_calculated_including_boundary(self):
        """
        Tests FAR and TAR at several thresholds, specifically checking that
        a score exactly equal to t is accepted (>= boundary).
        """
        # Impostor top1 scores: 0.60, 0.70, 0.80, 0.90
        # Genuine scores:       0.70, 0.80, 0.85, 0.95
        i_scores = [0.60, 0.70, 0.80, 0.90]
        g_scores = [0.70, 0.80, 0.85, 0.95]

        # Synthetic genuine scan records
        dummy_ts = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
        g_scans = []
        for s in g_scores:
            rec = ScanRecord(
                timestamp=dummy_ts,
                timestamp_raw=dummy_ts.isoformat(),
                decision="ACCEPTED",
                matched_user="alice",
                matched_user_id=1,
                score=s,
                threshold=0.75,
                ranked_candidates=[{"username": "alice", "user_id": 1, "score": s}],
                top1_user="alice",
                top1=s,
                top2_user="",
                top2=-1.0,
                margin=0.0,
            )
            rec.label = "GENUINE"
            rec.labeled_user = "alice"
            g_scans.append(rec)

        grid = evaluate_threshold_grid(i_scores, g_scores, g_scans)
        grid_by_t = {round(r["threshold"], 4): r for r in grid}

        # Boundary test at t = 0.70:
        # Impostors >= 0.70 are: 0.70, 0.80, 0.90 => 3 / 4 = 0.75
        # Genuines >= 0.70 are: 0.70, 0.80, 0.85, 0.95 => 4 / 4 = 1.00
        self.assertIn(0.70, grid_by_t)
        r_070 = grid_by_t[0.70]
        self.assertAlmostEqual(r_070["far"], 0.75, places=4)
        self.assertAlmostEqual(r_070["tar"], 1.00, places=4)
        self.assertAlmostEqual(r_070["frr"], 0.00, places=4)

        # Boundary test at t = 0.80:
        # Impostors >= 0.80 are: 0.80, 0.90 => 2 / 4 = 0.50
        # Genuines >= 0.80 are: 0.80, 0.85, 0.95 => 3 / 4 = 0.75
        self.assertIn(0.80, grid_by_t)
        r_080 = grid_by_t[0.80]
        self.assertAlmostEqual(r_080["far"], 0.50, places=4)
        self.assertAlmostEqual(r_080["tar"], 0.75, places=4)
        self.assertAlmostEqual(r_080["frr"], 0.25, places=4)

        # Test at t = 0.91:
        # Impostors >= 0.91 is empty => 0 / 4 = 0.00
        # Genuines >= 0.91 is 0.95 => 1 / 4 = 0.25
        self.assertIn(0.91, grid_by_t)
        r_091 = grid_by_t[0.91]
        self.assertAlmostEqual(r_091["far"], 0.00, places=4)
        self.assertAlmostEqual(r_091["tar"], 0.25, places=4)


class TestLabelingRulesAndTimezones(unittest.TestCase):
    def test_auto_before_enrollment_and_timezone_handling(self):
        """
        Verify that auto_before_enrollment labels a scan as IMPOSTOR when
        matched user was enrolled later, and does NOT when enrolled earlier.
        Also verifies ISO 'Z' strings vs naive SQLite UTC strings.
        """
        # User in SQLite DB: enrolled_at is a naive string generated by datetime('now')
        # e.g. "2026-09-25 10:00:00" (UTC)
        db_users = {
            "alice": {
                "id": 1,
                "username": "alice",
                "enrolled_at": parse_utc_datetime("2026-09-25 10:00:00"),
                "active": 1,
            }
        }

        # Scan 1: at 09:30:00 UTC (with 'Z' suffix) — 30 minutes BEFORE enrollment
        ts1 = parse_utc_datetime("2026-09-25T09:30:00Z")
        rec1 = ScanRecord(
            timestamp=ts1,
            timestamp_raw="2026-09-25T09:30:00Z",
            decision="REJECTED",
            matched_user=None,
            matched_user_id=None,
            score=0.68,
            threshold=0.75,
            ranked_candidates=[{"username": "alice", "score": 0.68}],
            top1_user="alice",
            top1=0.68,
            top2_user="",
            top2=-1.0,
            margin=0.0,
        )

        # Scan 2: at 10:05:00 UTC (with '+00:00' suffix) — 5 minutes AFTER enrollment
        ts2 = parse_utc_datetime("2026-09-25T10:05:00+00:00")
        rec2 = ScanRecord(
            timestamp=ts2,
            timestamp_raw="2026-09-25T10:05:00+00:00",
            decision="ACCEPTED",
            matched_user="alice",
            matched_user_id=1,
            score=0.88,
            threshold=0.75,
            ranked_candidates=[{"username": "alice", "score": 0.88}],
            top1_user="alice",
            top1=0.88,
            top2_user="",
            top2=-1.0,
            margin=0.0,
        )

        # Scan 3: at 12:00:00 UTC — 2 hours after enrollment
        ts3 = parse_utc_datetime("2026-09-25T12:00:00Z")
        rec3 = ScanRecord(
            timestamp=ts3,
            timestamp_raw="2026-09-25T12:00:00Z",
            decision="ACCEPTED",
            matched_user="alice",
            matched_user_id=1,
            score=0.85,
            threshold=0.75,
            ranked_candidates=[{"username": "alice", "score": 0.85}],
            top1_user="alice",
            top1=0.85,
            top2_user="",
            top2=-1.0,
            margin=0.0,
        )

        records = [rec1, rec2, rec3]
        apply_labels(records, manual_labels=[], db_users=db_users, post_enroll_minutes=10)

        # Scan 1 MUST be IMPOSTOR via auto_before_enrollment
        self.assertEqual(rec1.label, "IMPOSTOR")
        self.assertEqual(rec1.label_source, "auto_before_enrollment")

        # Scan 2 MUST be GENUINE via auto_post_enrollment
        self.assertEqual(rec2.label, "GENUINE")
        self.assertEqual(rec2.label_source, "auto_post_enrollment")

        # Scan 3 MUST be UNLABELED (outside 10 min window)
        self.assertEqual(rec3.label, "UNLABELED")
        self.assertEqual(rec3.label_source, "unlabeled")

    def test_misidentified_genuine_scan_score_extraction(self):
        """
        Asserts that a genuine scan that was misidentified by the matcher
        uses the labeled user's own score from ranked_candidates.
        """
        # Genuine scan labeled as 'alice', but top1 match is 'bob' (0.82)
        # Alice is rank 2 with score 0.76
        ts = parse_utc_datetime("2026-09-25T14:00:00Z")
        rec = ScanRecord(
            timestamp=ts,
            timestamp_raw="2026-09-25T14:00:00Z",
            decision="ACCEPTED",
            matched_user="bob",
            matched_user_id=2,
            score=0.82,
            threshold=0.75,
            ranked_candidates=[
                {"username": "bob", "score": 0.82},
                {"username": "alice", "score": 0.76},
            ],
            top1_user="bob",
            top1=0.82,
            top2_user="alice",
            top2=0.76,
            margin=0.06,
        )
        rec.label = "GENUINE"
        rec.labeled_user = "alice"

        i_scores, g_scores, i_scans, g_scans, missing_count = build_evaluation_sets([rec])

        self.assertEqual(missing_count, 0)
        self.assertEqual(len(g_scores), 1)
        # Must be Alice's own score 0.76, NOT Bob's top1 score 0.82
        self.assertEqual(g_scores[0], 0.76)


class TestRobustnessAndEdgeCases(unittest.TestCase):
    def test_zero_impostor_and_zero_genuine_inputs(self):
        """
        Asserts that zero-impostor and zero-genuine inputs do not crash and report clearly.
        """
        i_empty = []
        g_empty = []
        g_scans = []

        grid = evaluate_threshold_grid(i_empty, g_empty, g_scans)
        recs = recommend_thresholds(i_empty, g_empty, grid)
        margin = analyze_margin_rule(g_scans, [])
        users = per_user_check([], {})

        report = format_console_report(
            clean_counts={"total_lines": 0, "malformed_lines": 0, "out_of_window": 0, "non_scan_operations": 0, "empty_candidates": 0, "kept_scans": 0},
            label_counts={},
            capture_failures=0,
            g_scores=g_empty,
            i_scores=i_empty,
            grid_results=grid,
            recommendations=recs,
            margin_analysis=margin,
            user_analysis=users,
        )

        self.assertIn("PALM VEIN BIOMETRIC EXPO THRESHOLD ANALYSIS REPORT", report)
        self.assertIn("ZERO_FALSE_ACCEPT", report)
        self.assertIn("BALANCED_SAFE", report)

    def test_malformed_and_truncated_jsonl_skipped(self):
        """
        Asserts that malformed JSONL lines and truncated trailing lines are skipped
        without crashing and counted properly.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            log_path = os.path.join(temp_dir, "test_scans.jsonl")
            with open(log_path, "w", encoding="utf-8") as f:
                # Line 1: valid scan
                f.write(json.dumps({
                    "timestamp": "2026-09-25T10:00:00Z",
                    "operation": "scan",
                    "decision": "ACCEPTED",
                    "matched_user": "alice",
                    "score": 0.85,
                    "ranked_candidates": [{"username": "alice", "score": 0.85}, {"username": "bob", "score": 0.60}],
                }) + "\n")
                # Line 2: malformed JSON (missing closing braces)
                f.write('{"timestamp": "2026-09-25T10:01:00Z", "operation": "scan", "ranked_candidates": [\n')
                # Line 3: valid scan
                f.write(json.dumps({
                    "timestamp": "2026-09-25T10:02:00Z",
                    "operation": "scan",
                    "decision": "REJECTED",
                    "matched_user": None,
                    "score": 0.55,
                    "ranked_candidates": [{"username": "alice", "score": 0.55}],
                }) + "\n")
                # Line 4: truncated last line (e.g. server in the middle of writing)
                f.write('{"timestamp": "2026-09-25T10:03:00Z", "operation": "sca')

            records, counts = parse_and_clean_scans(log_path, db_users={})

            self.assertEqual(counts["total_lines"], 4)
            self.assertEqual(counts["malformed_lines"], 2)  # lines 2 and 4
            self.assertEqual(counts["kept_scans"], 2)       # lines 1 and 3
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].top1_user, "alice")
            self.assertEqual(records[0].top1, 0.85)
            self.assertEqual(records[0].top2_user, "bob")
            self.assertEqual(records[0].top2, 0.60)
            self.assertAlmostEqual(records[0].margin, 0.25, places=4)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
