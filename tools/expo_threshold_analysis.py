#!/usr/bin/env python3
"""
tools/expo_threshold_analysis.py
--------------------------------
Offline threshold analysis tool for palm-vein biometric kiosk.
Reads JSONL scan diagnostics and SQLite DB (in read-only mode) to determine
optimal matching threshold and margin parameters for live expo operation.

Python 3.9+ standard library only (numpy / matplotlib optional).
Safe to run while kiosk server is actively running.
"""

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any


# ── Statistical & Mathematical Helper Functions (Pure Standard Library) ─────

def parse_utc_datetime(ts_val: Any) -> Optional[datetime]:
    """
    Parses various timestamp representations into a timezone-aware UTC datetime.
    Supports ISO 8601 ('2026-09-23T08:03:36.684372+00:00', '...Z'),
    SQLite datetime('now') ('YYYY-MM-DD HH:MM:SS'), and standard time formats.
    """
    if ts_val is None:
        return None
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val.astimezone(timezone.utc)

    s = str(ts_val).strip()
    if not s:
        return None

    # Handle trailing Z
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"

    # Try fromisoformat (Python 3.11+ handles space and T, earlier handles T)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        pass

    # Fallback format parsing
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%H:%M:%S":
                now = datetime.now(timezone.utc)
                dt = dt.replace(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    return None


def mean(data: List[float]) -> float:
    """Computes arithmetic mean of a float list."""
    if not data:
        return 0.0
    return sum(data) / float(len(data))


def sample_std(data: List[float]) -> float:
    """Computes sample standard deviation (ddof=1) of a float list."""
    n = len(data)
    if n <= 1:
        return 0.0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / float(n - 1)
    return math.sqrt(variance)


def percentile_linear(data: List[float], p: float) -> float:
    """
    Computes p-th percentile with linear interpolation, matching numpy default.
    p must be between 0.0 and 100.0.
    """
    if not data:
        return 0.0
    s_data = sorted(data)
    n = len(s_data)
    if n == 1:
        return float(s_data[0])
    if p <= 0.0:
        return float(s_data[0])
    if p >= 100.0:
        return float(s_data[-1])

    q = (p / 100.0) * (n - 1)
    i = int(q)
    f = q - i
    if i >= n - 1:
        return float(s_data[-1])
    return float(s_data[i] + f * (s_data[i + 1] - s_data[i]))


def binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass function: P(X = k) for X ~ Bin(n, p)."""
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binom_cdf(k: int, n: int, p: float) -> float:
    """Binomial cumulative distribution function: P(X <= k) for X ~ Bin(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return sum(binom_pmf(j, n, p) for j in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    95% Clopper-Pearson (exact binomial) confidence interval for proportion k/n.
    Implemented without scipy using bisection on the binomial CDF.
    For k=0, computes one-sided 95% upper bound: 1 - alpha^(1/n) (approx 0.0950 for n=30).
    For k=n, computes one-sided lower bound: alpha^(1/n).
    For 0 < k < n, performs bisection to find exact tail probabilities alpha/2.
    """
    if n <= 0:
        return (0.0, 1.0)
    if k <= 0:
        # Standard one-sided upper bound for 0 observed events: (1 - p)^n = alpha => p = 1 - alpha^(1/n)
        return (0.0, 1.0 - (alpha ** (1.0 / float(n))))
    if k >= n:
        return (alpha ** (1.0 / float(n)), 1.0)

    # Lower bound: p such that P(Bin(n, p) >= k) = alpha / 2 => binom_cdf(k - 1, n, p) = 1 - alpha / 2
    low, high = 0.0, float(k) / float(n)
    target_low = 1.0 - (alpha / 2.0)
    for _ in range(60):
        mid = (low + high) / 2.0
        if binom_cdf(k - 1, n, mid) > target_low:
            low = mid
        else:
            high = mid
    p_lower = (low + high) / 2.0

    # Upper bound: p such that P(Bin(n, p) <= k) = alpha / 2 => binom_cdf(k, n, p) = alpha / 2
    low, high = float(k) / float(n), 1.0
    target_high = alpha / 2.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if binom_cdf(k, n, mid) < target_high:
            high = mid
        else:
            low = mid
    p_upper = (low + high) / 2.0

    return (p_lower, p_upper)


def rule_of_three(n: int) -> float:
    """Rule of three upper bound for zero observed events: 3 / n."""
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / float(n))


# ── Data Loading & Cleansing (Step 0 & Step 1) ────────────────────────────────

class ScanRecord:
    def __init__(
        self,
        timestamp: datetime,
        timestamp_raw: str,
        decision: str,
        matched_user: Optional[str],
        matched_user_id: Optional[int],
        score: float,
        threshold: float,
        ranked_candidates: List[Dict[str, Any]],
        top1_user: str,
        top1: float,
        top2_user: str,
        top2: float,
        margin: float,
    ):
        self.timestamp = timestamp
        self.timestamp_raw = timestamp_raw
        self.decision = decision
        self.matched_user = matched_user
        self.matched_user_id = matched_user_id
        self.score = score
        self.threshold = threshold
        self.ranked_candidates = ranked_candidates
        self.top1_user = top1_user
        self.top1 = top1
        self.top2_user = top2_user
        self.top2 = top2
        self.margin = margin
        self.label = "UNLABELED"
        self.label_source = "unlabeled"
        self.labeled_user: Optional[str] = None


def load_users_from_db(db_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Connects to SQLite in read-only mode (mode=ro URI) and loads all users.
    Returns mapping: username -> {id, username, enrolled_at, active}
    """
    if not os.path.exists(db_path):
        return {}

    abs_db = os.path.abspath(db_path)
    uri = f"file:{abs_db}?mode=ro"
    users = {}
    try:
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()
        cur.execute("SELECT id, username, enrolled_at, active FROM users")
        for uid, uname, enrolled_at_str, active in cur.fetchall():
            uname_clean = str(uname).strip().lower()
            enrolled_at_dt = parse_utc_datetime(enrolled_at_str)
            users[uname_clean] = {
                "id": uid,
                "username": uname_clean,
                "enrolled_at": enrolled_at_dt,
                "enrolled_at_raw": enrolled_at_str,
                "active": int(active) if active is not None else 1,
            }
        conn.close()
    except Exception as e:
        print(f"[!] Warning: Could not read SQLite DB at '{db_path}': {e}", file=sys.stderr)

    return users


def load_capture_failures(capture_log_path: str, since: Optional[datetime], until: Optional[datetime]) -> int:
    """Reads capture_diagnostics.jsonl to count failed captures excluded from analysis."""
    if not os.path.exists(capture_log_path):
        return 0

    count = 0
    try:
        with open(capture_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ts = parse_utc_datetime(data.get("timestamp"))
                    if ts:
                        if since and ts < since:
                            continue
                        if until and ts > until:
                            continue
                    count += 1
                except Exception:
                    continue
    except Exception:
        pass
    return count


def load_manual_labels(labels_csv_path: str) -> List[Dict[str, Any]]:
    """
    Reads manual labels CSV if it exists.
    Expected columns: start_time, end_time, label, username.
    """
    if not os.path.exists(labels_csv_path):
        return []

    labels = []
    try:
        with open(labels_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_dt = parse_utc_datetime(row.get("start_time"))
                end_dt = parse_utc_datetime(row.get("end_time"))
                lbl = str(row.get("label", "")).strip().lower()
                uname = str(row.get("username", "")).strip().lower() or None
                if start_dt and end_dt and lbl:
                    labels.append({
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "label": lbl,
                        "username": uname,
                    })
    except Exception as e:
        print(f"[!] Warning: Failed reading manual labels file '{labels_csv_path}': {e}", file=sys.stderr)

    return labels


def parse_and_clean_scans(
    log_path: str,
    db_users: Dict[str, Dict[str, Any]],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Tuple[List[ScanRecord], Dict[str, int]]:
    """
    Reads scan_diagnostics.jsonl line-by-line.
    Filters by window and valid operation, computes top1, top2, margin.
    Tolerates partially written / malformed lines.
    """
    records: List[ScanRecord] = []
    counts = {
        "total_lines": 0,
        "malformed_lines": 0,
        "out_of_window": 0,
        "non_scan_operations": 0,
        "empty_candidates": 0,
        "kept_scans": 0,
    }

    if not os.path.exists(log_path):
        return records, counts

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            counts["total_lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_lines"] += 1
                continue

            ts_raw = data.get("timestamp")
            ts = parse_utc_datetime(ts_raw)
            if not ts:
                counts["malformed_lines"] += 1
                continue

            if since and ts < since:
                counts["out_of_window"] += 1
                continue
            if until and ts > until:
                counts["out_of_window"] += 1
                continue

            if data.get("operation") != "scan":
                counts["non_scan_operations"] += 1
                continue

            raw_candidates = data.get("ranked_candidates") or []
            if not isinstance(raw_candidates, list) or len(raw_candidates) == 0:
                counts["empty_candidates"] += 1
                continue

            # Sort ranked_candidates independently descending by score
            try:
                candidates = sorted(raw_candidates, key=lambda c: float(c.get("score", -1.0)), reverse=True)
            except Exception:
                counts["malformed_lines"] += 1
                continue

            top1_c = candidates[0]
            top1_user = str(top1_c.get("username", "")).strip().lower()
            try:
                top1_score = float(top1_c.get("score", 0.0))
            except Exception:
                top1_score = float(data.get("score", 0.0))

            # Find top-2 candidate from a DIFFERENT user
            top2_user = ""
            top2_score = -1.0
            for c in candidates[1:]:
                c_user = str(c.get("username", "")).strip().lower()
                if c_user != top1_user:
                    top2_user = c_user
                    try:
                        top2_score = float(c.get("score", -1.0))
                    except Exception:
                        top2_score = -1.0
                    break

            margin = top1_score - top2_score if top2_score != -1.0 else 0.0

            rec = ScanRecord(
                timestamp=ts,
                timestamp_raw=str(ts_raw),
                decision=str(data.get("decision", "REJECTED")).upper(),
                matched_user=str(data.get("matched_user")).strip().lower() if data.get("matched_user") else None,
                matched_user_id=data.get("matched_user_id"),
                score=top1_score,
                threshold=float(data.get("threshold", 0.75)),
                ranked_candidates=candidates,
                top1_user=top1_user,
                top1=top1_score,
                top2_user=top2_user,
                top2=top2_score,
                margin=margin,
            )
            records.append(rec)
            counts["kept_scans"] += 1

    return records, counts


# ── Labeling Engine (Step 2) ──────────────────────────────────────────────────

def apply_labels(
    records: List[ScanRecord],
    manual_labels: List[Dict[str, Any]],
    db_users: Dict[str, Dict[str, Any]],
    post_enroll_minutes: int = 10,
    interactive: bool = False,
    labels_csv_path: str = "logs/expo_labels.csv",
) -> Dict[str, int]:
    """
    Labels each scan according to strict priority order:
    1. Manual labels file logs/expo_labels.csv
    2. Automatic IMPOSTOR rule (certain): scan < top1_user.enrolled_at (if reliable)
    3. Automatic GENUINE rule (probable): scan within post_enroll_minutes after top1_user.enrolled_at
    4. Interactive user input (if --interactive)
    5. UNLABELED
    """
    # Detect re-enrolled users: users whose current enrolled_at is LATER than a previously
    # ACCEPTED scan for that username. If re-enrolled, enrolled_at was reset, so Rule 2 is unreliable.
    re_enrolled_users = set()
    for rec in records:
        if rec.decision == "ACCEPTED" and rec.matched_user and rec.matched_user in db_users:
            enrolled_at = db_users[rec.matched_user].get("enrolled_at")
            if enrolled_at and rec.timestamp < enrolled_at:
                re_enrolled_users.add(rec.matched_user)

    label_counts = {
        "manual_genuine": 0,
        "manual_impostor": 0,
        "auto_before_enrollment": 0,
        "auto_post_enrollment": 0,
        "unlabeled": 0,
    }

    # Pass 1: Automated labeling rules
    for rec in records:
        # Rule 1: Manual labels
        matched_manual = None
        for ml in manual_labels:
            if ml["start_dt"] <= rec.timestamp <= ml["end_dt"]:
                matched_manual = ml
                break

        if matched_manual:
            if matched_manual["label"] == "genuine":
                rec.label = "GENUINE"
                rec.label_source = "manual"
                rec.labeled_user = matched_manual["username"] or rec.top1_user
                label_counts["manual_genuine"] += 1
            else:
                rec.label = "IMPOSTOR"
                rec.label_source = "manual"
                rec.labeled_user = None
                label_counts["manual_impostor"] += 1
            continue

        # Rule 2: Automatic IMPOSTOR rule (scan before enrollment)
        user_info = db_users.get(rec.top1_user)
        if user_info and user_info.get("enrolled_at"):
            enrolled_at = user_info["enrolled_at"]
            if rec.top1_user not in re_enrolled_users:
                if rec.timestamp < enrolled_at:
                    rec.label = "IMPOSTOR"
                    rec.label_source = "auto_before_enrollment"
                    rec.labeled_user = None
                    label_counts["auto_before_enrollment"] += 1
                    continue

        # Rule 3: Automatic GENUINE rule (scan within 10 min after enrollment)
        if user_info and user_info.get("enrolled_at"):
            enrolled_at = user_info["enrolled_at"]
            delta = (rec.timestamp - enrolled_at).total_seconds()
            if 0 <= delta <= (post_enroll_minutes * 60):
                rec.label = "GENUINE"
                rec.label_source = "auto_post_enrollment"
                rec.labeled_user = rec.top1_user
                label_counts["auto_post_enrollment"] += 1
                continue

        rec.label = "UNLABELED"
        rec.label_source = "unlabeled"
        label_counts["unlabeled"] += 1

    # Pass 2: Interactive mode for UNLABELED scans
    if interactive:
        unlabeled_scans = [r for r in records if r.label == "UNLABELED"]
        if unlabeled_scans:
            print(f"\n[?] Entering Interactive Labeling Mode for {len(unlabeled_scans)} unlabeled scans:")
            print("Commands: [g] genuine of top-1 user | [u] unknown (impostor) | [s] skip | [q] quit\n")
            
            # Ensure labels directory exists
            os.makedirs(os.path.dirname(os.path.abspath(labels_csv_path)), exist_ok=True)
            csv_exists = os.path.exists(labels_csv_path)

            with open(labels_csv_path, "a", newline="", encoding="utf-8") as f_csv:
                fieldnames = ["start_time", "end_time", "label", "username"]
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                if not csv_exists:
                    writer.writeheader()

                for i, r in enumerate(unlabeled_scans, 1):
                    ts_str = r.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                    print(
                        f"[{i}/{len(unlabeled_scans)}] {ts_str} | Top-1: {r.top1_user} ({r.top1:.4f}) | "
                        f"Top-2: {r.top2_user or 'none'} ({r.top2:.4f}) | Margin: {r.margin:.4f} | "
                        f"Server: {r.decision}"
                    )
                    try:
                        choice = input(f"  Label for scan (g={r.top1_user} / u / s / q): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[!] Exiting interactive mode.")
                        break

                    if choice == "q":
                        print("[*] Quitting interactive mode.")
                        break
                    elif choice == "g":
                        r.label = "GENUINE"
                        r.label_source = "manual"
                        r.labeled_user = r.top1_user
                        label_counts["manual_genuine"] += 1
                        label_counts["unlabeled"] -= 1
                        writer.writerow({
                            "start_time": r.timestamp.isoformat(),
                            "end_time": r.timestamp.isoformat(),
                            "label": "genuine",
                            "username": r.top1_user,
                        })
                        f_csv.flush()
                        print(f"  -> Labeled as GENUINE ({r.top1_user}) and saved.\n")
                    elif choice == "u":
                        r.label = "IMPOSTOR"
                        r.label_source = "manual"
                        r.labeled_user = None
                        label_counts["manual_impostor"] += 1
                        label_counts["unlabeled"] -= 1
                        writer.writerow({
                            "start_time": r.timestamp.isoformat(),
                            "end_time": r.timestamp.isoformat(),
                            "label": "unknown",
                            "username": "",
                        })
                        f_csv.flush()
                        print("  -> Labeled as UNKNOWN (IMPOSTOR) and saved.\n")
                    else:
                        print("  -> Skipped.\n")

    return label_counts


# ── The Mathematical Core (Step 3) ───────────────────────────────────────────

def build_evaluation_sets(records: List[ScanRecord]) -> Tuple[List[float], List[float], List[ScanRecord], List[ScanRecord], int]:
    """
    Extracts Impostor set I (top1 scores) and Genuine set G (score of correct user).
    Handles misidentified genuine users by pulling the labeled user's score from ranked_candidates.
    """
    impostor_scans = [r for r in records if r.label == "IMPOSTOR"]
    genuine_scans = [r for r in records if r.label == "GENUINE"]

    impostor_scores = [r.top1 for r in impostor_scans]

    genuine_scores = []
    missing_user_count = 0

    for r in genuine_scans:
        target_user = r.labeled_user or r.top1_user
        if r.top1_user == target_user:
            genuine_scores.append(r.top1)
        else:
            # Misidentified genuine scan: look up user's own score in ranked_candidates
            found_score = None
            for c in r.ranked_candidates:
                if str(c.get("username", "")).strip().lower() == target_user:
                    try:
                        found_score = float(c["score"])
                    except Exception:
                        pass
                    break
            if found_score is not None:
                genuine_scores.append(found_score)
            else:
                missing_user_count += 1

    return impostor_scores, genuine_scores, impostor_scans, genuine_scans, missing_user_count


def evaluate_threshold_grid(
    impostor_scores: List[float],
    genuine_scores: List[float],
    genuine_scans: List[ScanRecord],
) -> List[Dict[str, Any]]:
    """
    Evaluates threshold grid from 0.50 to 0.99 in 0.005 steps plus all unique observed scores.
    Uses >= boundary to strictly match the server's acceptance rule (score >= threshold).
    """
    n_i = len(impostor_scores)
    n_g = len(genuine_scores)

    # 1. Base grid from 0.500 to 0.990 in 0.005 steps
    grid_set = set(round(0.500 + i * 0.005, 4) for i in range(99))

    # 2. Add all unique observed scores within [0.50, 0.99]
    for s in impostor_scores + genuine_scores:
        if 0.50 <= s <= 0.99:
            grid_set.add(round(s, 4))

    sorted_thresholds = sorted(grid_set)
    results = []

    for t in sorted_thresholds:
        # FAR: count(I >= t) / N_I
        fa_count = sum(1 for s in impostor_scores if s >= t)
        far = fa_count / float(n_i) if n_i > 0 else 0.0

        # TAR: count(G >= t) / N_G
        ta_count = sum(1 for s in genuine_scores if s >= t)
        tar = ta_count / float(n_g) if n_g > 0 else 0.0
        frr = 1.0 - tar

        # Correct-identity accept rate: top1_user == labeled_user AND top1 >= t
        correct_id_accepts = 0
        if n_g > 0:
            for r in genuine_scans:
                target_user = r.labeled_user or r.top1_user
                if r.top1_user == target_user and r.top1 >= t:
                    correct_id_accepts += 1
            correct_id_rate = correct_id_accepts / float(n_g)
        else:
            correct_id_rate = 0.0

        ci_lower, ci_upper = clopper_pearson(fa_count, n_i, alpha=0.05) if n_i > 0 else (0.0, 0.0)

        results.append({
            "threshold": t,
            "far": far,
            "tar": tar,
            "frr": frr,
            "correct_id_rate": correct_id_rate,
            "fa_count": fa_count,
            "ta_count": ta_count,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return results


def recommend_thresholds(
    impostor_scores: List[float],
    genuine_scores: List[float],
    grid_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculates the 4 recommended thresholds per specification:
      A) ZERO_FALSE_ACCEPT (t_A): max(I) + 0.01 rounded up to 3 decimals, capped at 0.99
      B) TARGET_FAR: smallest t where FAR <= 1%, 5%, 0.1%
      C) EER: threshold minimizing |FAR - FRR| (tie-break: higher t)
      D) BALANCED_SAFE: smaller of t_A and TAR >= 70% threshold, but floor at max(I) + 0.005 if TAR >= 50%
    """
    n_i = len(impostor_scores)
    n_g = len(genuine_scores)

    recs: Dict[str, Any] = {}

    # A) ZERO_FALSE_ACCEPT
    if n_i > 0:
        max_i = max(impostor_scores)
        raw_t_a = max_i + 0.01
        # Round up to 3 decimals, capped at 0.99
        t_a = min(0.99, math.ceil(round(raw_t_a, 6) * 1000.0) / 1000.0)
        # Find metrics at t_a
        ta_count = sum(1 for s in genuine_scores if s >= t_a)
        tar_t_a = ta_count / float(n_g) if n_g > 0 else 0.0
        ci_t_a = clopper_pearson(0, n_i)
        recs["zero_false_accept"] = {
            "threshold": t_a,
            "tar": tar_t_a,
            "far": 0.0,
            "ci_lower": ci_t_a[0],
            "ci_upper": ci_t_a[1],
            "rule_of_three": rule_of_three(n_i),
        }
    else:
        recs["zero_false_accept"] = {
            "threshold": 0.75,
            "tar": 0.0,
            "far": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "rule_of_three": 1.0,
            "note": "No impostor scans available to calibrate zero false accept threshold.",
        }

    # B) TARGET_FAR (1%, 5%, 0.1%)
    target_fars = [0.05, 0.01, 0.001]
    recs["target_far"] = {}
    for target in target_fars:
        key = f"far_{int(target * 1000)}bp" if target < 0.01 else f"far_{int(target * 100)}pct"
        warning = None
        if target == 0.001 and n_i < 1000:
            warning = f"Warning: FAR 0.1% cannot be resolved with N_I={n_i} samples (minimum 1,000 required)."

        matching = [r for r in grid_results if r["far"] <= target]
        if matching:
            best_r = min(matching, key=lambda x: x["threshold"])
            recs["target_far"][key] = {
                "target": target,
                "threshold": best_r["threshold"],
                "far": best_r["far"],
                "tar": best_r["tar"],
                "ci_lower": best_r["ci_lower"],
                "ci_upper": best_r["ci_upper"],
                "warning": warning,
            }
        else:
            recs["target_far"][key] = {
                "target": target,
                "threshold": 0.99,
                "far": 0.0,
                "tar": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "warning": warning or "No threshold in grid achieved target FAR.",
            }

    # C) EER (Equal Error Rate)
    if grid_results:
        # Sort by |FAR - FRR| ascending, then threshold DESCENDING (tie-break rule: higher t)
        eer_candidate = min(grid_results, key=lambda x: (abs(x["far"] - x["frr"]), -x["threshold"]))
        recs["eer"] = {
            "threshold": eer_candidate["threshold"],
            "far": eer_candidate["far"],
            "frr": eer_candidate["frr"],
            "diff": abs(eer_candidate["far"] - eer_candidate["frr"]),
            "ci_lower": eer_candidate["ci_lower"],
            "ci_upper": eer_candidate["ci_upper"],
        }
    else:
        recs["eer"] = {"threshold": 0.75, "far": 0.0, "frr": 0.0, "diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    # D) BALANCED_SAFE
    # Rule: smaller of t_A and threshold keeping TAR >= 0.70; floor at max(I)+0.005 if that leaves TAR >= 0.50
    t_a_val = recs["zero_false_accept"]["threshold"]
    tar70_candidates = [r for r in grid_results if r["tar"] >= 0.70]
    if tar70_candidates:
        # Highest threshold where TAR >= 0.70
        t_tar70 = max(tar70_candidates, key=lambda x: x["threshold"])["threshold"]
    else:
        t_tar70 = 0.500

    binding_rule = "zero_false_accept" if t_a_val <= t_tar70 else "tar_70pct_retention"
    t_balanced = min(t_a_val, t_tar70)

    # Check floor condition: max(I) + 0.005
    if n_i > 0:
        floor_t = min(0.99, round(max(impostor_scores) + 0.005, 4))
        floor_tar = sum(1 for s in genuine_scores if s >= floor_t) / float(n_g) if n_g > 0 else 0.0
        if t_balanced < floor_t and floor_tar >= 0.50:
            t_balanced = floor_t
            binding_rule = "max_impostor_plus_0.005_floor"

    # Evaluate metrics at t_balanced
    fa_b = sum(1 for s in impostor_scores if s >= t_balanced)
    far_b = fa_b / float(n_i) if n_i > 0 else 0.0
    ta_b = sum(1 for s in genuine_scores if s >= t_balanced)
    tar_b = ta_b / float(n_g) if n_g > 0 else 0.0
    ci_b = clopper_pearson(fa_b, n_i) if n_i > 0 else (0.0, 0.0)

    separation_warning = None
    if n_i > 0 and n_g > 0 and recs["zero_false_accept"]["tar"] < 0.50:
        separation_warning = (
            f"Model cannot separate genuine and unknown palms well enough on this data: "
            f"TAR at zero false accept threshold ({t_a_val:.3f}) is only "
            f"{recs['zero_false_accept']['tar']*100:.1f}% (< 50%). Consider using the margin rule."
        )

    recs["balanced_safe"] = {
        "threshold": t_balanced,
        "far": far_b,
        "tar": tar_b,
        "frr": 1.0 - tar_b,
        "ci_lower": ci_b[0],
        "ci_upper": ci_b[1],
        "binding_rule": binding_rule,
        "separation_warning": separation_warning,
    }

    return recs


def analyze_margin_rule(
    genuine_scans: List[ScanRecord],
    impostor_scans: List[ScanRecord],
) -> Dict[str, Any]:
    """
    Evaluates combined rule: Accept if (top1 >= t AND margin >= m).
    Searches margin grid from 0.00 to 0.15 in 0.005 steps over threshold grid.
    Finds best (t, m) pair with FAR = 0 on impostor set and highest TAR.
    Only meaningful when 2+ users are enrolled (scans with top2 == -1.0 are skipped).
    """
    g_multi = [r for r in genuine_scans if r.top2 != -1.0]
    i_multi = [r for r in impostor_scans if r.top2 != -1.0]

    skipped_g = len(genuine_scans) - len(g_multi)
    skipped_i = len(impostor_scans) - len(i_multi)

    margin_stats: Dict[str, Any] = {
        "skipped_scans_single_user": skipped_g + skipped_i,
        "genuine_multi_count": len(g_multi),
        "impostor_multi_count": len(i_multi),
        "best_pair": None,
        "search_engine_supported": False,  # Checked against app/search_engine.py
    }

    if not g_multi or not i_multi:
        margin_stats["note"] = "Margin rule requires 2+ enrolled users in both genuine and impostor sets."
        return margin_stats

    g_margins = [r.margin for r in g_multi]
    i_margins = [r.margin for r in i_multi]

    margin_stats["genuine_margins"] = {
        "min": min(g_margins),
        "mean": mean(g_margins),
        "median": percentile_linear(g_margins, 50),
        "max": max(g_margins),
    }
    margin_stats["impostor_margins"] = {
        "min": min(i_margins),
        "mean": mean(i_margins),
        "median": percentile_linear(i_margins, 50),
        "max": max(i_margins),
    }

    best_t = 0.75
    best_m = 0.0
    best_tar = -1.0

    margin_candidates = [round(i * 0.005, 3) for i in range(31)]  # 0.000 to 0.150
    t_candidates = [round(0.50 + i * 0.01, 2) for i in range(50)]  # 0.50 to 0.99

    n_g_m = len(g_multi)
    n_i_m = len(i_multi)

    for m in margin_candidates:
        for t in t_candidates:
            # FAR on impostors
            fa = sum(1 for r in i_multi if r.top1 >= t and r.margin >= m)
            if fa == 0:
                # TAR on genuines: labeled user's score >= t AND margin >= m
                ta = 0
                for r in g_multi:
                    target_user = r.labeled_user or r.top1_user
                    user_score = r.top1
                    if r.top1_user != target_user:
                        for c in r.ranked_candidates:
                            if str(c.get("username", "")).strip().lower() == target_user:
                                user_score = float(c.get("score", -1.0))
                                break
                    if user_score >= t and r.margin >= m:
                        ta += 1
                tar = ta / float(n_g_m)
                if tar > best_tar:
                    best_tar = tar
                    best_t = t
                    best_m = m

    margin_stats["best_pair"] = {
        "threshold": best_t,
        "min_margin": best_m,
        "tar": best_tar if best_tar >= 0.0 else 0.0,
        "far": 0.0,
    }

    return margin_stats


def per_user_check(
    records: List[ScanRecord],
    db_users: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Per-user analysis:
      - Number of genuine scans
      - Min / mean genuine score
      - Highest impostor score that pointed to that user (stranger magnet check)
      - Flags users where impostor max is within 0.03 of genuine min
    """
    users_stats: Dict[str, Dict[str, Any]] = {}
    for uname in db_users:
        users_stats[uname] = {
            "genuine_count": 0,
            "genuine_scores": [],
            "impostor_pointing_scores": [],
            "magnet_flag": False,
        }

    for r in records:
        if r.label == "GENUINE":
            target = r.labeled_user or r.top1_user
            if target not in users_stats:
                users_stats[target] = {
                    "genuine_count": 0,
                    "genuine_scores": [],
                    "impostor_pointing_scores": [],
                    "magnet_flag": False,
                }
            # Look up score
            s = r.top1
            if r.top1_user != target:
                for c in r.ranked_candidates:
                    if str(c.get("username", "")).strip().lower() == target:
                        s = float(c.get("score", r.top1))
                        break
            users_stats[target]["genuine_count"] += 1
            users_stats[target]["genuine_scores"].append(s)

        elif r.label == "IMPOSTOR":
            target = r.top1_user
            if target:
                if target not in users_stats:
                    users_stats[target] = {
                        "genuine_count": 0,
                        "genuine_scores": [],
                        "impostor_pointing_scores": [],
                        "magnet_flag": False,
                    }
                users_stats[target]["impostor_pointing_scores"].append(r.top1)

    # Compute summary stats
    for uname, s in users_stats.items():
        g_scores = s["genuine_scores"]
        i_scores = s["impostor_pointing_scores"]

        g_min = min(g_scores) if g_scores else None
        g_mean = mean(g_scores) if g_scores else None
        i_max = max(i_scores) if i_scores else None

        s["genuine_min"] = g_min
        s["genuine_mean"] = g_mean
        s["impostor_max"] = i_max

        # Magnet check: impostor max is within 0.03 of genuine min
        if g_min is not None and i_max is not None:
            if (g_min - i_max) <= 0.03:
                s["magnet_flag"] = True

    return users_stats


# ── Report Generation & Output (Step 4) ──────────────────────────────────────

def format_console_report(
    clean_counts: Dict[str, int],
    label_counts: Dict[str, int],
    capture_failures: int,
    g_scores: List[float],
    i_scores: List[float],
    grid_results: List[Dict[str, Any]],
    recommendations: Dict[str, Any],
    margin_analysis: Dict[str, Any],
    user_analysis: Dict[str, Dict[str, Any]],
) -> str:
    """Formats full human-readable console report according to specifications."""
    lines = []
    lines.append("=" * 80)
    lines.append("          PALM VEIN BIOMETRIC EXPO THRESHOLD ANALYSIS REPORT          ")
    lines.append("=" * 80)

    # 1. DATA SUMMARY
    lines.append("\n[1] DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total JSONL log lines read:       {clean_counts['total_lines']}")
    lines.append(f"  Malformed / corrupted lines:      {clean_counts['malformed_lines']}")
    lines.append(f"  Outside time window:              {clean_counts['out_of_window']}")
    lines.append(f"  Non-scan operations excluded:     {clean_counts['non_scan_operations']}")
    lines.append(f"  Scans with no candidates:         {clean_counts['empty_candidates']}")
    lines.append(f"  Failed captures (capture log):    {capture_failures}")
    lines.append(f"  Valid matching scans kept:        {clean_counts['kept_scans']}")
    lines.append("  Scan Labels:")
    lines.append(f"    - Genuine (Manual):             {label_counts.get('manual_genuine', 0)}")
    lines.append(f"    - Genuine (Auto Post-Enroll):   {label_counts.get('auto_post_enrollment', 0)} (probable)")
    lines.append(f"    - Impostor (Manual):            {label_counts.get('manual_impostor', 0)}")
    lines.append(f"    - Impostor (Auto Pre-Enroll):   {label_counts.get('auto_before_enrollment', 0)} (certain)")
    lines.append(f"    - Unlabeled (Excluded from fit):{label_counts.get('unlabeled', 0)}")

    n_g = len(g_scores)
    n_i = len(i_scores)

    # Sample size warnings
    if n_i < 30 or n_g < 30:
        lines.append("\n  [!] STATISTICAL WARNING: Small sample size (N_G={n_g}, N_I={n_i}).".format(n_g=n_g, n_i=n_i))
        lines.append("      Error estimates (FAR/TAR) have wide confidence intervals when N < 30.")

    # 2. SCORE STATISTICS
    lines.append("\n[2] SCORE STATISTICS")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<18} | {'Genuine (G)':<25} | {'Impostor (I)':<25}")
    lines.append("-" * 80)
    lines.append(f"{'Count (N)':<18} | {n_g:<25} | {n_i:<25}")

    if n_g > 0:
        g_mean = f"{mean(g_scores):.4f}"
        g_std = f"{sample_std(g_scores):.4f}"
        g_min = f"{min(g_scores):.4f}"
        g_p05 = f"{percentile_linear(g_scores, 5):.4f}"
        g_med = f"{percentile_linear(g_scores, 50):.4f}"
        g_p95 = f"{percentile_linear(g_scores, 95):.4f}"
        g_max = f"{max(g_scores):.4f}"
    else:
        g_mean = g_std = g_min = g_p05 = g_med = g_p95 = g_max = "N/A"

    if n_i > 0:
        i_mean = f"{mean(i_scores):.4f}"
        i_std = f"{sample_std(i_scores):.4f}"
        i_min = f"{min(i_scores):.4f}"
        i_p05 = f"{percentile_linear(i_scores, 5):.4f}"
        i_med = f"{percentile_linear(i_scores, 50):.4f}"
        i_p95 = f"{percentile_linear(i_scores, 95):.4f}"
        i_max = f"{max(i_scores):.4f}"
    else:
        i_mean = i_std = i_min = i_p05 = i_med = i_p95 = i_max = "N/A"

    lines.append(f"{'Mean':<18} | {g_mean:<25} | {i_mean:<25}")
    lines.append(f"{'Std Dev (s)':<18} | {g_std:<25} | {i_std:<25}")
    lines.append(f"{'Min':<18} | {g_min:<25} | {i_min:<25}")
    lines.append(f"{'5th Percentile':<18} | {g_p05:<25} | {i_p05:<25}")
    lines.append(f"{'Median':<18} | {g_med:<25} | {i_med:<25}")
    lines.append(f"{'95th Percentile':<18} | {g_p95:<25} | {i_p95:<25}")
    lines.append(f"{'Max':<18} | {g_max:<25} | {i_max:<25}")

    if n_g > 0 and n_i > 0:
        gap = min(g_scores) - max(i_scores)
        gap_str = f"{gap:+.4f}"
        gap_desc = "(OVERLAP)" if gap < 0 else "(SEPARATED)"
        lines.append(f"{'Gap (min G - max I)':<18} | {gap_str + ' ' + gap_desc:<53}")
    lines.append("-" * 80)

    # 3. THRESHOLD PERFORMANCE TABLE
    lines.append("\n[3] THRESHOLD PERFORMANCE TABLE (0.05 step grid)")
    lines.append("-" * 80)
    lines.append(f"{'t':<6} | {'FAR %':<9} | {'TAR %':<9} | {'FRR %':<9} | {'Correct-ID %':<14} | {'95% CI (FAR)':<18}")
    lines.append("-" * 80)

    standard_points = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for sp in standard_points:
        # Match nearest in grid_results
        match = [r for r in grid_results if abs(r["threshold"] - sp) < 1e-4]
        if match:
            r = match[0]
            far_str = f"{r['far']*100:6.2f}%"
            tar_str = f"{r['tar']*100:6.2f}%"
            frr_str = f"{r['frr']*100:6.2f}%"
            cid_str = f"{r['correct_id_rate']*100:6.2f}%"
            ci_str = f"[{r['ci_lower']*100:4.1f}%, {r['ci_upper']*100:4.1f}%]"
            lines.append(f"{sp:<6.2f} | {far_str:<9} | {tar_str:<9} | {frr_str:<9} | {cid_str:<14} | {ci_str:<18}")
        else:
            lines.append(f"{sp:<6.2f} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<14} | {'N/A':<18}")
    lines.append("-" * 80)

    # 4. RECOMMENDED THRESHOLDS
    lines.append("\n[4] RECOMMENDED THRESHOLDS")
    lines.append("-" * 80)
    zfa = recommendations.get("zero_false_accept", {})
    t_zfa = zfa.get("threshold", 0.75)
    lines.append(f"  A) ZERO_FALSE_ACCEPT (t_A): {t_zfa:.3f}")
    lines.append(f"     -> Resulting TAR:       {zfa.get('tar', 0.0)*100:.1f}%")
    lines.append(f"     -> 95% Clopper-Pearson: [0.0%, {zfa.get('ci_upper', 0.0)*100:.2f}%]")
    if "rule_of_three" in zfa:
        lines.append(f"     -> Rule of Three bound: <= {zfa['rule_of_three']*100:.2f}%")

    tfar = recommendations.get("target_far", {})
    lines.append("\n  B) TARGET_FAR Thresholds:")
    for k, v in tfar.items():
        tgt_pct = v.get("target", 0.0) * 100
        t_val = v.get("threshold", 0.0)
        far_pct = v.get("far", 0.0) * 100
        tar_pct = v.get("tar", 0.0) * 100
        ci_str = f"[{v.get('ci_lower', 0.0)*100:.2f}%, {v.get('ci_upper', 0.0)*100:.2f}%]"
        lines.append(f"     -> FAR <= {tgt_pct:4.1f}%: t = {t_val:.3f} | FAR={far_pct:.2f}% | TAR={tar_pct:.1f}% | CI={ci_str}")
        if v.get("warning"):
            lines.append(f"        {v['warning']}")

    eer = recommendations.get("eer", {})
    lines.append(f"\n  C) EQUAL ERROR RATE (EER): t = {eer.get('threshold', 0.75):.3f}")
    lines.append(f"     -> FAR: {eer.get('far', 0.0)*100:.2f}% | FRR: {eer.get('frr', 0.0)*100:.2f}% (difference: {eer.get('diff', 0.0)*100:.2f}%)")

    bs = recommendations.get("balanced_safe", {})
    t_bs = bs.get("threshold", 0.75)
    lines.append(f"\n  D) BALANCED_SAFE (Recommended Default): t = {t_bs:.3f}")
    lines.append(f"     -> FAR: {bs.get('far', 0.0)*100:.2f}% (95% CI: [{bs.get('ci_lower', 0.0)*100:.2f}%, {bs.get('ci_upper', 0.0)*100:.2f}%])")
    lines.append(f"     -> TAR: {bs.get('tar', 0.0)*100:.1f}% (FRR: {bs.get('frr', 0.0)*100:.1f}%)")
    lines.append(f"     -> Binding rule: {bs.get('binding_rule')}")
    if bs.get("separation_warning"):
        lines.append(f"     -> {bs['separation_warning']}")

    # 5. MARGIN RULE ANALYSIS
    lines.append("\n[5] MARGIN RULE ANALYSIS (top1 - top2)")
    lines.append("-" * 80)
    if margin_analysis.get("skipped_scans_single_user"):
        lines.append(f"  Single-user enrolled scans skipped: {margin_analysis['skipped_scans_single_user']}")
    if margin_analysis.get("best_pair"):
        bp = margin_analysis["best_pair"]
        lines.append(f"  Best (Threshold, Min Margin) Pair with FAR = 0:")
        lines.append(f"     -> MATCH_THRESHOLD = {bp['threshold']:.3f}")
        lines.append(f"     -> MIN_MARGIN      = {bp['min_margin']:.3f}")
        lines.append(f"     -> Resulting TAR   = {bp['tar']*100:.1f}% (FAR = 0.0%)")
        if "genuine_margins" in margin_analysis:
            gm = margin_analysis["genuine_margins"]
            im = margin_analysis["impostor_margins"]
            lines.append(f"  Genuine margins:  min={gm['min']:.3f}, median={gm['median']:.3f}, max={gm['max']:.3f}")
            lines.append(f"  Impostor margins: min={im['min']:.3f}, median={im['median']:.3f}, max={im['max']:.3f}")
    if margin_analysis.get("note"):
        lines.append(f"  {margin_analysis['note']}")

    # 6. PER-USER ANALYSIS
    lines.append("\n[6] PER-USER SENSITIVITY & STRANGER MAGNET CHECK")
    lines.append("-" * 80)
    lines.append(f"{'Username':<16} | {'Genuine N':<10} | {'Gen Min':<9} | {'Gen Mean':<9} | {'Imp Max':<9} | {'Status'}")
    lines.append("-" * 80)
    for uname, us in sorted(user_analysis.items()):
        gn = us["genuine_count"]
        gmin_s = f"{us['genuine_min']:.3f}" if us["genuine_min"] is not None else "-"
        gmean_s = f"{us['genuine_mean']:.3f}" if us["genuine_mean"] is not None else "-"
        imax_s = f"{us['impostor_max']:.3f}" if us["impostor_max"] is not None else "-"
        status = "FLAG: MAGNET USER" if us["magnet_flag"] else "OK"
        lines.append(f"{uname:<16} | {gn:<10} | {gmin_s:<9} | {gmean_s:<9} | {imax_s:<9} | {status}")
    lines.append("-" * 80)

    # 7. WHAT TO DO SECTION
    lines.append("\n[7] WHAT TO DO")
    lines.append("=" * 80)
    lines.append("Exact commands to configure the kiosk:")
    lines.append(f"  export MATCH_THRESHOLD={t_bs:.3f}")

    has_margin = False
    if margin_analysis.get("best_pair") and margin_analysis["best_pair"]["min_margin"] > 0.0:
        bp = margin_analysis["best_pair"]
        lines.append(f"  export MIN_MARGIN={bp['min_margin']:.3f}")
        has_margin = True

    lines.append("")
    lines.append("Note on MIN_MARGIN:")
    lines.append("  MIN_MARGIN is not currently implemented in app/search_engine.py (decision rule uses")
    lines.append("  MATCH_THRESHOLD only). To enforce margin filtering, search_engine.py would need")
    lines.append("  to check (best_score >= MATCH_THRESHOLD and margin >= MIN_MARGIN).")

    lines.append("\nPlain-English Summary:")
    lines.append(
        f"  At MATCH_THRESHOLD={t_bs:.3f}, unknown/unenrolled palms were falsely accepted "
        f"{bs.get('far', 0.0)*100:.1f}% of the time in {n_i} tests; "
        f"genuine enrolled users were correctly accepted {bs.get('tar', 0.0)*100:.1f}% of the time in {n_g} tests."
    )
    lines.append("=" * 80)

    return "\n".join(lines)


# ── Optional Matplotlib Plotting ──────────────────────────────────────────────

def try_plot_distributions(
    g_scores: List[float],
    i_scores: List[float],
    recommendations: Dict[str, Any],
    plot_path: str,
):
    """Saves histogram if matplotlib is installed; skips silently if not."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        if i_scores:
            plt.hist(i_scores, bins=25, alpha=0.6, color="#FF4081", label=f"Impostor (N={len(i_scores)})", density=True)
        if g_scores:
            plt.hist(g_scores, bins=25, alpha=0.6, color="#00C853", label=f"Genuine (N={len(g_scores)})", density=True)

        t_zfa = recommendations.get("zero_false_accept", {}).get("threshold")
        if t_zfa:
            plt.axvline(t_zfa, color="#E65100", linestyle="--", linewidth=2, label=f"Zero FA (t={t_zfa:.3f})")

        t_bs = recommendations.get("balanced_safe", {}).get("threshold")
        if t_bs:
            plt.axvline(t_bs, color="#1E88E5", linestyle="-", linewidth=2.5, label=f"Balanced Safe (t={t_bs:.3f})")

        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Density")
        plt.title("Palm Vein Biometrics: Score Distributions (Genuine vs Impostor)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[+] Saved distribution plot: {plot_path}")
    except Exception:
        # Silently skip per prompt specification if matplotlib is not available or errors
        pass


# ── Main CLI Orchestrator ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline Threshold Analysis Tool for Palm Vein Biometric Kiosk."
    )
    parser.add_argument("--since", help="Filter scans since ISO timestamp (e.g. 2026-09-23T08:00:00Z)")
    parser.add_argument("--until", help="Filter scans until ISO timestamp (e.g. 2026-09-23T18:00:00Z)")
    parser.add_argument("--interactive", action="store_true", help="Interactively label UNLABELED scans")
    parser.add_argument("--post-enroll-minutes", type=int, default=10, help="Window after enrollment for auto genuine (default 10)")
    parser.add_argument("--labels", default="logs/expo_labels.csv", help="Path to manual labels CSV")
    parser.add_argument("--log", default="logs/scan_diagnostics.jsonl", help="Path to scan_diagnostics.jsonl")
    parser.add_argument("--db", default="data/palm_vein.db", help="Path to SQLite DB (opened read-only)")
    parser.add_argument("--capture-log", default="logs/capture_diagnostics.jsonl", help="Path to capture_diagnostics.jsonl")
    parser.add_argument("--plot", action="store_true", help="Generate distribution histogram plot (if matplotlib installed)")
    parser.add_argument("--output-dir", default="logs/analysis", help="Output directory for reports and CSVs")

    args = parser.parse_args()

    since_dt = parse_utc_datetime(args.since) if args.since else None
    until_dt = parse_utc_datetime(args.until) if args.until else None

    # Step 0: Load DB users & capture failures
    db_users = load_users_from_db(args.db)
    capture_failures = load_capture_failures(args.capture_log, since_dt, until_dt)

    # Step 1: Parse and clean scans
    records, clean_counts = parse_and_clean_scans(args.log, db_users, since_dt, until_dt)

    # Step 2: Apply labels
    manual_labels = load_manual_labels(args.labels)
    label_counts = apply_labels(
        records=records,
        manual_labels=manual_labels,
        db_users=db_users,
        post_enroll_minutes=args.post_enroll_minutes,
        interactive=args.interactive,
        labels_csv_path=args.labels,
    )

    # Step 3: Math
    impostor_scores, genuine_scores, impostor_scans, genuine_scans, missing_u_count = build_evaluation_sets(records)
    grid_results = evaluate_threshold_grid(impostor_scores, genuine_scores, genuine_scans)
    recommendations = recommend_thresholds(impostor_scores, genuine_scores, grid_results)
    margin_analysis = analyze_margin_rule(genuine_scans, impostor_scans)
    user_analysis = per_user_check(records, db_users)

    # Step 4: Output
    console_report = format_console_report(
        clean_counts=clean_counts,
        label_counts=label_counts,
        capture_failures=capture_failures,
        g_scores=genuine_scores,
        i_scores=impostor_scores,
        grid_results=grid_results,
        recommendations=recommendations,
        margin_analysis=margin_analysis,
        user_analysis=user_analysis,
    )
    print(console_report)

    # Save output artifacts in output-dir
    os.makedirs(args.output_dir, exist_ok=True)
    utc_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(args.output_dir, f"threshold_report_{utc_str}.json")
    csv_path = os.path.join(args.output_dir, f"scores_labeled_{utc_str}.csv")
    plot_path = os.path.join(args.output_dir, f"threshold_plot_{utc_str}.png")

    # Save JSON report
    report_dict = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_log": args.log,
        "input_db": args.db,
        "counts": clean_counts,
        "labels": label_counts,
        "capture_failures_excluded": capture_failures,
        "missing_genuine_candidates": missing_u_count,
        "recommendations": recommendations,
        "margin_analysis": margin_analysis,
        "user_analysis": user_analysis,
        "threshold_grid_sample": [
            r for r in grid_results if abs(r["threshold"] * 100 % 5) < 1e-4
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save labeled scans CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "decision", "score", "top1_user", "top1",
            "top2_user", "top2", "margin", "label", "label_source", "labeled_user"
        ])
        for r in records:
            writer.writerow([
                r.timestamp_raw, r.decision, r.score, r.top1_user, r.top1,
                r.top2_user, r.top2, r.margin, r.label, r.label_source, r.labeled_user or ""
            ])

    print(f"\n[+] Saved detailed JSON report: {json_path}")
    print(f"[+] Saved labeled scans CSV:    {csv_path}")

    # Optional plot
    if args.plot:
        try_plot_distributions(genuine_scores, impostor_scores, recommendations, plot_path)


if __name__ == "__main__":
    main()
