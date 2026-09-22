#!/usr/bin/env python3
"""
real_data_analysis.py
---------------------
Offline real-data mining and biometric diagnostics for the Palm Vein Biometric System.
v2 Architecture: Evaluates 512-dim CNN embeddings using cosine similarity.

Mines real ground truth data from:
1. data/palm_vein.db (stored enrollment templates with 512-dim float32 embeddings).
2. roi_clahe/ (real captured scan-time CLAHE ROIs).
3. captures/ and access_log table (session timestamp cross-referencing).

Performs:
- Full all-pairs cosine similarity comparisons between enrolled templates.
- Scan-time probe feature extraction and full identification simulation.
- Genuine vs Impostor cosine similarity distributions (min, mean, max, std).
- Precise identification of False Accept and False Reject cases.
- Empirical threshold sweep table (FAR / FRR at 0.3, 0.4, 0.5, 0.6, 0.7, 0.8).
"""

import os
import sys
import glob
import time
import re
import json
import sqlite3
import argparse
from datetime import datetime, timezone
import numpy as np
import cv2

# Project root resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from app.constants import (
        MATCH_THRESHOLD, DB_PATH, ROI_DIR, CAPTURE_DIR, LOGS_DIR, EMBEDDING_DIM,
    )
    from app.cnn_extractor import extract_embedding, cosine_similarity, MODEL_LOADED
except ImportError:
    from constants import (
        MATCH_THRESHOLD, DB_PATH, ROI_DIR, CAPTURE_DIR, LOGS_DIR, EMBEDDING_DIM,
    )
    from cnn_extractor import extract_embedding, cosine_similarity, MODEL_LOADED


# ---------------------------------------------------------------------------
# Database Template Loading
# ---------------------------------------------------------------------------

def load_enrolled_templates(db_path: str) -> tuple:
    """
    Loads all enrolled templates and users from the SQLite database.
    Returns:
        users: {user_id: {'username': str, 'active': int, 'enrolled_at': str}}
        templates: list of dicts with keys:
            'id', 'user_id', 'username', 'sample_idx', 'embedding'
    """
    if not os.path.exists(db_path):
        print(f"[!] Database file not found at: {db_path}")
        return {}, []

    users = {}
    templates = []

    with sqlite3.connect(db_path) as conn:
        # Load users
        cur = conn.execute("SELECT id, username, active, enrolled_at FROM users")
        for uid, uname, active, enrolled_at in cur.fetchall():
            users[uid] = {
                'username': uname,
                'active': active,
                'enrolled_at': enrolled_at,
            }

        # Check column names in templates table
        cur = conn.execute("PRAGMA table_info(templates)")
        cols = [c[1] for c in cur.fetchall()]
        if 'embedding' not in cols:
            print("[!] Legacy templates table detected without 'embedding' column. Run init_db() to migrate.")
            return users, []

        cur = conn.execute(
            """
            SELECT t.id, t.user_id, t.sample_idx, t.embedding, u.username
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            ORDER  BY t.user_id, t.sample_idx
            """
        )

        for tid, uid, s_idx, emb_blob, uname in cur.fetchall():
            try:
                emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
                templates.append({
                    'id': tid,
                    'user_id': uid,
                    'username': uname,
                    'sample_idx': s_idx,
                    'embedding': emb,
                })
            except Exception as e:
                print(f"[!] Warning: Failed unpacking template id={tid}: {e}")

    return users, templates


# ---------------------------------------------------------------------------
# All-Pairs Enrolled Template Evaluation (Cosine Similarity)
# ---------------------------------------------------------------------------

def evaluate_all_pairs_templates(templates: list) -> dict:
    """
    Computes all-pairs cosine similarity between every enrolled template.
    Genuine pairs = same user_id. Impostor pairs = different user_id.
    """
    genuine_scores = []
    impostor_scores = []
    comparisons = []

    n = len(templates)
    print(f"[*] Computing all-pairs comparisons for {n} enrolled templates ({n * (n - 1) // 2} pairs)...")

    for i in range(n):
        for j in range(i + 1, n):
            t1 = templates[i]
            t2 = templates[j]

            sim = cosine_similarity(t1['embedding'], t2['embedding'])
            is_genuine = (t1['user_id'] == t2['user_id'])

            record = {
                't1_id': t1['id'],
                't2_id': t2['id'],
                'u1': t1['username'],
                'u2': t2['username'],
                'is_genuine': is_genuine,
                'score': sim,
            }
            comparisons.append(record)

            if is_genuine:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    return {
        'comparisons': comparisons,
        'genuine_scores': np.array(genuine_scores, dtype=np.float32),
        'impostor_scores': np.array(impostor_scores, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Ground-Truth Resolution for Scan ROIs
# ---------------------------------------------------------------------------

def parse_scan_timestamp(filename: str):
    """Extract timestamp string or datetime from standard filename."""
    m = re.search(r'(\d{8}_\d{6})', filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None


def resolve_ground_truth(filename: str, enrolled_usernames: list, access_logs: list) -> str:
    """Resolves ground truth identity for a scan ROI file."""
    basename = os.path.basename(filename)

    for uname in enrolled_usernames:
        if basename.startswith(uname + "_") or basename.startswith(uname + "-"):
            return uname

    lower_base = basename.lower()
    for name in ['yesh', 'nassir']:
        if name in lower_base and not lower_base.startswith("unknown"):
            for u in enrolled_usernames:
                if name in u.lower():
                    return u

    file_dt = parse_scan_timestamp(basename)
    if file_dt and access_logs:
        closest_log = None
        min_delta = float('inf')
        for log in access_logs:
            delta = abs((file_dt - log['dt']).total_seconds())
            if delta < min_delta and delta <= 90:
                min_delta = delta
                closest_log = log

        if closest_log and closest_log['username']:
            return closest_log['username']

    return "unknown"


def load_access_logs(db_path: str) -> list:
    """Load access logs with timestamps and usernames for session correlation."""
    if not os.path.exists(db_path):
        return []
    logs = []
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                SELECT a.scan_at, u.username, a.score, a.accepted
                FROM   access_log a
                LEFT JOIN users u ON u.id = a.user_id
                ORDER  BY a.id
                """
            )
            for ts_str, uname, score, accepted in cur.fetchall():
                try:
                    dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                    logs.append({'dt': dt, 'username': uname, 'score': score, 'accepted': accepted})
                except Exception:
                    pass
    except Exception as e:
        print(f"[!] Warning: Could not read access_log: {e}")
    return logs


# ---------------------------------------------------------------------------
# Scan-Time ROI Mining & Identification Pipeline Simulation
# ---------------------------------------------------------------------------

def mine_scan_rois(roi_dir: str, templates: list, users: dict, db_path: str) -> dict:
    """
    Extracts CNN embeddings for saved scan ROIs and evaluates cosine matching.
    """
    scan_files = glob.glob(os.path.join(roi_dir, "*scan*.png"))
    if not scan_files:
        scan_files = [f for f in glob.glob(os.path.join(roi_dir, "*.png")) if "_enroll_" not in f]

    print(f"[*] Found {len(scan_files)} scan ROI files in {roi_dir}")
    if not scan_files or not templates:
        return {'scan_results': [], 'genuine_scores': np.array([]), 'impostor_scores': np.array([])}

    if not MODEL_LOADED:
        print("[!] Warning: CNN ONNX model not loaded. Skipping scan ROI embedding extraction.")
        return {'scan_results': [], 'genuine_scores': np.array([]), 'impostor_scores': np.array([])}

    enrolled_usernames = [u['username'] for u in users.values() if u['active']]
    access_logs = load_access_logs(db_path)

    scan_results = []
    genuine_scores = []
    impostor_scores = []

    for idx, fpath in enumerate(scan_files):
        fname = os.path.basename(fpath)
        ground_truth = resolve_ground_truth(fname, enrolled_usernames, access_logs)

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224))

        try:
            probe_emb = extract_embedding(img)
        except Exception as e:
            print(f"[!] Error extracting embedding from {fname}: {e}")
            continue

        evaluations = []
        user_scores = {}

        for t in templates:
            score = cosine_similarity(t['embedding'], probe_emb)
            uname = t['username']

            evaluations.append({
                'template_id': t['id'],
                'username': uname,
                'sample_idx': t['sample_idx'],
                'score': score,
            })

            # User aggregation: BEST (MAX) similarity score across templates
            if uname not in user_scores or score > user_scores[uname]:
                user_scores[uname] = score

        ranked_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        winner_uname, winner_score = ranked_users[0]

        gt_in_enrolled = (ground_truth in user_scores)
        gt_score = user_scores.get(ground_truth, None)

        record = {
            'filename': fname,
            'ground_truth': ground_truth,
            'gt_in_enrolled': gt_in_enrolled,
            'winner_user': winner_uname,
            'winner_score': winner_score,
            'gt_score': gt_score,
            'ranked_users': ranked_users,
            'evaluations': evaluations,
        }
        scan_results.append(record)

        for uname, score in user_scores.items():
            if gt_in_enrolled and uname == ground_truth:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return {
        'scan_results': scan_results,
        'genuine_scores': np.array(genuine_scores, dtype=np.float32),
        'impostor_scores': np.array(impostor_scores, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Statistics, Flags & Threshold Sweep Reporting
# ---------------------------------------------------------------------------

def calculate_stats(scores: np.ndarray) -> dict:
    """Calculates summary statistics for a score array."""
    if len(scores) == 0:
        return {'count': 0, 'min': 0.0, 'mean': 0.0, 'max': 0.0, 'std': 0.0}
    return {
        'count': int(len(scores)),
        'min': float(np.min(scores)),
        'mean': float(np.mean(scores)),
        'max': float(np.max(scores)),
        'std': float(np.std(scores)),
    }


def print_score_distribution(title: str, genuine: np.ndarray, impostor: np.ndarray):
    """Prints a formatted score distribution table (Cosine Similarity: Higher is Better)."""
    g_stats = calculate_stats(genuine)
    i_stats = calculate_stats(impostor)

    print(f"\n{'=' * 65}")
    print(f" {title.upper()} (COSINE SIMILARITY: HIGHER = BETTER)")
    print(f"{'=' * 65}")
    print(f"{'Metric':<18} | {'Genuine (Same Palm)':<20} | {'Impostor (Different Palm)':<20}")
    print(f"{'-' * 18}-+-{'-' * 20}-+-{'-' * 20}")
    print(f"{'Samples Count':<18} | {g_stats['count']:<20} | {i_stats['count']:<20}")
    print(f"{'Minimum Score':<18} | {g_stats['min']:<20.4f} | {i_stats['min']:<20.4f}")
    print(f"{'Mean Score':<18} | {g_stats['mean']:<20.4f} | {i_stats['mean']:<20.4f}")
    print(f"{'Maximum Score':<18} | {g_stats['max']:<20.4f} | {i_stats['max']:<20.4f}")
    print(f"{'Std Deviation':<18} | {g_stats['std']:<20.4f} | {i_stats['std']:<20.4f}")

    if g_stats['count'] > 0 and i_stats['count'] > 0:
        gap = g_stats['min'] - i_stats['max']
        gap_str = f"{gap:+.4f}"
        status = "PERFECT SEPARATION (NO OVERLAP)" if gap > 0 else "OVERLAP DETECTED (TUNING REQUIRED)"
        print(f"\nSeparation Gap (Genuine Min - Impostor Max): {gap_str}  -->  {status}")


def flag_errors_and_near_misses(scan_results: list, threshold: float):
    """
    Specifically flags False Accepts and False Rejects.
    Note: Under cosine similarity, score >= threshold indicates ACCEPT.
    """
    print(f"\n{'=' * 75}")
    print(f" ERROR AUDIT & NEAR-MISS ANALYSIS (MATCH_THRESHOLD = {threshold:.4f})")
    print(f"{'=' * 75}")

    false_accepts = []
    false_rejects = []

    for res in scan_results:
        fname = res['filename']
        gt = res['ground_truth']
        winner = res['winner_user']
        score = res['winner_score']
        gt_in_enrolled = res['gt_in_enrolled']
        accepted = (score >= threshold)

        if accepted:
            if winner != gt:
                false_accepts.append(res)
        else:
            if gt_in_enrolled:
                false_rejects.append(res)

    print(f"Total Scans Audited: {len(scan_results)}")
    print(f"False Accepts (FA) : {len(false_accepts)}")
    print(f"False Rejects (FR) : {len(false_rejects)}")

    if false_accepts:
        print(f"\n[!] FALSE ACCEPT DETAILS ({len(false_accepts)} cases):")
        print(f"{'Scan Filename':<26} | {'Ground Truth':<15} | {'Accepted As':<15} | {'Score':<8} | {'GT Score':<8}")
        print(f"{'-' * 26}-+-{'-' * 15}-+-{'-' * 15}-+-{'-' * 8}-+-{'-' * 8}")
        for fa in false_accepts:
            gt_s_str = f"{fa['gt_score']:.4f}" if fa['gt_score'] is not None else "N/A"
            print(f"{fa['filename']:<26} | {fa['ground_truth']:<15} | {fa['winner_user']:<15} | {fa['winner_score']:<8.4f} | {gt_s_str:<8}")
    else:
        print("\n[+] Zero False Accepts detected at current threshold.")

    if false_rejects:
        print(f"\n[!] FALSE REJECT DETAILS ({len(false_rejects)} cases):")
        print(f"{'Scan Filename':<28} | {'Ground Truth':<16} | {'Score':<8} | {'Threshold':<9}")
        print(f"{'-' * 28}-+-{'-' * 16}-+-{'-' * 8}-+-{'-' * 9}")
        for fr in false_rejects:
            print(f"{fr['filename']:<28} | {fr['ground_truth']:<16} | {fr['winner_score']:<8.4f} | {threshold:<9.4f}")
    else:
        print("[+] Zero False Rejects detected at current threshold.")


def compute_threshold_sweep(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                            thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """
    Computes and prints the FAR/FRR threshold sweep table on real cosine similarity data.
    Decision: score >= tau ACCEPTS, score < tau REJECTS.
    """
    print(f"\n{'=' * 75}")
    print(f" REAL-DATA THRESHOLD SWEEP TABLE (FAR / FRR)")
    print(f"{'=' * 75}")
    print(f"{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Decisions':<10} | {'False Accepts':<14} | {'False Rejects'}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 14}")

    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    total_decisions = n_gen + n_imp

    if total_decisions == 0:
        print("No evaluation scores available to compute sweep.")
        return

    best_diff = float('inf')
    eer_point = None

    for tau in thresholds:
        # Impostor accepted if similarity >= tau
        fa_count = int(np.sum(impostor_scores >= tau)) if n_imp > 0 else 0
        # Genuine rejected if similarity < tau
        fr_count = int(np.sum(genuine_scores < tau)) if n_gen > 0 else 0

        far = (fa_count / n_imp * 100.0) if n_imp > 0 else 0.0
        frr = (fr_count / n_gen * 100.0) if n_gen > 0 else 0.0

        print(f"{tau:<10.4f} | {far:<10.2f} | {frr:<10.2f} | {total_decisions:<10} | {fa_count:<14} | {fr_count:<14}")

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_point = (tau, (far + frr) / 2.0)

    if eer_point:
        print(f"\nEmpirical EER Estimate: ~{eer_point[1]:.2f}% near threshold {eer_point[0]:.4f}")
        print("NOTE: This script reports metrics for review. Update MATCH_THRESHOLD in constants.py manually.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mine real captured palm vein templates & ROI scans for empirical accuracy (v2 CNN embeddings)."
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})"
    )
    parser.add_argument(
        "--roi-dir", default=ROI_DIR,
        help=f"Path to CLAHE ROI directory (default: {ROI_DIR})"
    )
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Baseline match threshold (default: {MATCH_THRESHOLD})"
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Optional path to export full analysis report as JSON."
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" PALM VEIN BIOMETRIC REAL-DATA DIAGNOSTICS & ACCURACY AUDIT (v2 CNN)")
    print("=" * 75)
    print(f"Database   : {args.db}")
    print(f"ROI Dir    : {args.roi_dir}")
    print(f"Threshold  : {args.threshold:.4f}")

    # 1. Load Enrolled Templates
    users, templates = load_enrolled_templates(args.db)
    print(f"\n[+] Loaded {len(users)} users ({sum(1 for u in users.values() if u['active'])} active) "
          f"and {len(templates)} enrolled templates.")

    for uid, uinfo in users.items():
        u_templates = [t for t in templates if t['user_id'] == uid]
        status = "ACTIVE" if uinfo['active'] else "INACTIVE"
        print(f"    - User '{uinfo['username']}' (ID {uid}, {status}): {len(u_templates)} template(s)")

    if not templates:
        print("\n[*] No enrolled templates found in database (or templates table empty). Exiting analysis.")
        return

    # 2. All-Pairs Template Analysis
    template_results = evaluate_all_pairs_templates(templates)
    print_score_distribution(
        "Enrolled Templates All-Pairs Comparison (Cross-Match / Self-Match)",
        template_results['genuine_scores'],
        template_results['impostor_scores']
    )

    # 3. Mine Scan ROIs
    scan_analysis = mine_scan_rois(args.roi_dir, templates, users, args.db)

    if len(scan_analysis['scan_results']) > 0:
        print_score_distribution(
            "Scan-Time Probe vs Enrolled Templates Comparison",
            scan_analysis['genuine_scores'],
            scan_analysis['impostor_scores']
        )
        flag_errors_and_near_misses(scan_analysis['scan_results'], args.threshold)

        comb_genuine = np.concatenate([template_results['genuine_scores'], scan_analysis['genuine_scores']])
        comb_impostor = np.concatenate([template_results['impostor_scores'], scan_analysis['impostor_scores']])
        compute_threshold_sweep(comb_genuine, comb_impostor)
    else:
        print("\n[*] No scan ROI images evaluated; running sweep on enrolled template pairs:")
        compute_threshold_sweep(template_results['genuine_scores'], template_results['impostor_scores'])

    # 4. Optional JSON export
    if args.output_json:
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_users': len(users),
            'total_templates': len(templates),
            'template_pairs_count': len(template_results['comparisons']),
            'scan_images_count': len(scan_analysis['scan_results']),
            'threshold_analyzed': args.threshold,
        }
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Exported summary report to {args.output_json}")


if __name__ == "__main__":
    main()
