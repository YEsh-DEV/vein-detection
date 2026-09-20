#!/usr/bin/env python3
"""
real_data_analysis.py
---------------------
Offline real-data mining and biometric diagnostics for the Palm Vein Biometric System.

Mines real ground truth data from:
1. data/palm_vein.db (stored enrollment templates, VR, VI, and 64-float signatures).
2. roi_clahe/ (real captured scan-time CLAHE ROIs).
3. captures/ and access_log table (session timestamp cross-referencing).

Performs:
- Full all-pairs template comparisons across Layer 1 (64-float Euclidean signature distance)
  and Layer 2 (multi-rotation MNHD with angle bracket).
- Scan-time probe feature extraction and full identification simulation.
- Genuine vs Impostor score distributions (min, mean, max, std).
- Precise identification of False Accept and False Reject cases with per-layer score breakdowns.
- Empirical threshold sweep table (FAR / FRR at 0.30, 0.35, 0.365, 0.38, 0.40, 0.45).
"""

import os
import sys
import glob
import time
import zlib
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
        MATCH_THRESHOLD, L1_THRESHOLD, DB_PATH, ROI_DIR, CAPTURE_DIR,
        LOGS_DIR, TOP_K, L1_BYPASS_MAX_TEMPLATES,
    )
    from app.gabor import match_templates, extract_veincode
    from app.db_manager import compute_signature
except ImportError:
    from constants import (
        MATCH_THRESHOLD, L1_THRESHOLD, DB_PATH, ROI_DIR, CAPTURE_DIR,
        LOGS_DIR, TOP_K, L1_BYPASS_MAX_TEMPLATES,
    )
    from gabor import match_templates, extract_veincode
    from db_manager import compute_signature


# ---------------------------------------------------------------------------
# Database Template Loading
# ---------------------------------------------------------------------------

def load_enrolled_templates(db_path: str) -> tuple:
    """
    Loads all enrolled templates and users from the SQLite database.
    Returns:
        users: {user_id: {'username': str, 'active': int, 'enrolled_at': str}}
        templates: list of dicts with keys:
            'id', 'user_id', 'username', 'sample_idx', 'VR', 'VI', 'signature', 'vr_mean', 'vi_mean'
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

        # Load templates for active users (or all if specified)
        cur = conn.execute(
            """
            SELECT t.id, t.user_id, t.sample_idx, t.vr_blob, t.vi_blob,
                   t.signature, t.vr_mean, t.vi_mean, u.username
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            ORDER  BY t.user_id, t.sample_idx
            """
        )

        for tid, uid, s_idx, vr_blob, vi_blob, sig_blob, vr_mean, vi_mean, uname in cur.fetchall():
            try:
                VR = np.frombuffer(zlib.decompress(vr_blob), dtype=np.uint8).reshape(256, 256)
                VI = np.frombuffer(zlib.decompress(vi_blob), dtype=np.uint8).reshape(256, 256)
                sig = np.frombuffer(sig_blob, dtype=np.float32)
                templates.append({
                    'id': tid,
                    'user_id': uid,
                    'username': uname,
                    'sample_idx': s_idx,
                    'VR': VR,
                    'VI': VI,
                    'signature': sig,
                    'vr_mean': vr_mean,
                    'vi_mean': vi_mean,
                })
            except Exception as e:
                print(f"[!] Warning: Failed unpacking template id={tid}: {e}")

    return users, templates


# ---------------------------------------------------------------------------
# All-Pairs Enrolled Template Evaluation
# ---------------------------------------------------------------------------

def evaluate_all_pairs_templates(templates: list) -> dict:
    """
    Re-runs the full pipeline (Layer 1 signature filter + Layer 2 MNHD)
    as all-pairs comparison between every enrolled template.
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

            # Layer 1: Euclidean distance on 64-float signatures
            l1_dist = float(np.linalg.norm(t1['signature'] - t2['signature']))
            l1_filtered = (l1_dist >= L1_THRESHOLD)

            # Layer 2: MNHD with angle bracket
            l2_score = float(match_templates(t1, t2))

            is_genuine = (t1['user_id'] == t2['user_id'])

            record = {
                't1_id': t1['id'],
                't2_id': t2['id'],
                'u1': t1['username'],
                'u2': t2['username'],
                'is_genuine': is_genuine,
                'l1_dist': l1_dist,
                'l1_filtered': l1_filtered,
                'l2_score': l2_score,
            }
            comparisons.append(record)

            if is_genuine:
                genuine_scores.append(l2_score)
            else:
                impostor_scores.append(l2_score)

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
    # Pattern: ..._YYYYMMDD_HHMMSS...
    m = re.search(r'(\d{8}_\d{6})', filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None


def resolve_ground_truth(filename: str, enrolled_usernames: list, access_logs: list) -> str:
    """
    Resolves the ground truth identity for a scan ROI file:
    1. Checks filename prefix for exact match with enrolled usernames.
    2. For 'unknown_scan' files, correlates timestamp with nearby access_log or labeled scans.
    """
    basename = os.path.basename(filename)

    # Direct username prefix match
    for uname in enrolled_usernames:
        if basename.startswith(uname + "_") or basename.startswith(uname + "-"):
            return uname

    # Check for known names like yesh, nassir
    lower_base = basename.lower()
    for name in ['yesh', 'nassir']:
        if name in lower_base and not lower_base.startswith("unknown"):
            # Try to match to full username in enrolled_usernames
            for u in enrolled_usernames:
                if name in u.lower():
                    return u

    # Ambiguous or unknown_scan: cross-reference with access_log by timestamp
    file_dt = parse_scan_timestamp(basename)
    if file_dt and access_logs:
        closest_log = None
        min_delta = float('inf')
        for log in access_logs:
            delta = abs((file_dt - log['dt']).total_seconds())
            if delta < min_delta and delta <= 90:  # within 90-second session window
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
                SELECT a.timestamp, u.username, a.score, a.accepted
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
    Extracts VeinCodes for saved scan ROIs, computes full candidate rankings,
    and checks Layer 1 filter behavior and Layer 2 matching.
    """
    scan_files = glob.glob(os.path.join(roi_dir, "*scan*.png"))
    if not scan_files:
        # Fallback to any png in roi_dir that isn't clearly enroll
        scan_files = [f for f in glob.glob(os.path.join(roi_dir, "*.png")) if "_enroll_" not in f]

    print(f"[*] Found {len(scan_files)} scan ROI files in {roi_dir}")
    if not scan_files or not templates:
        return {'scan_results': [], 'genuine_scores': np.array([]), 'impostor_scores': np.array([])}

    enrolled_usernames = [u['username'] for u in users.values() if u['active']]
    access_logs = load_access_logs(db_path)

    # Pre-index template signatures
    template_sigs = np.stack([t['signature'] for t in templates], axis=0)

    scan_results = []
    genuine_scores = []
    impostor_scores = []

    for idx, fpath in enumerate(scan_files):
        fname = os.path.basename(fpath)
        ground_truth = resolve_ground_truth(fname, enrolled_usernames, access_logs)

        # Read CLAHE ROI image
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256))

        try:
            probe_code = extract_veincode(img)
            probe_sig = compute_signature(probe_code['VR'], probe_code.get('VI'))
        except Exception as e:
            print(f"[!] Error extracting VeinCode from {fname}: {e}")
            continue

        # Layer 1: compute Euclidean distance to all templates
        l1_dists = np.linalg.norm(template_sigs - probe_sig, axis=1)

        # Layer 2: compute MNHD against all templates
        evaluations = []
        user_scores = {}
        user_l1_dists = {}

        for t_idx, t in enumerate(templates):
            dist_l1 = float(l1_dists[t_idx])
            score_l2 = float(match_templates(t, probe_code))
            uname = t['username']

            evaluations.append({
                'template_id': t['id'],
                'username': uname,
                'sample_idx': t['sample_idx'],
                'l1_dist': dist_l1,
                'l1_filtered': (dist_l1 >= L1_THRESHOLD),
                'l2_score': score_l2,
            })

            # User aggregation: best (minimum) L2 score across poses
            if uname not in user_scores or score_l2 < user_scores[uname]:
                user_scores[uname] = score_l2
            if uname not in user_l1_dists or dist_l1 < user_l1_dists[uname]:
                user_l1_dists[uname] = dist_l1

        # Winning candidate
        ranked_users = sorted(user_scores.items(), key=lambda x: x[1])
        winner_uname, winner_score = ranked_users[0]
        winner_l1 = user_l1_dists[winner_uname]

        # Check ground-truth specifics
        gt_in_enrolled = (ground_truth in user_scores)
        gt_score = user_scores.get(ground_truth, None)
        gt_l1 = user_l1_dists.get(ground_truth, None)

        # Check if Layer 1 would have filtered out the genuine user
        # (i.e. ALL templates of ground-truth had l1_dist >= L1_THRESHOLD)
        gt_l1_filtered_out = False
        if gt_in_enrolled:
            gt_templates_l1 = [e['l1_dist'] for e in evaluations if e['username'] == ground_truth]
            gt_l1_filtered_out = all(d >= L1_THRESHOLD for d in gt_templates_l1)

        record = {
            'filename': fname,
            'ground_truth': ground_truth,
            'gt_in_enrolled': gt_in_enrolled,
            'winner_user': winner_uname,
            'winner_score': winner_score,
            'winner_l1': winner_l1,
            'gt_score': gt_score,
            'gt_l1': gt_l1,
            'gt_l1_filtered_out': gt_l1_filtered_out,
            'ranked_users': ranked_users,
            'evaluations': evaluations,
        }
        scan_results.append(record)

        # Record scores into distribution
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
    """Prints a formatted score distribution table."""
    g_stats = calculate_stats(genuine)
    i_stats = calculate_stats(impostor)

    print(f"\n{'=' * 65}")
    print(f" {title.upper()}")
    print(f"{'=' * 65}")
    print(f"{'Metric':<18} | {'Genuine (Same Palm)':<20} | {'Impostor (Different Palm)':<20}")
    print(f"{'-' * 18}-+-{'-' * 20}-+-{'-' * 20}")
    print(f"{'Samples Count':<18} | {g_stats['count']:<20} | {i_stats['count']:<20}")
    print(f"{'Minimum Score':<18} | {g_stats['min']:<20.4f} | {i_stats['min']:<20.4f}")
    print(f"{'Mean Score':<18} | {g_stats['mean']:<20.4f} | {i_stats['mean']:<20.4f}")
    print(f"{'Maximum Score':<18} | {g_stats['max']:<20.4f} | {i_stats['max']:<20.4f}")
    print(f"{'Std Deviation':<18} | {g_stats['std']:<20.4f} | {i_stats['std']:<20.4f}")

    if g_stats['count'] > 0 and i_stats['count'] > 0:
        gap = i_stats['min'] - g_stats['max']
        gap_str = f"{gap:+.4f}"
        status = "PERFECT SEPARATION (NO OVERLAP)" if gap > 0 else "OVERLAP DETECTED (TUNING REQUIRED)"
        print(f"\nSeparation Gap (Impostor Min - Genuine Max): {gap_str}  -->  {status}")


def flag_errors_and_near_misses(scan_results: list, threshold: float):
    """
    Specifically flags every case where:
    1. Scan matched the WRONG identity (False Accept).
    2. Scan failed to match the RIGHT identity (False Reject).
    3. Layer 1 excluded the correct candidate before Layer 2 had a chance.
    """
    print(f"\n{'=' * 75}")
    print(f" ERROR AUDIT & NEAR-MISS ANALYSIS (MATCH_THRESHOLD = {threshold:.4f})")
    print(f"{'=' * 75}")

    false_accepts = []
    false_rejects = []
    l1_premature_filters = []

    for res in scan_results:
        fname = res['filename']
        gt = res['ground_truth']
        winner = res['winner_user']
        score = res['winner_score']
        gt_in_enrolled = res['gt_in_enrolled']
        accepted = (score <= threshold)

        # Check Layer 1 filter failure on genuine palm
        if gt_in_enrolled and res['gt_l1_filtered_out']:
            l1_premature_filters.append(res)

        if accepted:
            # False Accept: Accepted as winner, but winner != ground truth
            if winner != gt:
                false_accepts.append(res)
        else:
            # False Reject: Genuine user rejected
            if gt_in_enrolled:
                false_rejects.append(res)

    print(f"Total Scans Audited         : {len(scan_results)}")
    print(f"False Accepts (FA)          : {len(false_accepts)}")
    print(f"False Rejects (FR)          : {len(false_rejects)}")
    print(f"Layer 1 Premature Exclusions: {len(l1_premature_filters)}")

    # Print Layer 1 Premature Filter Failures
    if l1_premature_filters:
        print(f"\n[!] ALERT: {len(l1_premature_filters)} genuine scans were filtered out by Layer 1!")
        print(f"{'Scan Filename':<32} | {'True User':<16} | {'True L1 Dist':<12} | {'L1 Threshold'}")
        print(f"{'-' * 32}-+-{'-' * 16}-+-{'-' * 12}-+-{'-' * 12}")
        for r in l1_premature_filters:
            print(f"{r['filename']:<32} | {r['ground_truth']:<16} | {r['gt_l1']:<12.4f} | {L1_THRESHOLD:.4f}")

    # Print False Accepts
    if false_accepts:
        print(f"\n[!] FALSE ACCEPT DETAILS ({len(false_accepts)} cases):")
        print(f"{'Scan Filename':<26} | {'Ground Truth':<15} | {'Accepted As':<15} | {'Score':<8} | {'GT Score':<8} | {'Winner L1'}")
        print(f"{'-' * 26}-+-{'-' * 15}-+-{'-' * 15}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 9}")
        for fa in false_accepts:
            gt_s_str = f"{fa['gt_score']:.4f}" if fa['gt_score'] is not None else "N/A"
            print(f"{fa['filename']:<26} | {fa['ground_truth']:<15} | {fa['winner_user']:<15} | {fa['winner_score']:<8.4f} | {gt_s_str:<8} | {fa['winner_l1']:<9.4f}")
    else:
        print("\n[+] Zero False Accepts detected at current threshold.")

    # Print False Rejects
    if false_rejects:
        print(f"\n[!] FALSE REJECT DETAILS ({len(false_rejects)} cases):")
        print(f"{'Scan Filename':<28} | {'Ground Truth':<16} | {'Score':<8} | {'Threshold':<9} | {'L1 Dist':<8} | {'L1 Filtered?'}")
        print(f"{'-' * 28}-+-{'-' * 16}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 12}")
        for fr in false_rejects:
            l1_str = f"{fr['gt_l1']:.4f}" if fr['gt_l1'] is not None else "N/A"
            filt_str = "YES (PREMATURE)" if fr['gt_l1_filtered_out'] else "NO"
            print(f"{fr['filename']:<28} | {fr['ground_truth']:<16} | {fr['winner_score']:<8.4f} | {threshold:<9.4f} | {l1_str:<8} | {filt_str}")
    else:
        print("[+] Zero False Rejects detected at current threshold.")


def compute_threshold_sweep(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                            thresholds=(0.30, 0.35, 0.365, 0.38, 0.40, 0.45)):
    """
    Computes and prints the FAR/FRR threshold sweep table on real data.
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
        fa_count = int(np.sum(impostor_scores <= tau)) if n_imp > 0 else 0
        fr_count = int(np.sum(genuine_scores > tau)) if n_gen > 0 else 0

        far = (fa_count / n_imp * 100.0) if n_imp > 0 else 0.0
        frr = (fr_count / n_gen * 100.0) if n_gen > 0 else 0.0

        print(f"{tau:<10.4f} | {far:<10.2f} | {frr:<10.2f} | {total_decisions:<10} | {fa_count:<14} | {fr_count:<14}")

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_point = (tau, (far + frr) / 2.0)

    if eer_point:
        print(f"\nEmpirical EER Estimate: ~{eer_point[1]:.2f}% near threshold {eer_point[0]:.4f}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mine real captured palm vein templates & ROI scans for empirical accuracy."
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
    print(" PALM VEIN BIOMETRIC REAL-DATA DIAGNOSTICS & ACCURACY AUDIT")
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
        print("\n[!] No enrolled templates found in database. Exiting analysis.")
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

        # Combined distributions for overall sweep
        comb_genuine = np.concatenate([template_results['genuine_scores'], scan_analysis['genuine_scores']])
        comb_impostor = np.concatenate([template_results['impostor_scores'], scan_analysis['impostor_scores']])
        compute_threshold_sweep(comb_genuine, comb_impostor)
    else:
        print("\n[*] No scan ROI images to evaluate; running sweep on enrolled template pairs:")
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
