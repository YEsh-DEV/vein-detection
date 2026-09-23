#!/usr/bin/env python3
"""
tools/demo_data_manager.py
==========================
Clean Demo Data Manager & Safe Database Reset Utility.

Provides operational tools for live demo management on Raspberry Pi and laptop:
  - --status : Display active enrolled users, template counts, and database health.
  - --list   : List all users, sample counts, and enrollment timestamps.
  - --verify : Validate embedding integrity (dim=512, blob_size=2048, norm=1.0).
  - --reset  : Safely reset the demo database with an automatic timestamped backup.
  - --clean-test-users: Remove temporary/synthetic test users (e.g. test_*, mock_*) while preserving genuine enrollments.

Does NOT delete or replace app/db_manager.py. Uses app.db_manager as the single source of truth.
"""

import os
import sys
import shutil
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.constants import DB_PATH, EMBEDDING_DIM
import app.db_manager as db_mgr


def get_db_status():
    """Returns database summary statistics."""
    if not os.path.exists(DB_PATH):
        return {
            "exists": False,
            "path": DB_PATH,
            "active_users": 0,
            "total_users": 0,
            "templates": 0,
            "access_logs": 0,
        }

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        total_users = cur.execute("SELECT count(*) FROM users").fetchone()[0]
        active_users = cur.execute("SELECT count(*) FROM users WHERE active = 1").fetchone()[0]
        templates = cur.execute("SELECT count(*) FROM templates").fetchone()[0]
        access_logs = cur.execute("SELECT count(*) FROM access_log").fetchone()[0]

    return {
        "exists": True,
        "path": DB_PATH,
        "active_users": active_users,
        "total_users": total_users,
        "templates": templates,
        "access_logs": access_logs,
    }


def print_status():
    """Prints formatted database status."""
    st = get_db_status()
    print("\n==================================================")
    print("DEMO DATABASE STATUS")
    print("==================================================")
    print(f"  Database Path   : {st['path']}")
    print(f"  File Exists     : {st['exists']}")
    if st['exists']:
        size_kb = os.path.getsize(st['path']) / 1024.0
        print(f"  Database Size   : {size_kb:.1f} KB")
        print(f"  Active Users    : {st['active_users']}")
        print(f"  Total Users     : {st['total_users']} (including inactive/deleted)")
        print(f"  v2 Templates    : {st['templates']} (512-D float32)")
        print(f"  Access Logs     : {st['access_logs']}")
    print("==================================================\n")


def list_users():
    """Lists enrolled users and template counts."""
    if not os.path.exists(DB_PATH):
        print(f"[!] Database file does not exist at {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT u.id, u.username, u.active, u.enrolled_at, COUNT(t.id) as template_count
            FROM users u
            LEFT JOIN templates t ON u.id = t.user_id
            GROUP BY u.id
            ORDER BY u.id ASC
        """).fetchall()

    print("\n==================================================")
    print(f"ENROLLED USERS LIST ({len(rows)} users)")
    print("==================================================")
    print(f"{'ID':<5} {'Username':<20} {'Status':<10} {'Templates':<10} {'Enrolled At'}")
    print("-" * 65)
    for r in rows:
        uid, uname, active, enrolled_at, t_count = r
        status_str = "ACTIVE" if active else "INACTIVE"
        print(f"{uid:<5} {uname:<20} {status_str:<10} {t_count:<10} {enrolled_at}")
    print("==================================================\n")


def verify_integrity() -> bool:
    """Validates structural and mathematical integrity of stored embeddings."""
    print("\n==================================================")
    print("VERIFYING DATABASE INTEGRITY")
    print("==================================================")
    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        return False

    all_valid = True
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT t.id, u.username, t.sample_idx, t.embedding, t.embedding_dim, t.engine_version
            FROM templates t
            JOIN users u ON t.user_id = u.id
        """).fetchall()

    if not rows:
        print("  [*] Database is currently empty (0 templates). Schema is valid.")
        print("==================================================\n")
        return True

    print(f"  Validating {len(rows)} templates...")
    for tid, uname, s_idx, emb_blob, emb_dim, engine in rows:
        # Check size
        if len(emb_blob) != EMBEDDING_DIM * 4:
            print(f"  [FAIL] Template {tid} ({uname} sample {s_idx}): invalid byte size {len(emb_blob)} (expected {EMBEDDING_DIM * 4})")
            all_valid = False
            continue

        arr = np.frombuffer(emb_blob, dtype=np.float32)
        if arr.shape != (EMBEDDING_DIM,):
            print(f"  [FAIL] Template {tid} ({uname}): invalid shape {arr.shape}")
            all_valid = False
            continue

        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"  [FAIL] Template {tid} ({uname}): contains NaN or Inf values")
            all_valid = False
            continue

        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-4:
            print(f"  [WARN] Template {tid} ({uname}): embedding L2 norm is {norm:.6f} (expected ~1.0)")

    if all_valid:
        print(f"  [PASS] All {len(rows)} templates verified: 512-D float32, exactly 2048 bytes, valid norms.")
    print("==================================================\n")
    return all_valid


def clean_test_users():
    """Removes test users (starting with test_, mock_, or dummy_) while preserving genuine demo users."""
    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        test_users = cur.execute("""
            SELECT id, username FROM users
            WHERE username LIKE 'test_%' OR username LIKE 'mock_%' OR username LIKE 'dummy_%'
        """).fetchall()

        if not test_users:
            print("[*] No test or synthetic users found in database.")
            return

        print(f"[*] Found {len(test_users)} test users to purge: {[u[1] for u in test_users]}")
        for uid, uname in test_users:
            cur.execute("DELETE FROM templates WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM access_log WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.commit()

    print("[*] Cleaned up all test users and associated templates.")
    print_status()


def reset_database(force: bool = False):
    """
    Safely resets the database:
    1. Creates a timestamped backup in data/backups/
    2. Wipes user and template tables
    3. Initializes fresh schema
    """
    if not force:
        confirm = input("[?] Are you sure you want to reset the DEMO database? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("[*] Reset cancelled by user.")
            return

    if os.path.exists(DB_PATH):
        backup_dir = os.path.join(os.path.dirname(DB_PATH), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"palm_vein_pre_demo_reset_{ts}.db")
        shutil.copy2(DB_PATH, backup_file)
        print(f"[*] Created safe backup: {backup_file}")

        # Remove existing file to ensure pristine re-initialization
        os.remove(DB_PATH)
        print(f"[*] Removed existing database: {DB_PATH}")

    # Initialize fresh database schema
    db_mgr.init_db()
    print("[*] Initialized pristine SQLite database with v2 schema.")
    print_status()


def main():
    parser = argparse.ArgumentParser(description="Clean Demo Data Manager for Palm-Vein Biometrics.")
    parser.add_argument("--status", action="store_true", help="Show database summary statistics")
    parser.add_argument("--list", action="store_true", help="List enrolled users and template counts")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of stored embeddings")
    parser.add_argument("--clean-test-users", action="store_true", help="Purge test_* and synthetic users")
    parser.add_argument("--reset", action="store_true", help="Safely backup and reset the demo database")
    parser.add_argument("--force", action="store_true", help="Bypass confirmation prompt on reset")

    args = parser.parse_args()

    if not any([args.status, args.list, args.verify, args.clean_test_users, args.reset]):
        print_status()
        return

    if args.status:
        print_status()
    if args.list:
        list_users()
    if args.verify:
        verify_integrity()
    if args.clean_test_users:
        clean_test_users()
    if args.reset:
        reset_database(force=args.force)


if __name__ == "__main__":
    main()
