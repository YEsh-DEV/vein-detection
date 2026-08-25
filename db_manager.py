#!/usr/bin/env python3
"""
db_manager.py
-------------
Single source of truth for all SQLite access in the palm vein system.
No other module in this project imports sqlite3 or touches palm_vein.db.
"""

import os
import zlib
import sqlite3
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "palm_vein.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT UNIQUE NOT NULL COLLATE NOCASE,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS templates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    sample_idx   INTEGER NOT NULL DEFAULT 0,
    vr_blob      BLOB NOT NULL,
    vi_blob      BLOB NOT NULL,
    signature    BLOB NOT NULL,
    vr_mean      REAL,
    vi_mean      REAL,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, sample_idx)
);

CREATE TABLE IF NOT EXISTS access_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER,
    score        REAL NOT NULL,
    accepted     INTEGER NOT NULL,
    scan_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key          TEXT PRIMARY KEY,
    value        TEXT
);

CREATE INDEX IF NOT EXISTS idx_templates_user ON templates(user_id);
CREATE INDEX IF NOT EXISTS idx_users_active   ON users(active, username);
"""


def compute_signature(VR: np.ndarray) -> np.ndarray:
    """Computes a compact 16-float signature from a 256x256 binary VR array."""
    block_means = VR.reshape(8, 32, 8, 32).mean(axis=(1, 3))
    sig = block_means.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    return sig.flatten().astype(np.float32)


def init_db():
    """Initializes SQLite database tables and indices."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', '1')")
        conn.commit()


def user_exists(username: str) -> bool:
    """Returns True if user is enrolled and active."""
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ? AND active = 1",
            (username,)
        ).fetchone()
    return row is not None


def enroll_user(username: str, veincode_list: list) -> int:
    """Enrolls a new user with multiple VeinCode templates."""
    username = username.strip().lower()

    if user_exists(username):
        raise ValueError(f"User '{username}' is already enrolled and active.")

    with sqlite3.connect(DB_PATH) as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username) VALUES (?)", (username,)
            )
            user_id = cursor.lastrowid

            for sample_idx, code in enumerate(veincode_list):
                VR = code['VR'].astype(np.uint8)
                VI = code['VI'].astype(np.uint8)

                vr_blob   = zlib.compress(VR.tobytes())
                vi_blob   = zlib.compress(VI.tobytes())
                sig       = compute_signature(VR)
                sig_blob  = sig.tobytes()
                vr_mean   = float(VR.mean())
                vi_mean   = float(VI.mean())

                conn.execute(
                    """
                    INSERT INTO templates
                        (user_id, sample_idx, vr_blob, vi_blob,
                         signature, vr_mean, vi_mean)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, sample_idx, vr_blob, vi_blob,
                     sig_blob, vr_mean, vi_mean)
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return user_id


def get_all_signatures() -> dict:
    """Loads all active users' template signatures from SQLite into memory."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.user_id, t.signature
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            WHERE  u.active = 1
            ORDER  BY t.id
            """
        ).fetchall()

    if not rows:
        return {
            'matrix':       np.zeros((0, 16), dtype=np.float32),
            'template_ids': [],
            'user_ids':     [],
        }

    template_ids = []
    user_ids     = []
    sigs         = []

    for tid, uid, sig_blob in rows:
        template_ids.append(tid)
        user_ids.append(uid)
        sigs.append(np.frombuffer(sig_blob, dtype=np.float32))

    return {
        'matrix':       np.stack(sigs, axis=0),
        'template_ids': template_ids,
        'user_ids':     user_ids,
    }


def get_all_templates() -> list:
    """Returns all enrolled templates in the database."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT t.id, u.username, t.vr_blob, t.vi_blob
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            WHERE  u.active = 1
            """
        ).fetchall()

    results = []
    for tid, uname, vr_blob, vi_blob in rows:
        VR = np.frombuffer(zlib.decompress(vr_blob), dtype=np.uint8).reshape(256, 256)
        VI = np.frombuffer(zlib.decompress(vi_blob), dtype=np.uint8).reshape(256, 256)
        results.append({'id': tid, 'username': uname, 'template': {'VR': VR, 'VI': VI}})
    return results


def get_templates_by_ids(template_ids: list) -> list:
    """Loads and decompresses full VeinCode templates for specified IDs."""
    if not template_ids:
        return []

    placeholders = ",".join("?" for _ in template_ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT id, vr_blob, vi_blob FROM templates WHERE id IN ({placeholders})",
            template_ids
        ).fetchall()

    row_map = {}
    for tid, vr_blob, vi_blob in rows:
        VR = np.frombuffer(zlib.decompress(vr_blob), dtype=np.uint8).reshape(256, 256)
        VI = np.frombuffer(zlib.decompress(vi_blob), dtype=np.uint8).reshape(256, 256)
        row_map[tid] = {'VR': VR, 'VI': VI}

    return [row_map[tid] for tid in template_ids if tid in row_map]


def get_username(user_id: int) -> str:
    """Returns username for a given user ID."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"No user found with id={user_id}")
    return row[0]


def log_access(user_id, score: float, accepted: bool):
    """Inserts an access event into the audit log."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO access_log (user_id, score, accepted) VALUES (?, ?, ?)",
            (user_id, float(score), int(accepted))
        )
        conn.commit()


def list_users() -> list:
    """Returns list of all active enrolled users with sample counts."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT u.username,
                   COUNT(t.id) AS sample_count,
                   u.enrolled_at
            FROM   users u
            LEFT JOIN templates t ON t.user_id = u.id
            WHERE  u.active = 1
            GROUP  BY u.id
            ORDER  BY u.enrolled_at DESC
            """
        ).fetchall()

    return [
        {'username': row[0], 'sample_count': row[1], 'enrolled_at': row[2]}
        for row in rows
    ]


def delete_user(username: str):
    """Soft-deletes a user from active queries."""
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, active FROM users WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            raise ValueError(f"User '{username}' not found.")
        if row[1] == 0:
            raise ValueError(f"User '{username}' is already inactive.")

        conn.execute(
            "UPDATE users SET active = 0 WHERE username = ?", (username,)
        )
        conn.commit()
