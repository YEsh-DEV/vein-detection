#!/usr/bin/env python3
"""
db_manager.py
-------------
Single source of truth for all SQLite access in the palm vein system.
No other module in this project imports sqlite3 or touches palm_vein.db.

v2 Architecture: Stores 512-dim float32 embeddings (2048-byte BLOBs) extracted
by AMPVNet CNN. Replaces legacy v1 Gabor bitplanes and signatures.
"""

import os
import shutil
import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    from app.constants import DB_PATH, EMBEDDING_DIM
except ImportError:
    from constants import DB_PATH, EMBEDDING_DIM

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
    embedding    BLOB NOT NULL,      -- 512 x float32, exactly 2048 bytes
    quality_norm REAL,               -- placeholder column for future AdaFace-quality-based
                                     -- diagnostics; write NULL for now, do not compute
                                     -- a fake value
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


def _check_and_backup_v1_db():
    """
    Checks if palm_vein.db exists with legacy v1 Gabor schema columns
    (vr_blob, vi_blob, signature). If detected, copies the database to
    palm_vein_v1_gabor_backup_<YYYYMMDD_HHMMSS>.db before any ALTER/DROP.
    """
    if not os.path.exists(DB_PATH):
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            table_check = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='templates'"
            ).fetchone()
            if not table_check:
                return

            columns = [col[1] for col in cur.execute("PRAGMA table_info(templates)").fetchall()]
            is_v1 = any(c in columns for c in ("vr_blob", "vi_blob", "signature"))

            if is_v1:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"palm_vein_v1_gabor_backup_{ts}.db"
                backup_path = os.path.join(os.path.dirname(DB_PATH), backup_name)
                shutil.copy2(DB_PATH, backup_path)
                print(f"[!] Legacy v1 Gabor schema detected. Automatically backed up '{DB_PATH}' -> '{backup_path}'")

                # Drop old templates table so v2 schema can be applied cleanly
                cur.execute("DROP TABLE templates")
                cur.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
                conn.commit()
                print("[*] Legacy templates table dropped; ready for v2 embedding schema.")
    except Exception as e:
        print(f"[!] Warning during v1 database backup check: {e}")


def init_db():
    """Initializes SQLite database tables and indices with automated v1 backup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _check_and_backup_v1_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
        conn.commit()


def store_embedding(user_id: int, sample_idx: int, embedding: np.ndarray, quality_norm: Optional[float] = None) -> int:
    """
    Serializes and stores a 512-dim float32 embedding (strict 2048 bytes).
    Returns the newly created template_id.
    """
    blob = embedding.astype(np.float32).tobytes()
    if len(blob) != 2048:
        raise ValueError(f"Embedding must be exactly 2048 bytes (512 float32), got {len(blob)} bytes.")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO templates (user_id, sample_idx, embedding, quality_norm)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, sample_idx, blob, quality_norm)
        )
        conn.commit()
        return cur.lastrowid


def load_embedding(blob: bytes) -> np.ndarray:
    """
    Deserializes a 2048-byte BLOB into a writable 512-dim float32 numpy vector.
    The .copy() is required because np.frombuffer returns a read-only buffer view.
    """
    if len(blob) != 2048:
        raise ValueError(f"Invalid embedding BLOB size: expected 2048 bytes, got {len(blob)} bytes.")
    return np.frombuffer(blob, dtype=np.float32).copy()


def user_exists(username: str) -> bool:
    """Returns True if user is enrolled and active."""
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ? AND active = 1",
            (username,)
        ).fetchone()
    return row is not None


def enroll_user(username: str, embedding_list: list) -> int:
    """
    Enrolls a new user with multiple 512-dim CNN embeddings.
    Maintains re-enrollment hygiene: hard-deletes any soft-deleted records for this username
    before inserting the new user.
    """
    username = username.strip().lower()

    if user_exists(username):
        raise ValueError(f"User '{username}' is already enrolled and active.")

    with sqlite3.connect(DB_PATH) as conn:
        try:
            # Hard-delete any soft-deleted user (and their templates) with this username
            conn.execute(
                "DELETE FROM templates WHERE user_id IN (SELECT id FROM users WHERE username = ? AND active = 0)",
                (username,)
            )
            conn.execute("DELETE FROM users WHERE username = ? AND active = 0", (username,))

            cursor = conn.execute(
                "INSERT INTO users (username) VALUES (?)", (username,)
            )
            user_id = cursor.lastrowid

            for sample_idx, sample in enumerate(embedding_list):
                if isinstance(sample, dict) and "embedding" in sample:
                    emb = sample["embedding"]
                else:
                    emb = sample

                blob = emb.astype(np.float32).tobytes()
                if len(blob) != 2048:
                    raise ValueError(f"Embedding must be exactly 2048 bytes, got {len(blob)} bytes.")

                conn.execute(
                    """
                    INSERT INTO templates (user_id, sample_idx, embedding, quality_norm)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, sample_idx, blob, None)
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return user_id


def get_all_embeddings() -> dict:
    """
    Loads all active users' template embeddings from SQLite into memory.
    Returns:
        {
            'matrix': np.ndarray of shape (N, 512), float32,
            'template_ids': list[int],
            'user_ids': list[int]
        }
    All three are kept in strict lockstep order from a single query.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.user_id, t.embedding
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            WHERE  u.active = 1
            ORDER  BY t.id
            """
        ).fetchall()

    if not rows:
        return {
            'matrix': np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            'template_ids': [],
            'user_ids': [],
        }

    template_ids = []
    user_ids = []
    embeddings = []

    for tid, uid, blob in rows:
        emb = load_embedding(blob)
        template_ids.append(tid)
        user_ids.append(uid)
        embeddings.append(emb)

    return {
        'matrix': np.stack(embeddings, axis=0) if embeddings else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        'template_ids': template_ids,
        'user_ids': user_ids,
    }


def get_all_signatures() -> dict:
    """Backward compatibility wrapper for get_all_embeddings()."""
    return get_all_embeddings()


def get_templates_by_ids(template_ids: list) -> list:
    """
    Loads full templates for specified IDs, returning (template_id, user_id, template_dict) tuples.
    Preserves the exact 3-tuple contract: (template_id, user_id, {"embedding": np.ndarray(512,)}).

    If an ID in template_ids does not exist in the database, it is omitted. The returned list
    has exactly the count of existing templates, correctly paired with their user_ids.
    """
    if not template_ids:
        return []

    placeholders = ",".join("?" for _ in template_ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT id, user_id, embedding FROM templates WHERE id IN ({placeholders})",
            template_ids
        ).fetchall()

    row_map = {}
    for tid, uid, blob in rows:
        emb = load_embedding(blob)
        row_map[tid] = (uid, {'embedding': emb})

    return [(tid, row_map[tid][0], row_map[tid][1]) for tid in template_ids if tid in row_map]


def get_username(user_id: int) -> str:
    """Returns username for a given user ID."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"No user found with id={user_id}")
    return row[0]


def log_access(user_id: Optional[int], score: float, accepted: bool):
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


def get_user_id(username: str):
    """Returns user_id for a given username if active, else None."""
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ? AND active = 1", (username,)
        ).fetchone()
    return row[0] if row else None


def reset_all_tables():
    """Clears all users, templates, and access logs."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM templates")
        conn.execute("DELETE FROM access_log")
        conn.execute("DELETE FROM users")
        conn.commit()


def compute_signature(VR: np.ndarray, VI: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Legacy 64-float signature helper retained for backward compatibility with
    historical analysis or tools. Unused in v2 runtime.
    """
    if VI is not None:
        comb = (VR.astype(np.float32) + VI.astype(np.float32)) * 0.5
    else:
        comb = VR.astype(np.float32)
    sig = comb.reshape(8, 32, 8, 32).mean(axis=(1, 3)).flatten().astype(np.float32)
    return sig
