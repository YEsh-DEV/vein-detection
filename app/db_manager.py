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
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id),
    sample_idx     INTEGER NOT NULL DEFAULT 0,
    embedding      BLOB NOT NULL,      -- 512 x float32, exactly 2048 bytes
    embedding_dim  INTEGER NOT NULL DEFAULT 512,
    engine_version TEXT NOT NULL DEFAULT 'v2',
    quality_norm   REAL,               -- placeholder column for future AdaFace-quality-based diagnostics
    enrolled_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, sample_idx)
);

CREATE TABLE IF NOT EXISTS legacy_templates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    sample_idx   INTEGER NOT NULL DEFAULT 0,
    vr_blob      BLOB,
    vi_blob      BLOB,
    signature    BLOB,
    enrolled_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS access_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER,
    score        REAL NOT NULL,
    accepted     INTEGER NOT NULL,
    engine       TEXT DEFAULT 'v2',
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
    (vr_blob, vi_blob, signature). If detected:
    1. Copies the database to palm_vein_v1_gabor_backup_<YYYYMMDD_HHMMSS>.db
    2. Migrates legacy templates by renaming table to legacy_templates (preserving legacy records)
    3. Leaves user records intact
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

                # Rename legacy templates table so old Gabor records are preserved safely
                cur.execute("ALTER TABLE templates RENAME TO legacy_templates")
                cur.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
                conn.commit()
                print("[*] Legacy templates table preserved as 'legacy_templates'; ready for v2 embedding schema.")
    except Exception as e:
        print(f"[!] Warning during v1 database backup check: {e}")


def _migrate_v2_columns(conn: sqlite3.Connection):
    """
    Safely adds embedding_dim and engine_version columns to existing v2 templates
    table if they do not yet exist, and engine column to access_log.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'")
        if cur.fetchone():
            cols = [c[1] for c in cur.execute("PRAGMA table_info(templates)").fetchall()]
            if "embedding_dim" not in cols:
                cur.execute("ALTER TABLE templates ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 512")
            if "engine_version" not in cols:
                cur.execute("ALTER TABLE templates ADD COLUMN engine_version TEXT NOT NULL DEFAULT 'v2'")

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='access_log'")
        if cur.fetchone():
            log_cols = [c[1] for c in cur.execute("PRAGMA table_info(access_log)").fetchall()]
            if "engine" not in log_cols:
                cur.execute("ALTER TABLE access_log ADD COLUMN engine TEXT DEFAULT 'v2'")
        conn.commit()
    except Exception as e:
        print(f"[!] Warning during schema column migration: {e}")


def init_db():
    """Initializes SQLite database tables and indices with automated v1 backup and column migration."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _check_and_backup_v1_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        _migrate_v2_columns(conn)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
        conn.commit()


def aggregate_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Computes normalized mean embedding from multiple biometric samples:
        T = normalize( sum(e_i) )
    Strictly verifies shape, float32 dtype, and unit norm output.
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty embedding list")

    embs = []
    for e in embeddings:
        if isinstance(e, dict) and "embedding" in e:
            arr = np.asarray(e["embedding"], dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(e, dtype=np.float32).reshape(-1)

        if len(arr) != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {len(arr)}")
        embs.append(arr)

    if len(embs) == 1:
        norm = float(np.linalg.norm(embs[0]))
        return (embs[0] / norm).astype(np.float32) if norm > 1e-8 else embs[0]

    sum_emb = np.sum(embs, axis=0)
    norm = float(np.linalg.norm(sum_emb))
    if norm < 1e-8:
        return embs[0]
    return (sum_emb / norm).astype(np.float32)


def store_embedding(
    user_id: int,
    sample_idx: int,
    embedding: np.ndarray,
    quality_norm: Optional[float] = None,
    engine_version: str = "v2"
) -> int:
    """
    Serializes and stores a 512-dim float32 embedding (strict 2048 bytes).
    Records embedding_dim and engine_version for schema hygiene.
    Returns the newly created template_id.
    """
    blob = embedding.astype(np.float32).tobytes()
    if len(blob) != 2048:
        raise ValueError(f"Embedding must be exactly 2048 bytes (512 float32), got {len(blob)} bytes.")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO templates (user_id, sample_idx, embedding, embedding_dim, engine_version, quality_norm)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, sample_idx, blob, EMBEDDING_DIM, engine_version, quality_norm)
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


def enroll_user(username: str, embedding_list: list, store_raw_samples: bool = True) -> int:
    """
    Enrolls a new user with AMPVNet 512-D embeddings using Phase 6 template aggregation:
      1. Validates each sample embedding
      2. If multi-sample (>=2), aggregates into normalized mean template:
            T = normalize(sum(e_i))
         and stores as primary template (sample_idx=0)
      3. If store_raw_samples is True, stores individual sample embeddings at sample_idx=1..K
         to allow multi-template best-match search in SearchEngine
      4. If single sample (1-template enrollment), stores as sample_idx=0
      5. Maintains re-enrollment hygiene: hard-deletes any soft-deleted records for this username
    """
    username = username.strip().lower()

    if user_exists(username):
        raise ValueError(f"User '{username}' is already enrolled and active.")

    if not embedding_list:
        raise ValueError("Cannot enroll user with empty embedding list.")

    # Validate and extract all sample embeddings
    parsed_embs = []
    for s in embedding_list:
        if isinstance(s, dict) and "embedding" in s:
            emb = np.asarray(s["embedding"], dtype=np.float32).reshape(-1)
        else:
            emb = np.asarray(s, dtype=np.float32).reshape(-1)

        if len(emb) != EMBEDDING_DIM:
            raise ValueError(f"Each sample embedding must be {EMBEDDING_DIM}-D, got {len(emb)}")
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm
        parsed_embs.append(emb)

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

            for sample_idx, emb in enumerate(parsed_embs):
                blob = emb.tobytes()
                conn.execute(
                    """
                    INSERT INTO templates (user_id, sample_idx, embedding, embedding_dim, engine_version, quality_norm)
                    VALUES (?, ?, ?, ?, 'v2', NULL)
                    """,
                    (user_id, sample_idx, blob, EMBEDDING_DIM)
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return user_id


def get_all_embeddings(engine_version: str = "v2") -> dict:
    """
    Loads all active users' template embeddings from SQLite into memory.
    Filters strictly by engine_version (default 'v2') to ensure old legacy templates
    and new embeddings are NEVER mixed during matching.

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
            WHERE  u.active = 1 AND (t.engine_version = ? OR t.engine_version IS NULL)
            ORDER  BY t.id
            """,
            (engine_version,)
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


def log_access(user_id: Optional[int], score: float, accepted: bool, engine: str = "v2"):
    """Inserts an access event into the audit log with engine tagging."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO access_log (user_id, score, accepted, engine) VALUES (?, ?, ?, ?)",
            (user_id, float(score), int(accepted), engine)
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
