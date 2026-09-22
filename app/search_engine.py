#!/usr/bin/env python3
"""
search_engine.py
----------------
Vectorized cosine similarity search engine for 512-dim palm vein embeddings.
Replaces the legacy two-layer (RAM Euclidean pre-filter + parallel multiprocessing MNHD pool)
engine with a single-pass BLAS matrix-vector dot product in RAM.
"""

import time
from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    from app.constants import MATCH_THRESHOLD, EMBEDDING_DIM
    from app.db_manager import get_all_embeddings, get_username
except ImportError:
    from constants import MATCH_THRESHOLD, EMBEDDING_DIM
    from db_manager import get_all_embeddings, get_username


class SearchEngine:
    """
    In-memory fast cosine search engine for AMPVNet biometric templates.
    Caches all enrolled templates in a contiguous 2D float32 numpy matrix.
    """

    def __init__(self):
        self._embedding_matrix: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._template_ids: List[int] = []
        self._user_ids: List[int] = []
        self.MATCH_THRESHOLD: float = MATCH_THRESHOLD
        self.refresh_cache()

    def refresh_cache(self):
        """Reload all embeddings from DB into a contiguous in-RAM matrix."""
        data = get_all_embeddings()
        if len(data['template_ids']) == 0:
            self._embedding_matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._template_ids = []
            self._user_ids = []
            return

        # Ensure lockstep ordering
        self._embedding_matrix = data['matrix'].astype(np.float32)
        self._template_ids = list(data['template_ids'])
        self._user_ids = list(data['user_ids'])

    def identify(self, probe_embedding: np.ndarray) -> Tuple[Optional[str], float, Optional[int]]:
        """
        Identify a probe embedding against enrolled templates.
        Returns: (username, best_score, best_user_id) if accepted, else (None, best_score, None).
        If database is empty, returns (None, -1.0, None) immediately without calling argmax.
        """
        diag = self.identify_with_diagnostics(probe_embedding)
        return diag['username'], diag['score'], diag['user_id']

    def identify_with_diagnostics(self, probe_embedding: np.ndarray) -> dict:
        """
        Computes cosine similarity against all enrolled templates via a single
        matrix-vector dot product and aggregates scores per user using best-match (MAX).

        Returns diagnostic dictionary:
        - winner username, user_id, best_score, accepted decision
        - full ranked list of ALL enrolled users with their highest similarity score
        - matching latency in ms
        """
        empty_res = {
            'username': None,
            'score': -1.0,
            'user_id': None,
            'accepted': False,
            'threshold': MATCH_THRESHOLD,
            'ranked_candidates': [],
            't_match_ms': 0.0,
        }

        if self._embedding_matrix is None or len(self._template_ids) == 0:
            return empty_res

        t0 = time.time()

        # Flatten probe embedding defensively to (512,)
        probe = probe_embedding.reshape(-1).astype(np.float32)

        # Single BLAS matrix-vector product: shape (N_templates,)
        similarities = self._embedding_matrix @ probe

        # Best-Match Aggregation: user_score = MAX(similarities for that user's templates)
        # Cosine similarity: higher is better (opposite of MNHD distance which used MIN)
        user_best_score: Dict[int, float] = {}
        for uid, sim in zip(self._user_ids, similarities):
            sim_val = float(sim)
            if uid not in user_best_score or sim_val > user_best_score[uid]:
                user_best_score[uid] = sim_val

        t_match_ms = round((time.time() - t0) * 1000, 2)

        if not user_best_score:
            return empty_res

        # Build full ranked candidates list sorted by score DESCENDING (highest similarity first)
        ranked_candidates = []
        for uid, score in user_best_score.items():
            try:
                uname = get_username(uid)
            except KeyError:
                uname = f"user_{uid}"

            ranked_candidates.append({
                'username': uname,
                'user_id': uid,
                'score': round(float(score), 4),
            })

        ranked_candidates.sort(key=lambda x: x['score'], reverse=True)

        best_user_id = max(user_best_score, key=user_best_score.get)
        best_score = float(user_best_score[best_user_id])

        # Decision rule: ACCEPT if best_score >= MATCH_THRESHOLD else REJECT
        # Greater-than-or-equal indicates acceptance (higher cosine similarity = more confident match)
        accepted = (best_score >= MATCH_THRESHOLD)

        winner_username = None
        if accepted:
            try:
                winner_username = get_username(best_user_id)
            except KeyError:
                winner_username = None

        return {
            'username': winner_username,
            'score': round(best_score, 4),
            'user_id': best_user_id if accepted else None,
            'accepted': accepted,
            'threshold': MATCH_THRESHOLD,
            'ranked_candidates': ranked_candidates,
            't_match_ms': t_match_ms,
        }

    def close(self):
        """No-op kept for lifecycle compatibility since multiprocessing pool is retired."""
        pass
