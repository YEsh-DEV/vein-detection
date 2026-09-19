#!/usr/bin/env python3
"""
search_engine.py
----------------
Two-layer vein template search. This is the ONLY module that calls
match_templates() from gabor.py. No other file calls it directly.

Layer 1 — Signature pre-filter (RAM, ~0.5ms for 3,000 templates):
    Euclidean distance on 64-float (VR+VI)/2 signatures. Keeps candidates below
    L1_THRESHOLD.

Layer 2 — Parallel MNHD (4 Pi 5 cores, ~2ms per template):
    match_templates() via multiprocessing.Pool, created once and reused.
"""

import numpy as np
from multiprocessing import Pool

try:
    from app.constants import (
        MATCH_THRESHOLD, L1_THRESHOLD, TOP_K, L1_BYPASS_MAX_TEMPLATES,
    )
    from app.gabor import match_templates
    from app.db_manager import (
        compute_signature, get_all_signatures,
        get_templates_by_ids, get_username,
    )
except ImportError:
    from constants import (
        MATCH_THRESHOLD, L1_THRESHOLD, TOP_K, L1_BYPASS_MAX_TEMPLATES,
    )
    from gabor import match_templates
    from db_manager import (
        compute_signature, get_all_signatures,
        get_templates_by_ids, get_username,
    )


def _match_worker(args):
    """Module-level function required for multiprocessing.Pool pickling."""
    template, probe = args
    return match_templates(template, probe)


class SearchEngine:
    def __init__(self, n_workers: int = 4):
        self._pool         = Pool(processes=n_workers)
        self._sig_matrix   = None
        self._template_ids = []
        self._user_ids     = []
        self.L1_THRESHOLD  = L1_THRESHOLD
        self.TOP_K        = TOP_K
        self.L1_BYPASS_MAX = L1_BYPASS_MAX_TEMPLATES
        self.refresh_cache()

    def refresh_cache(self):
        """Reload all signatures from DB into RAM."""
        data = get_all_signatures()

        if len(data['template_ids']) == 0:
            self._sig_matrix   = np.zeros((0, 64), dtype=np.float32)
            self._template_ids = []
            self._user_ids     = []
            return

        self._sig_matrix   = data['matrix']
        self._template_ids = data['template_ids']
        self._user_ids     = data['user_ids']

    def identify(self, probe_veincode: dict):
        """
        Identify a probe VeinCode against enrolled biometric database.
        Returns (username, best_score, best_user_id) if verified, else (None, best_score, None).
        """
        if self._sig_matrix is None or self._sig_matrix.shape[0] == 0:
            return None, 1.0, None

        return self._run_search(probe_veincode)

    def _run_search(self, probe_veincode: dict):
        n_templates = len(self._template_ids)
        if n_templates == 0:
            return None, 1.0, None

        if n_templates <= self.L1_BYPASS_MAX:
            # Small edge database: pass all enrolled templates to parallel Layer 2 without risk of premature filtering
            candidate_template_ids = self._template_ids
        else:
            probe_sig = compute_signature(probe_veincode['VR'], probe_veincode.get('VI'))
            dists     = np.linalg.norm(self._sig_matrix - probe_sig, axis=1)

            sorted_idx = np.argsort(dists)
            candidates = [i for i in sorted_idx if dists[i] < self.L1_THRESHOLD]
            candidates = candidates[:self.TOP_K]

            if len(candidates) == 0:
                candidates = sorted_idx[:min(10, len(sorted_idx))].tolist()

            candidate_template_ids = [self._template_ids[i] for i in candidates]

        # get_templates_by_ids returns synchronized [(tid, uid, template_dict), ...]
        records = get_templates_by_ids(candidate_template_ids)
        if not records:
            return None, 1.0, None

        match_user_ids = [r[1] for r in records]
        templates      = [r[2] for r in records]

        args   = [(t, probe_veincode) for t in templates]
        scores = self._pool.map(_match_worker, args)

        user_best = self._aggregate_per_user(match_user_ids, scores)
        if not user_best:
            return None, 1.0, None

        best_user_id = min(user_best, key=user_best.get)
        best_score   = user_best[best_user_id]

        if best_score <= MATCH_THRESHOLD:
            username = get_username(best_user_id)
            return username, best_score, best_user_id

        return None, best_score, None

    def _aggregate_per_user(self, user_ids: list, scores: list) -> dict:
        """
        Best-match aggregation: identity is confirmed if ANY enrolled pose matches.
        Eliminates penalty on diverse multi-pose enrollments.
        """
        user_best = {}
        for uid, score in zip(user_ids, scores):
            if uid not in user_best or score < user_best[uid]:
                user_best[uid] = score
        return user_best

    def close(self):
        """Terminates the multiprocessing pool."""
        self._pool.close()
        self._pool.join()
