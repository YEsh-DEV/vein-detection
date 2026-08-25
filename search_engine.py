#!/usr/bin/env python3
"""
search_engine.py
----------------
Two-layer vein template search. This is the ONLY module that calls
match_templates() from gabor.py. No other file calls it directly.

Layer 1 — Signature pre-filter (RAM, ~0.5ms for 3 000 templates):
    Euclidean distance on 16-float signatures. Keeps candidates below
    L1_THRESHOLD.

Layer 2 — Parallel MNHD (4 Pi 5 cores, ~2ms per template):
    match_templates() via multiprocessing.Pool, created once and reused.
"""

import numpy as np
from multiprocessing import Pool

from gabor import match_templates, MATCH_THRESHOLD
from db_manager import (
    compute_signature, get_all_signatures,
    get_templates_by_ids, get_username,
)


def _match_worker(args):
    """Module-level function required for multiprocessing.Pool pickling."""
    template, probe = args
    return match_templates(template, probe)


class SearchEngine:
    L1_THRESHOLD = 0.25
    TOP_K        = 80

    def __init__(self, n_workers: int = 4):
        self._pool         = Pool(processes=n_workers)
        self._sig_matrix   = None
        self._template_ids = []
        self._user_ids     = []
        self.refresh_cache()

    def refresh_cache(self):
        """Reload all signatures from DB into RAM."""
        data = get_all_signatures()

        if len(data['template_ids']) == 0:
            self._sig_matrix   = np.zeros((0, 16), dtype=np.float32)
            self._template_ids = []
            self._user_ids     = []
            return

        self._sig_matrix   = data['matrix']
        self._template_ids = data['template_ids']
        self._user_ids     = data['user_ids']

    def identify(self, probe_veincode: dict):
        """Identify a probe VeinCode against enrolled biometric database."""
        if self._sig_matrix.shape[0] == 0:
            return None, 1.0

        return self._run_search(probe_veincode)

    def _run_search(self, probe_veincode: dict):
        probe_sig = compute_signature(probe_veincode['VR'])
        dists     = np.linalg.norm(self._sig_matrix - probe_sig, axis=1)

        sorted_idx = np.argsort(dists)
        candidates = [i for i in sorted_idx if dists[i] < self.L1_THRESHOLD]
        candidates = candidates[:self.TOP_K]

        if len(candidates) == 0:
            candidates = sorted_idx[:min(10, len(sorted_idx))].tolist()

        candidate_template_ids = [self._template_ids[i] for i in candidates]
        candidate_user_ids     = [self._user_ids[i]     for i in candidates]
        templates              = get_templates_by_ids(candidate_template_ids)

        args   = [(t, probe_veincode) for t in templates]
        scores = self._pool.map(_match_worker, args)

        user_best = self._aggregate_per_user(candidate_user_ids, scores)

        best_user_id = min(user_best, key=user_best.get)
        best_score   = user_best[best_user_id]

        if best_score <= MATCH_THRESHOLD:
            username = get_username(best_user_id)
            return username, best_score

        return None, best_score

    def _aggregate_per_user(self, user_ids: list, scores: list) -> dict:
        user_scores_all: dict = {}
        for user_id, score in zip(user_ids, scores):
            user_scores_all.setdefault(user_id, []).append(score)

        user_best = {}
        for user_id, score_list in user_scores_all.items():
            min_s  = min(score_list)
            mean_s = sum(score_list) / len(score_list)
            user_best[user_id] = 0.7 * min_s + 0.3 * mean_s
        return user_best

    def shutdown(self):
        self._pool.terminate()
        self._pool.join()
