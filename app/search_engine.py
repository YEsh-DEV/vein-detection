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
        diag = self.identify_with_diagnostics(probe_veincode)
        return diag['username'], diag['score'], diag['user_id']

    def identify_with_diagnostics(self, probe_veincode: dict) -> dict:
        """
        Runs two-layer biometric identification and returns comprehensive diagnostic telemetry:
        - winner username, user_id, best_score, accepted decision
        - full ranked list of ALL enrolled users with both L1 signature distance and L2 MNHD score
        - list of templates/users filtered by Layer 1
        - per-stage latency (L1 and L2 matching in ms)
        """
        empty_res = {
            'username': None,
            'score': 1.0,
            'user_id': None,
            'accepted': False,
            'threshold': MATCH_THRESHOLD,
            'ranked_candidates': [],
            'l1_filtered_out': [],
            't_l1_ms': 0.0,
            't_l2_ms': 0.0,
        }

        if self._sig_matrix is None or len(self._template_ids) == 0:
            return empty_res

        n_templates = len(self._template_ids)

        # ── Layer 1: Vectorized Signature Pre-Filter ──
        t_l1_start = time.time()
        probe_sig = compute_signature(probe_veincode['VR'], probe_veincode.get('VI'))
        l1_dists = np.linalg.norm(self._sig_matrix - probe_sig, axis=1)
        t_l1_ms = round((time.time() - t_l1_start) * 1000, 2)

        # Track L1 distances and filtered templates
        l1_filtered_out = []
        for idx, (tid, uid) in enumerate(zip(self._template_ids, self._user_ids)):
            dist = float(l1_dists[idx])
            if dist >= self.L1_THRESHOLD:
                try:
                    uname = get_username(uid)
                except KeyError:
                    uname = f"user_{uid}"
                l1_filtered_out.append({
                    'template_id': tid,
                    'user_id': uid,
                    'username': uname,
                    'l1_dist': round(dist, 4),
                    'l1_threshold': self.L1_THRESHOLD,
                })

        # Determine candidates for Layer 2
        if n_templates <= self.L1_BYPASS_MAX:
            # Edge database bypass: evaluate all templates in Layer 2
            candidate_indices = list(range(n_templates))
        else:
            sorted_idx = np.argsort(l1_dists)
            candidates = [i for i in sorted_idx if l1_dists[i] < self.L1_THRESHOLD]
            candidates = candidates[:self.TOP_K]
            if len(candidates) == 0:
                candidates = sorted_idx[:min(10, len(sorted_idx))].tolist()
            candidate_indices = candidates

        candidate_template_ids = [self._template_ids[i] for i in candidate_indices]

        # ── Layer 2: Multiprocessing Parallel MNHD ──
        t_l2_start = time.time()
        records = get_templates_by_ids(candidate_template_ids)
        if not records:
            return empty_res

        match_user_ids = [r[1] for r in records]
        templates      = [r[2] for r in records]

        args = [(t, probe_veincode) for t in templates]
        scores = self._pool.map(_match_worker, args)
        t_l2_ms = round((time.time() - t_l2_start) * 1000, 2)

        # Aggregate L2 scores and L1 distances per user
        user_best_l2 = {}
        for uid, score in zip(match_user_ids, scores):
            if uid not in user_best_l2 or score < user_best_l2[uid]:
                user_best_l2[uid] = float(score)

        user_min_l1 = {}
        for idx, uid in enumerate(self._user_ids):
            dist = float(l1_dists[idx])
            if uid not in user_min_l1 or dist < user_min_l1[uid]:
                user_min_l1[uid] = dist

        # Build full ranked candidates list
        all_unique_uids = list(dict.fromkeys(self._user_ids))
        ranked_candidates = []
        for uid in all_unique_uids:
            try:
                uname = get_username(uid)
            except KeyError:
                uname = f"user_{uid}"

            best_l2 = user_best_l2.get(uid, 1.0)
            min_l1 = user_min_l1.get(uid, 1.0)
            passed_l1 = (min_l1 < self.L1_THRESHOLD)

            ranked_candidates.append({
                'username': uname,
                'user_id': uid,
                'score': round(best_l2, 4),
                'l1_dist': round(min_l1, 4),
                'passed_l1': passed_l1,
            })

        # Rank by best L2 score ascending
        ranked_candidates.sort(key=lambda x: x['score'])

        if not user_best_l2:
            return empty_res

        best_user_id = min(user_best_l2, key=user_best_l2.get)
        best_score   = user_best_l2[best_user_id]
        accepted     = (best_score <= MATCH_THRESHOLD)

        winner_username = None
        if accepted:
            try:
                winner_username = get_username(best_user_id)
            except KeyError:
                winner_username = None

        return {
            'username': winner_username,
            'score': round(float(best_score), 4),
            'user_id': best_user_id if accepted else None,
            'accepted': accepted,
            'threshold': MATCH_THRESHOLD,
            'ranked_candidates': ranked_candidates,
            'l1_filtered_out': l1_filtered_out,
            't_l1_ms': t_l1_ms,
            't_l2_ms': t_l2_ms,
        }

    def close(self):
        """Terminates the multiprocessing pool."""
        self._pool.close()
        self._pool.join()
