"""
mattress/identify.py — nearest-reference matching with BOTH guards (G7).

A positive SKU call requires: nearest distance <= REJECTION_MAX_DIST AND the
gap to the nearest DIFFERENT SKU >= MARGIN_GUARD. Anything else is
NEEDS_REVIEW — a confident wrong SKU is the expensive error.
"""
import numpy as np
import config as cfg

NEEDS_REVIEW = "NEEDS_REVIEW"


class NearestSignatureIdentifier:

    def __init__(self, reference_library):
        # {sku: np.ndarray of shape (n_refs, dim)} for vectorised distance
        self.library = {sku: np.asarray(sigs)
                        for sku, sigs in reference_library.items() if len(sigs)}

    def sku_distances(self, query_sig):
        """Min distance to each SKU's cluster (kept as a spread, matched to min)."""
        out = {}
        for sku, refs in self.library.items():
            out[sku] = float(np.min(np.linalg.norm(refs - query_sig, axis=1)))
        return out

    def identify_signature(self, query_sig, return_detail=False):
        if not self.library:
            return (NEEDS_REVIEW, None) if return_detail else NEEDS_REVIEW

        d = self.sku_distances(query_sig)
        ranked = sorted(d.items(), key=lambda kv: kv[1])
        best_sku, best = ranked[0]

        verdict = best_sku
        if best > cfg.REJECTION_MAX_DIST:
            verdict = NEEDS_REVIEW                       # too far from everything
        elif len(ranked) > 1 and (ranked[1][1] - best) < cfg.MARGIN_GUARD:
            verdict = NEEDS_REVIEW                       # ambiguous between SKUs

        if return_detail:
            return verdict, {'ranked': ranked, 'best': best,
                             'margin': (ranked[1][1] - best) if len(ranked) > 1 else None}
        return verdict
