"""
mattress/pipeline.py — orchestrate crop -> gate -> sample -> extract -> identify.
"""
from enum import Enum
from collections import Counter
import cv2

import config as cfg
from crop import localise_cover
from imageio import compute_sharpness, extract_grid_patches
from identify import NearestSignatureIdentifier, NEEDS_REVIEW


class PipelineStatus(Enum):
    SUCCESS = "SUCCESS"
    BLURRY = "NEEDS_REVIEW_BLURRY"
    ZERO_CLEAN_PATCHES = "NEEDS_REVIEW_ZERO_PATCHES"
    LOW_VOTE_CONFIDENCE = "NEEDS_REVIEW_LOW_VOTE"
    UNKNOWN = "NEEDS_REVIEW_UNKNOWN"


class IdentificationPipeline:

    def __init__(self, reference_store):
        if reference_store.pipeline is None:
            raise ValueError("ReferenceStore has no fitted pipeline — enroll or load first.")
        self.store = reference_store
        self.identifier = NearestSignatureIdentifier(reference_store.library)

    def process_frame(self, frame_bgr):
        cropped = localise_cover(frame_bgr)             # raises on wrong resolution
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        if compute_sharpness(gray) < cfg.SHARPNESS_GATE:
            return PipelineStatus.BLURRY, NEEDS_REVIEW

        patches, st = extract_grid_patches(cropped, return_stats=True)
        if not patches:
            # fail loud with the reason breakdown, not a silent drop (G9)
            print(f"[pipeline] zero clean patches — rejections: {st}")
            return PipelineStatus.ZERO_CLEAN_PATCHES, NEEDS_REVIEW

        votes = [self.identifier.identify_signature(
                    self.store.pipeline.get_signature(p)) for p in patches]

        counts = Counter(v for v in votes if v != NEEDS_REVIEW)
        if not counts:
            return PipelineStatus.UNKNOWN, NEEDS_REVIEW

        winner, n = counts.most_common(1)[0]
        if n / len(votes) >= cfg.VOTE_MIN_FRAC:
            return PipelineStatus.SUCCESS, winner

        # Secondary, abstention-aware acceptance (2026-07-16): too-far
        # NEEDS_REVIEW patches are abstentions (white sash / floor / glare
        # -- material outside the library), not votes against the winner.
        # Accept when enough patches DECIDED, they agree overwhelmingly,
        # and no other SKU holds a meaningful share (genuine inter-SKU
        # confusion still lands in review). See config.py for the
        # controlled-experiment calibration behind these three knobs.
        decided_n = sum(counts.values())
        runner_n = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        if (decided_n >= cfg.VOTE_MIN_DECIDED
                and n / decided_n >= cfg.VOTE_MIN_FRAC_DECIDED
                and runner_n <= cfg.VOTE_DISSENT_MAX_FRAC * decided_n):
            return PipelineStatus.SUCCESS, winner

        # plurality exists but isn't confident enough (bias toward review)
        return PipelineStatus.LOW_VOTE_CONFIDENCE, NEEDS_REVIEW
