"""
mattress/reference.py — enrollment: build per-SKU signature clusters (G2, G5, G9).

Two-phase enroll:
  Phase 1 fits the stats-block normalisation (GLCM / macro z-score params) on
  ENROLL patches only; Phase 2 builds the per-SKU clusters with those frozen
  params. Query time reuses the stored params — never refit on val/live data.
Empty SKU folders (placeholders like memorise/, purity_plus/, _unknown/) are
skipped with a warning; a NON-empty folder that yields zero clean patches is a
hard error (fail loud, G9).
"""
import os
import glob
import pickle
import cv2
import numpy as np

import config as cfg
from crop import localise_cover
from imageio import extract_grid_patches
from features import (CompositeSignaturePipeline, raw_blocks,
                      fit_stats_norm_params)


class ReferenceStore:

    def __init__(self):
        self.library = {}           # sku -> list of composite signatures
        self.norm_params = {}
        self.enrolled_resolution = (cfg.CAPTURE_W, cfg.CAPTURE_H)
        self.pipeline = None
        self.enroll_report = {}     # sku -> patch/rejection accounting

    # ------------------------------------------------------------------
    def _iter_sku_frames(self, root_dir, split):
        for sku_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
            if not os.path.isdir(sku_dir):
                continue
            sku = os.path.basename(sku_dir).lower()
            files = sorted(glob.glob(os.path.join(sku_dir, split, "*.jpg")))
            yield sku, files

    def enroll_from_directory(self, root_dir):
        # ---- collect clean patches per SKU (with accounting) ----
        sku_patches, skipped = {}, []
        for sku, files in self._iter_sku_frames(root_dir, "enroll"):
            if not files:
                skipped.append(sku)
                continue
            patches, agg = [], {'total': 0, 'sash_or_dark': 0, 'glare': 0,
                                'contrast': 0, 'kept': 0, 'frames': 0,
                                'zero_patch_frames': []}
            for path in files:
                img = cv2.imread(path)
                if img is None:
                    raise IOError(f"Unreadable image: {path}")
                cropped = localise_cover(img)          # raises on wrong resolution
                p, st = extract_grid_patches(cropped, return_stats=True)
                agg['frames'] += 1
                for k in ('total', 'sash_or_dark', 'glare', 'contrast', 'kept'):
                    agg[k] += st[k]
                if not p:
                    agg['zero_patch_frames'].append(os.path.basename(path))
                patches.extend(p)
            if not patches:
                raise ValueError(
                    f"FAIL LOUD: SKU '{sku}' has {len(files)} enroll frames but "
                    f"ZERO clean patches survived masking. Rejections: {agg}")
            sku_patches[sku] = patches
            self.enroll_report[sku] = agg

        if skipped:
            print(f"[enroll] skipping empty SKU folders: {', '.join(skipped)}")
        if not sku_patches:
            raise FileNotFoundError(f"Zero enrollment images under {root_dir}")

        # ---- Phase 1: fit stats normalisation on ALL enroll raw blocks ----
        all_raw = []
        sku_raw = {}
        for sku, patches in sku_patches.items():
            blocks = [raw_blocks(p) for p in patches]
            sku_raw[sku] = blocks
            all_raw.extend(blocks)
        self.norm_params = fit_stats_norm_params(all_raw)
        self.pipeline = CompositeSignaturePipeline(self.norm_params)

        # ---- Phase 2: compose signatures with frozen params ----
        for sku, blocks in sku_raw.items():
            self.library[sku] = [self.pipeline.compose(b) for b in blocks]
            self.enroll_report[sku]['signatures'] = len(self.library[sku])

    # ------------------------------------------------------------------
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'library': self.library,
                         'norm_params': self.norm_params,
                         'resolution': self.enrolled_resolution,
                         'enroll_report': self.enroll_report}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.library = payload['library']
        self.norm_params = payload['norm_params']
        self.enrolled_resolution = payload['resolution']
        self.enroll_report = payload.get('enroll_report', {})
        self.pipeline = CompositeSignaturePipeline(self.norm_params)
