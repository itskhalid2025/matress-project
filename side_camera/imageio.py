"""
mattress/imageio.py — get from a raw frame to clean fabric patches.

Owns: deterministic grid tiling at native scale, frame-level exclusion masks
(sash by appearance, dark-ink/background, glare), and per-reason rejection
accounting so nothing is dropped silently (G1, G3, G4, G9).
"""
import cv2
import numpy as np
import config as cfg


def compute_sharpness(gray_img):
    """Laplacian variance quality metric."""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def _big_blob_mask(binary_u8, min_area, dilate_px):
    """Keep only connected components >= min_area, dilated by dilate_px."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)
    out = np.zeros_like(binary_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    if dilate_px > 0 and out.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        out = cv2.dilate(out, k)
    return out.astype(bool)


def _dominant_fabric_hue(hsv):
    """Mode of the hue histogram over moderately-saturated pixels — the cover's
    own fabric colour. Returns None when the frame has too few saturated pixels
    (near-white covers), in which case ANY saturated blob is sash-like."""
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    sel = s > cfg.FABRIC_HUE_MIN_SAT
    if sel.mean() < cfg.FABRIC_HUE_MIN_FRAC:
        return None
    hist = np.bincount(h[sel].ravel(), minlength=180)
    return int(np.argmax(hist))


def build_exclusion_masks(frame_bgr):
    """
    Frame-level masks computed ONCE per frame (G3):
      sash  — large connected blob of saturated+bright pixels whose HUE DIFFERS
              from the frame's own dominant fabric hue. Frame-adaptive: survives
              session-to-session saturation shifts (gravite's blue fabric rose
              from sat p90 114 to 174 between rot sessions, overlapping the sash
              range — a fixed saturation threshold provably cannot separate them).
              Appearance-based, NOT a fixed box.
      dark  — large connected dark blob (grey-sash print, background gaps).
      glare — near-clipping low-saturation pixels (plastic wrap).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    sat_bright = (s > cfg.SASH_S_THRESH) & (v > cfg.SASH_V_THRESH)
    fabric_hue = _dominant_fabric_hue(hsv)
    if fabric_hue is None:
        sash_px = sat_bright                     # near-white cover: any saturated blob
    else:
        dh = np.abs(h.astype(int) - fabric_hue)
        hue_far = np.minimum(dh, 180 - dh) > cfg.SASH_HUE_DELTA   # circular distance
        sash_px = sat_bright & hue_far
    sash = _big_blob_mask(sash_px.astype(np.uint8), cfg.SASH_MIN_AREA, cfg.SASH_DILATE)

    dark_px = (v < cfg.DARK_V_THRESH).astype(np.uint8)
    dark = _big_blob_mask(dark_px, cfg.DARK_MIN_AREA, cfg.DARK_DILATE)

    # WHITE sash (glossy laminate on a near-white cover) -- invisible to the
    # saturation rule above. Its signature is bright + unsaturated + locally
    # SMOOTH (quilted fabric always has stitching/weave micro-contrast; the
    # laminate has none). See config.py WHITE_SASH_* for the calibration
    # evidence (zero extra patches lost across all 74 real val frames).
    white_sash = np.zeros(v.shape, dtype=bool)
    if getattr(cfg, "WHITE_SASH_MASK", False):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = cfg.WHITE_SASH_STD_WIN
        mu = cv2.boxFilter(gray, -1, (win, win))
        mu2 = cv2.boxFilter(gray * gray, -1, (win, win))
        local_std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
        white_px = ((v > cfg.WHITE_SASH_V_MIN) & (s < cfg.WHITE_SASH_S_MAX)
                    & (local_std < cfg.WHITE_SASH_STD_MAX)).astype(np.uint8)
        white_sash = _big_blob_mask(white_px, cfg.WHITE_SASH_MIN_AREA,
                                    cfg.WHITE_SASH_DILATE)

    glare = (v > cfg.GLARE_V_THRESH) & (s < cfg.GLARE_S_THRESH)
    return (sash | dark | white_sash), glare


def extract_grid_patches(frame_bgr, return_stats=False, return_coords=False):
    """
    G1: deterministic fixed grid at native scale. Patches are REJECTED whole
    (never pixel-zeroed, G4) when they overlap the exclusion mask, are glare-
    heavy, or are blank. Returns list of patches (and, optionally, a rejection
    breakdown so callers can fail loud with reasons, G9).

    return_coords (2026-07-16, added for diagnose_texture_votes.py's
    visualizer -- non-breaking, off by default): also returns a list of
    (x, y) top-left pixel coordinates aligned 1:1 with the returned patches,
    so a caller can draw an overlay showing WHERE each vote came from in
    the actual crop.
    """
    h, w = frame_bgr.shape[:2]
    exclude, glare = build_exclusion_masks(frame_bgr)

    my = int(h * cfg.INTERIOR_MARGIN)
    mx = int(w * cfg.INTERIOR_MARGIN)
    ps, st = cfg.PATCH_SIZE, cfg.PATCH_STRIDE

    patches = []
    coords = []
    stats = {'total': 0, 'sash_or_dark': 0, 'glare': 0, 'contrast': 0, 'kept': 0}

    for y in range(my, h - my - ps + 1, st):
        for x in range(mx, w - mx - ps + 1, st):
            stats['total'] += 1
            if exclude[y:y + ps, x:x + ps].mean() > cfg.EXCLUDE_REJECT_FRAC:
                stats['sash_or_dark'] += 1
                continue
            if glare[y:y + ps, x:x + ps].mean() > cfg.GLARE_REJECT_FRAC:
                stats['glare'] += 1
                continue
            patch = frame_bgr[y:y + ps, x:x + ps]
            if np.std(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < cfg.MIN_CONTRAST_THRESH:
                stats['contrast'] += 1
                continue
            stats['kept'] += 1
            patches.append(patch)
            coords.append((x, y))

    result = [patches]
    if return_stats:
        result.append(stats)
    if return_coords:
        result.append(coords)
    return tuple(result) if len(result) > 1 else patches
