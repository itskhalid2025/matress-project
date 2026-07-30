"""
mattress/features.py — pure feature extractors + balanced composite (G5).

Design (fixes P0-3 and P1-1 from the v2 review):
  * Every block is brought to a COMPARABLE unit scale before weighting:
      - histogram blocks (colour, LBP): L1-normalised distribution, then unit L2;
      - stats blocks (GLCM, macro-FFT): z-scored with parameters FITTED ON THE
        ENROLL SET ONLY, clipped to +/-STATS_Z_CLIP, scaled so block norm <= 1.
    This makes the WEIGHT_* knobs actually mean something.
  * block_layout is DERIVED from the real vectors at build time — never hardcoded.
  * Macro-pattern cue = FFT radial/angular profile (band energy shares + angular
    anisotropy per band). Anisotropy is the measured discriminator for
    ortholex (oriented periodic) vs maxi_pro (isotropic mottle), and the stats
    are rotation-invariant by construction (band totals / max-over-angle).
  * Rotation invariance: uniform rotation-invariant LBP; GLCM props averaged over
    angles; FFT stats orientation-agnostic.
  * No file I/O, no camera, no matching in this module.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import config as cfg

LBP_N_BINS = cfg.LBP_P + 2  # uniform LBP values 0..P+1  -> 10 bins for P=8


# ---------------- raw (unnormalised) block extractors ----------------

def color_block(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [cfg.COLOR_H_BINS, cfg.COLOR_S_BINS],
                        [0, 180, 0, 256]).flatten()
    return hist / (hist.sum() + 1e-9)          # L1 distribution


def lbp_block(gray):
    lbp = local_binary_pattern(gray, cfg.LBP_P, cfg.LBP_R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_BINS + 1),
                           range=(0, LBP_N_BINS))
    return hist / (hist.sum() + 1e-9)          # L1 distribution


def glcm_block(gray):
    q = (gray // (256 // cfg.GLCM_LEVELS)).astype(np.uint8)   # quantise
    glcm = graycomatrix(q, distances=cfg.GLCM_DISTANCES,
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=cfg.GLCM_LEVELS, symmetric=True, normed=True)
    # average over angles AND distances -> rotation-robust scalars
    return np.array([np.mean(graycoprops(glcm, p)) for p in cfg.GLCM_PROPS])


def macro_fft_block(gray):
    """Band energy shares + per-band angular anisotropy (peak/uniform ratio)."""
    g = gray.astype(np.float64)
    f = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean()))) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(f.shape)
    r = np.hypot(y - cy, x - cx)
    ang = (np.arctan2(y - cy, x - cx) % np.pi)

    shares, anisos = [], []
    for r_lo, r_hi in cfg.MACRO_BANDS:
        sel = (r >= r_lo) & (r < r_hi)
        e_band = f[sel]
        tot = e_band.sum() + 1e-12
        shares.append(tot)
        bins = (ang[sel] / np.pi * cfg.MACRO_N_ANGLES).astype(int)
        bins = np.clip(bins, 0, cfg.MACRO_N_ANGLES - 1)
        e_ang = np.bincount(bins, e_band, minlength=cfg.MACRO_N_ANGLES)
        anisos.append(e_ang.max() / (tot / cfg.MACRO_N_ANGLES))

    shares = np.array(shares)
    shares = shares / (shares.sum() + 1e-12)   # relative energy per band
    return np.concatenate([shares, np.array(anisos)])


def raw_blocks(patch_bgr):
    """Patch -> dict of raw per-cue vectors. Pure function."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return {
        'color': color_block(patch_bgr),
        'lbp':   lbp_block(gray),
        'glcm':  glcm_block(gray),
        'macro': macro_fft_block(gray),
    }


STATS_BLOCKS = ('glcm', 'macro')               # z-scored with enroll params
BLOCK_ORDER = ('color', 'lbp', 'glcm', 'macro')
WEIGHTS = {'color': cfg.WEIGHT_COLOR, 'lbp': cfg.WEIGHT_LBP,
           'glcm': cfg.WEIGHT_GLCM, 'macro': cfg.WEIGHT_MACRO}


# ---------------- composite assembly ----------------

class CompositeSignaturePipeline:
    """
    Normalise-weight-concatenate. norm_params holds {block: (mean, std)} for
    stats blocks, fitted on ENROLL data only (never refit at query time).
    """

    def __init__(self, norm_params=None):
        self.norm_params = norm_params or {}
        self.block_layout = self._derive_layout()

    def _derive_layout(self):
        probe = raw_blocks(np.zeros((cfg.PATCH_SIZE, cfg.PATCH_SIZE, 3), np.uint8))
        layout, off = {}, 0
        for name in BLOCK_ORDER:
            n = len(probe[name])
            layout[name] = (off, off + n)
            off += n
        return layout

    def _normalise(self, name, vec):
        if name in STATS_BLOCKS:
            mean, std = self.norm_params.get(name, (np.zeros_like(vec),
                                                    np.ones_like(vec)))
            z = np.clip((vec - mean) / (std + 1e-9),
                        -cfg.STATS_Z_CLIP, cfg.STATS_Z_CLIP)
            return z / (cfg.STATS_Z_CLIP * np.sqrt(len(vec)))   # norm <= 1
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec                        # unit L2

    def compose(self, blocks):
        parts = [self._normalise(n, blocks[n]) * WEIGHTS[n] for n in BLOCK_ORDER]
        v = np.concatenate(parts)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def get_signature(self, patch_bgr):
        return self.compose(raw_blocks(patch_bgr))

    def slice_block(self, sig, name):
        a, b = self.block_layout[name]
        return sig[a:b]


def fit_stats_norm_params(raw_block_dicts):
    """Fit per-dimension mean/std for the stats blocks from enroll raw blocks."""
    params = {}
    for name in STATS_BLOCKS:
        mat = np.array([d[name] for d in raw_block_dicts])
        params[name] = (mat.mean(axis=0), mat.std(axis=0) + 1e-9)
    return params
