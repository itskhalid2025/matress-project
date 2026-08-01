"""
banner.py — Mattress Banner/Sash Detection and OCR Verification.

Combines visual template matching (ORB + RANSAC homography) and text recognition
(pytesseract OCR) in a 2-stage verification flow:
  Stage A: Fast path (hue isolation -> deskew -> ORB visual match -> OCR on band)
  Stage B: Fallback budgeted sweep (rotates full frame in fine angles -> OCR)
"""

import os
import re
import time
import pickle
import numpy as np
import cv2
import pytesseract
import platform

if platform.system() == "Windows":
    TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_WIN_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_WIN_PATH

import config as cfg

# ==============================================================================
# Constants & SKU Normalization
# ==============================================================================
CANONICAL_SKUS = {"gravite", "maxi_plush", "ortholex", "maxi_pro", "purity_plus", "memorise"}

_ALIASES = {
    "gravite": "gravite",
    "gravitemattress": "gravite",
    "maxiplush": "maxi_plush",
    "maxiplushmattress": "maxi_plush",
    "ortholex": "ortholex",
    "ortholexmattress": "ortholex",
    "maxipro": "maxi_pro",
    "maxiplushpro": "maxi_pro",
    "maxiplushpromattress": "maxi_pro",
    "purityplus": "purity_plus",
    "memorise": "memorise",
    "memorisemattress": "memorise",
}

def _squash(s: str) -> str:
    """Lowers and strips all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Build the lookup table for SKUs and aliases
_LOOKUP = {_squash(sku): sku for sku in CANONICAL_SKUS}
_LOOKUP.update(_ALIASES)
_LOOKUP_BY_LEN = sorted(_LOOKUP, key=len, reverse=True)

def normalize_sku(raw_text):
    """Maps raw OCR or template label to a canonical SKU string."""
    if not raw_text:
        return None
    sq = _squash(raw_text)
    if not sq:
        return None
    if sq in _LOOKUP:
        return _LOOKUP[sq]
    # Longest substring matching to resolve "maxi_pro" correctly before "maxi_plush"
    for key in _LOOKUP_BY_LEN:
        if key in sq:
            return _LOOKUP[key]
    return None

def _disambiguate_maxi(sku, raw_text):
    """
    Disambiguation logic to resolve the 'maxi_plush' vs 'maxi_pro' OCR confusion.
    If OCR misses/garbles 'PRO' in the banner, normalize_sku might incorrectly return 'maxi_plush'.
    """
    if sku != "maxi_plush":
        return sku
    tokens = re.findall(r"[a-z]+", raw_text.lower())
    if "pro" in tokens:
        return "maxi_pro"
    for i, tok in enumerate(tokens[:-1]):
        if "maxiplush" in tok:
            nxt = tokens[i + 1]
            if 2 <= len(nxt) <= 3 and "r" in nxt:
                return "maxi_pro"
    return sku

def _normalize_and_disambiguate(text):
    return _disambiguate_maxi(normalize_sku(text), text)

# ==============================================================================
# Template Matching Configurations (Stage C Visual Match)
# ==============================================================================
MIN_INLIERS = 100
CANON_BAND_HEIGHT = 300
ORB_N_FEATURES = 1500
MIN_KEYPOINTS = 20
LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 8
RANSAC_REPROJ_THRESH = 5.0

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")

SKU_GALLERIES = {
    "gravite":    {"hue": (0, 25),    "path": os.path.join(TEMPLATES_DIR, "gravite_banner_templates.pkl")},
    "ortholex":   {"hue": (95, 120),  "path": os.path.join(TEMPLATES_DIR, "ortholex_banner_templates.pkl")},
    "maxi_plush": {"hue": (135, 170), "path": os.path.join(TEMPLATES_DIR, "maxi_plush_banner_templates.pkl")},
}

_gallery_cache = {}
_orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def load_gallery(cache_path):
    """Loads and deserializes the ORB template features from the pickle cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Template pickle not found: {cache_path}")
    with open(cache_path, "rb") as f:
        serializable = pickle.load(f)
    templates = []
    for gray, kp_data, des, name in serializable:
        kp = [cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle,
                           response=response, octave=octave, class_id=class_id)
              for (pt, size, angle, response, octave, class_id) in kp_data]
        templates.append((gray, kp, des, name))
    return templates

def get_galleries_for_hue(hue_mode):
    """Lazy-loads and returns template galleries matching a hue range."""
    out = []
    for sku, entry in SKU_GALLERIES.items():
        lo, hi = entry["hue"]
        if lo <= hue_mode <= hi:
            if sku not in _gallery_cache:
                try:
                    _gallery_cache[sku] = load_gallery(entry["path"])
                except Exception as e:
                    print(f"[banner] Warning: Could not load template gallery for {sku}: {str(e)}")
                    _gallery_cache[sku] = []
            if _gallery_cache[sku]:
                out.append((sku, _gallery_cache[sku]))
    return out

def _canon_gray(band_bgr):
    """Resizes the banner band to a canonical height for stable ORB matching."""
    h, w = band_bgr.shape[:2]
    scale = CANON_BAND_HEIGHT / h
    resized = cv2.resize(band_bgr, (max(1, int(w * scale)), CANON_BAND_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

def match_score(band_bgr, templates):
    """Finds the maximum RANSAC inlier count against a set of templates."""
    gray_q = _canon_gray(band_bgr)
    kp_q, des_q = _orb.detectAndCompute(gray_q, None)
    if des_q is None or len(kp_q) < MIN_KEYPOINTS:
        return 0, None

    best_inliers, best_name = 0, None
    for gray_t, kp_t, des_t, name in templates:
        matches = _bf.knnMatch(des_q, des_t, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_GOOD_MATCHES:
            continue
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
            if mask is None:
                continue
            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_name = name
        except Exception:
            continue
            
    return best_inliers, best_name

# ==============================================================================
# Helper Math & Image Ops
# ==============================================================================
def _rotate_bound(image, angle):
    """Rotates an image by the given angle in degrees without clipping the corners."""
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=255)

# ==============================================================================
# Stage A: Fast Path (Hue Mode Isolation -> Axis Deskew -> Visual Match/OCR)
# ==============================================================================
BAND_HUE_TOL = 12
BAND_MAX_HUE_MODES = 2
BAND_MIN_AREA_FRAC2 = 0.004
BAND_OCR_HEIGHTS = (110, 160)
WEAK_MAGENTA = (135, 170)
WEAK_ORANGE_MAX = 25

def _hue_modes(h, sel, max_modes=BAND_MAX_HUE_MODES):
    """Extracts top peak hue values from saturated pixels, spaced 25 degrees apart."""
    if not sel.any():
        return []
    hist = np.bincount(h[sel].ravel(), minlength=180).astype(float)
    modes = []
    for _ in range(max_modes):
        m = int(np.argmax(hist))
        if hist[m] < 500:
            break
        modes.append(m)
        # Suppress surrounding histogram bin range
        idx = np.arange(180)
        circ = np.minimum(np.abs(idx - m), 180 - np.abs(idx - m))
        hist[circ <= 25] = 0
    return modes

def _band_for_hue(crop, hsv, hue_mode):
    """
    Isolates the largest connected blob of saturated pixels near hue_mode, 
    and crops a deskewed band aligned to the blob's long axis.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat_bright = (s > cfg.SASH_S_THRESH) & (v > cfg.SASH_V_THRESH)
    dh = np.abs(h.astype(int) - hue_mode)
    mask = (sat_bright & (np.minimum(dh, 180 - dh) <= BAND_HUE_TOL)).astype(np.uint8)
    
    # Smooth the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, None
        
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < BAND_MIN_AREA_FRAC2 * crop.shape[0] * crop.shape[1]:
        return None, None
        
    blob = (labels == big).astype(np.uint8)
    ys, xs = np.where(blob > 0)
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    (cx, cy), (rw, rh), ang = rect
    
    if rw == 0 or rh == 0:
        return None, None
        
    if rw < rh:
        rw, rh = rh, rw
        ang += 90.0
        
    # Rotate & Crop
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    H, W = crop.shape[:2]
    diag = int(np.ceil(np.hypot(H, W)))
    M[0, 2] += diag / 2 - cx
    M[1, 2] += diag / 2 - cy
    
    rot = cv2.warpAffine(crop, M, (diag, diag), flags=cv2.INTER_AREA)
    rblob = cv2.warpAffine(blob * 255, M, (diag, diag)) > 127
    
    pad = int(0.15 * rh)
    x0 = max(0, int(diag / 2 - rw / 2 - pad))
    x1 = min(diag, int(diag / 2 + rw / 2 + pad))
    y0 = max(0, int(diag / 2 - rh / 2 - pad))
    y1 = min(diag, int(diag / 2 + rh / 2 + pad))
    
    band = rot[y0:y1, x0:x1]
    bblob = rblob[y0:y1, x0:x1]
    
    if band.size == 0 or not bblob.any():
        return None, None
        
    return band, bblob

def _band_ocr_texts(band, bblob):
    """OCR binarization sweeps on the cropped band."""
    bhsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    bs, bv = bhsv[:, :, 1], bhsv[:, :, 2]

    imgs = []
    # 1. White text mask
    tp = ((bs < 70) & (bv > 140) & bblob).astype(np.uint8)
    n, ll, st, _ = cv2.connectedComponentsWithStats(tp, 8)
    keep = np.zeros_like(tp)
    for i in range(1, n):
        if 150 <= st[i, cv2.CC_STAT_AREA] <= 200000:
            keep[ll == i] = 1
    if keep.sum() >= 400:
        k = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        imgs.append(("textmask", (255 - k * 255).astype(np.uint8)))

    # 2. Green and Grayscale Otsu sweeps at native and downscaled heights
    for cname, ch in (("green", band[:, :, 1]), ("gray", cv2.cvtColor(band, cv2.COLOR_BGR2GRAY))):
        med = int(np.median(ch[bblob]))
        chm = np.where(bblob, ch, med).astype(np.uint8)
        for hgt in BAND_OCR_HEIGHTS:
            sc = min(1.0, hgt / chm.shape[0])
            small = cv2.resize(chm, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else chm
            small = cv2.GaussianBlur(small, (3, 3), 0)
            _, t = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            imgs.append((f"{cname}-h{hgt}", t))
            imgs.append((f"{cname}-h{hgt}-inv", 255 - t))

    for name, img in imgs:
        if max(img.shape) > 1400:
            sc = 1400 / max(img.shape)
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        # Try both direct and inverted orientations
        for rot_tag, im2 in (("0", img), ("180", cv2.rotate(img, cv2.ROTATE_180))):
            for psm in (7, 6):
                try:
                    text = pytesseract.image_to_string(im2, config=f"--psm {psm}")
                    yield f"band-{name}/{rot_tag}/psm{psm}", text
                except Exception as e:
                    print(f"[banner] OCR Exception: {str(e)}")
                    yield f"band-{name}/{rot_tag}/psm{psm}", ""

def _weak_band_evidence(texts, hue_mode):
    """Heuristic logic to match OCR drafts under tight hue constraint."""
    tokens = set()
    for t in texts:
        tokens.update(re.findall(r"[a-z]+", t.lower()))
    if WEAK_MAGENTA[0] <= hue_mode <= WEAK_MAGENTA[1]:
        if any(tok.startswith("plu") and len(tok) >= 3 for tok in tokens):
            return "maxi_plush"
    if hue_mode <= WEAK_ORANGE_MAX or hue_mode >= 170:
        if any(tok.startswith("grav") for tok in tokens):
            return "gravite"
    return None

def read_banner_fast(frame_bgr):
    """Executes Stage A analysis. Returns (sku, raw_text, details_list)."""
    tried = []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    sel = (s > cfg.SASH_S_THRESH) & (v > cfg.SASH_V_THRESH)
    if sel.sum() < 2000:
        return None, "", tried

    hue_modes = _hue_modes(hsv[:, :, 0], sel)
    bands = []
    for hm in hue_modes:
        band, bblob = _band_for_hue(frame_bgr, hsv, hm)
        if band is not None:
            bands.append((hm, band, bblob))

    # Pass 1: Cheap Visual Template Matching
    for hm, band, bblob in bands:
        for gal_sku, templates in get_galleries_for_hue(hm):
            try:
                inliers, tname = match_score(band, templates)
                tried.append(("C", f"template-{gal_sku}-hue{hm}-inliers{inliers}"))
                if inliers >= MIN_INLIERS:
                    return gal_sku, f"<visual template match: {inliers} inliers vs {tname}>", tried
            except Exception as e:
                print(f"[banner] Visual template match failed: {str(e)}")

    # Pass 2: OCR on band
    for hm, band, bblob in bands:
        texts = []
        for vname, text in _band_ocr_texts(band, bblob):
            tried.append(("A", vname))
            texts.append(text)
            sku = _normalize_and_disambiguate(text)
            if sku is not None:
                return sku, text, tried
        
        weak = _weak_band_evidence(texts, hm)
        if weak is not None:
            tried.append(("A", f"weak-evidence-hue{hm}"))
            concat = " | ".join(t.strip()[:30] for t in texts if t.strip())[:110]
            return weak, concat, tried

    return None, "", tried

# ==============================================================================
# Stage B: Fallback (Time-Budgeted Sweep OCR)
# ==============================================================================
BANNER_SWEEP_MAX_SIDE = 1600
BANNER_SWEEP_STEP_DEG = 15
BANNER_ADAPTIVE_BLOCK = 35
BANNER_ADAPTIVE_C = 15

def read_banner_sweep(frame_bgr, step_deg=BANNER_SWEEP_STEP_DEG, max_side=BANNER_SWEEP_MAX_SIDE, time_budget_s=15.0):
    """
    Fallback method: downscales the frame, rotates it in increments, 
    and tests multiple threshold/channel OCR passes within a time budget.
    """
    t_start = time.time()
    h0, w0 = frame_bgr.shape[:2]
    scale = max_side / max(h0, w0)
    if scale < 1.0:
        work_img = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        work_img = frame_bgr

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blue_ch = work_img[:, :, 0]
    tried = []

    def over_budget():
        return time_budget_s is not None and (time.time() - t_start) > time_budget_s

    angles = [0] + [a for a in range(step_deg, 360, step_deg)]
    for angle in angles:
        if over_budget():
            tried.append((angle, "BUDGET_EXHAUSTED"))
            break

        # Rotate images
        rot_gray = gray if angle == 0 else _rotate_bound(gray, angle)
        rot_blue = blue_ch if angle == 0 else _rotate_bound(blue_ch, angle)

        # Tier 1: Grayscale + Otsu
        _, t1 = cv2.threshold(rot_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = pytesseract.image_to_string(t1)
            tried.append((angle, "gray-otsu"))
            sku = _normalize_and_disambiguate(text)
            if sku is not None:
                return sku, angle, text.replace("\n", " ").strip(), tried
        except Exception as e:
            print(f"[banner] Sweep Tier 1 Exception: {str(e)}")

        if over_budget():
            tried.append((angle, "BUDGET_EXHAUSTED"))
            break

        # Tier 2: Grayscale + Adaptive Thresholding
        t2 = cv2.adaptiveThreshold(rot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, BANNER_ADAPTIVE_BLOCK, BANNER_ADAPTIVE_C)
        try:
            text = pytesseract.image_to_string(t2)
            tried.append((angle, "gray-adaptive"))
            sku = _normalize_and_disambiguate(text)
            if sku is not None:
                return sku, angle, text.replace("\n", " ").strip(), tried
        except Exception as e:
            print(f"[banner] Sweep Tier 2 Exception: {str(e)}")

        if over_budget():
            tried.append((angle, "BUDGET_EXHAUSTED"))
            break

        # Tier 3: Blue channel + Otsu + Sparse PSM
        _, t3 = cv2.threshold(rot_blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = pytesseract.image_to_string(t3, config="--psm 11")
            tried.append((angle, "blue-otsu-psm11"))
            sku = _normalize_and_disambiguate(text)
            if sku is not None:
                return sku, angle, text.replace("\n", " ").strip(), tried
        except Exception as e:
            print(f"[banner] Sweep Tier 3 Exception: {str(e)}")

    return None, None, "", tried

# ==============================================================================
# Public API Entry Point
# ==============================================================================
def read_banner(frame_bgr, step_deg=BANNER_SWEEP_STEP_DEG, max_side=BANNER_SWEEP_MAX_SIDE, time_budget_s=20.0):
    """
    Main entry point: Runs fast Stage A, and if unsuccessful, falls back to the Stage B sweep.
    Returns: (sku, rotation_angle_or_None, raw_text, details_tried_list)
    """
    t0 = time.time()
    
    # 1. Run Stage A (Isolate, visual matching, and local band OCR)
    sku, text, tried_a = read_banner_fast(frame_bgr)
    if sku is not None:
        return sku, None, text.replace("\n", " ").strip(), tried_a

    # Calculate remaining time budget
    elapsed = time.time() - t0
    remaining_budget = max(1.0, time_budget_s - elapsed)

    # 2. Fall back to Stage B sweep
    sku, angle, text, tried_b = read_banner_sweep(
        frame_bgr, step_deg=step_deg, max_side=max_side, time_budget_s=remaining_budget
    )
    
    return sku, angle, text, tried_a + tried_b
