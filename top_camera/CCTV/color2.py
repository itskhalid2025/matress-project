#!/usr/bin/env python3
"""
============================================================
FILE: cctv_bestdimension_color3.py
PATH: top_camera/global_shutter/cctv_bestdimension_color3.py
============================================================
DESCRIPTION:
  Live Web Streaming & Measurement Rig powered by a CCTV RTSP
  camera stream, using PURE COLOR / CONTOUR based detection
  (no neural network) to locate and measure a mattress/box
  against a contrasting background.

CHANGELOG vs cctv_bestdimension_color2.py:
  - FIXED: single physical box getting split into two separate
    contours (seen in the wild as a "green" box and a "blue"
    box both showing the same printed "FRAGILE / KEEP DRY /
    THIS END UP" text). Root cause: a strong internal edge
    (printed warning text block, tape seam, shadow line) on
    the box produces a dense Canny edge cluster that the
    15x15 morphological close (at 0.25 working scale, ~60px
    in the real frame) wasn't wide enough to bridge. The
    largest-first contour walk then picked up only one half
    of the box, leaving the other half to be reported as a
    separate object.
  - Added _merge_nearby_contours(): merges contours whose
    bounding boxes lie within gap_px of each other (at the
    0.25 working scale) BEFORE the fill/select stage. This is
    robust regardless of *why* a box got split (text, seam,
    shadow) and doesn't require blindly enlarging the close
    kernel, which risks re-fusing the object to the background
    (the exact bug the OSD-band fix in _v2 solved). Genuinely
    separate physical boxes that are far enough apart keep
    their own contours.
  - gap_px defaults to 40 on the 0.25-scale working image
    (~160px in a 1920-wide real frame). Tune via
    CONTOUR_MERGE_GAP_PX below if boxes still split, or if two
    genuinely separate objects start getting merged into one
    (lower the value).
============================================================
"""

import sys
import os
import time
import threading
import re
import math
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# Force TCP transport for RTSP
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Import the banner OCR engine from the top_camera module if available
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if TOP_CAM_DIR not in sys.path:
    sys.path.insert(0, TOP_CAM_DIR)

try:
    from banner import read_banner_fast
    import config as _cfg
    _cfg.SASH_S_THRESH = 60
    _cfg.SASH_V_THRESH = 60
    BANNER_OCR_AVAILABLE = True
    print("[cctv_stream] banner.py OCR engine loaded")
except Exception as _be:
    BANNER_OCR_AVAILABLE = False
    print(f"[cctv_stream] banner.py not available: {_be}")

app = Flask(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
CAMERA_DISTANCE_CM = 200.0     # Fixed camera distance to mattress plane
KNOWN_REF_WIDTH_CM = 47.0      # Physical reference horizontal width
KNOWN_REF_HEIGHT_CM = 46.0     # Physical reference vertical length

INVERT_MASK = True               # True: mattress is BRIGHT on a darker background
MIN_RECTANGULARITY = 0.35        # lowered to allow thin or irregular objects
MIN_MATTRESS_AREA_RATIO = 0.015  # lowered to 1.5% to detect the thin blue foam object
MAX_MATTRESS_AREA_RATIO = 0.95   # raised to 95% to allow detecting the large background cardboard
MAX_ANGLE_DEG = 89.0             # allow any angle
MIN_CONTOUR_AREA_PX = 3000       # low floor, true filtering done by area ratio
MAX_OBJECTS = 10                 # max number of objects to detect and annotate

# Camera's own OSD band (timestamp / label baked into the raw RTSP frame).
# Blanked out before edge detection so it can never seed or bridge a false
# contour to the frame border. Tune with DEBUG_SHOW_RAW_MASK=True if your
# camera's overlay sits somewhere other than the very top/bottom strip.
OSD_TOP_BAND_PCT = 0.07
OSD_BOTTOM_BAND_PCT = 0.07

# Gap (in pixels, measured on the 0.25-scale working image used inside
# _get_raw_mask) within which two separate contours are CANDIDATES to
# be merged as pieces of the SAME physical object. Whether they
# actually get merged is then decided by _has_strong_boundary_edge()
# below, which reads the real Sobel gradient in the gap -- not just
# distance. ~40px here corresponds to ~160px in a real 1920-wide frame.
CONTOUR_MERGE_GAP_PX = 40

# Sobel gradient magnitude (0-255 scale image, ksize=3) above which a
# pixel counts as "on a strong edge" when scanning the gap between two
# contour fragments for a real object boundary (a seam between two
# separate boxes) vs. weak/no edge (fragmentation of one object).
GAP_EDGE_GRAD_THRESH = 35
# Fraction of the gap's overlap-length that must be covered by strong
# edge pixels for the gap to be treated as a real boundary (don't
# merge). Below this fraction, the gap is treated as internal to a
# single object (merge).
GAP_EDGE_FRAC_THRESH = 0.5

# Half-width, in ORIGINAL full-resolution pixels, of the band searched
# around each side of a detected bounding box to snap it onto the true
# object edge (see _snap_box_to_gradient). The 0.25-scale mask pipeline
# is fast but its box can be several real-frame pixels loose or tight;
# this recovers precision using the full-res Sobel gradient.
EDGE_SNAP_SEARCH_PX = 20

# GrabCut refinement (see _grabcut_refine): grows the detected box into
# regions where the object and background have near-identical
# color/brightness -- e.g. a light-grey box against a light-grey wall.
# Canny/Sobel-based edge detection CANNOT recover a boundary that isn't
# there in the pixels; GrabCut instead uses per-region color statistics
# to keep expanding the foreground until it hits a genuine color/texture
# change, which is what these low-contrast edges actually need.
USE_GRABCUT_REFINE = True
GRABCUT_PAD_FRAC = 0.15     # how far outside the coarse box to search
GRABCUT_ITERS = 4

DETECT_EVERY_N_FRAMES = 1        # color detection is cheap -- safe to run every frame

SMOOTH_HISTORY = 7                # frames of rolling-median history
SMOOTH_MAX_DEVIATION = 0.35       # reject outlier readings (>35% deviation)
SMOOTH_WARMUP = 3                 # frames before measurement reported stable

RTSP_RECONNECT_AFTER_FAILURES = 60
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)

pixels_per_cm_x = 10.0      # Horizontal pixels per cm (calibrated via UI)
pixels_per_cm_y = 10.0      # Vertical pixels per cm   (calibrated via UI)
edge_correction = 1.0
_calib_lock = threading.Lock()

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Colors for annotating multiple objects (BGR)
_OBJECT_COLORS = [
    (0, 255, 0),    # green
    (255, 165, 0),  # orange
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 0, 0),    # blue
    (0, 165, 255),  # dark orange
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
    (255, 128, 0),  # sky blue
    (128, 255, 0),  # chartreuse
]


# ==============================================================================
# Color / contour segmentation (this replaces the YOLO stage entirely)
# ==============================================================================
def _normalize_angle(angle):
    a = angle % 90
    if a > 45:
        a -= 90
    return a


def compute_gradient_features(bgr_roi):
    """Sobel-based texture features for a region."""
    if bgr_roi.size == 0:
        return {"mean_gradient": 0.0, "gradient_std": 0.0, "edge_density": 0.0}
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return {
        "mean_gradient": float(np.mean(mag)),
        "gradient_std": float(np.std(mag)),
        "edge_density": float(np.mean(mag > 40)),  # fraction of pixels with a strong edge
    }

def compute_color_features(bgr_roi):
    if bgr_roi.size == 0:
        return {"hue_mean": 0.0, "sat_mean": 0.0, "val_mean": 0.0}
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "hue_mean": float(np.mean(h)),
        "sat_mean": float(np.mean(s)),
        "val_mean": float(np.mean(v)),
    }

DEBUG_SHOW_RAW_MASK = False  # Set to True to visualize the raw binary detection mask

# --- HYSTERESIS GLOBALS ---
MAX_MISSED_FRAMES = 15
_missed_frames = 0
_last_primary_box = None
_last_primary_contour = None
_last_primary_features = None
_last_primary_w_cm = None
_last_primary_h_cm = None
_last_primary_stable = False


def _gap_rect_between(b1, b2):
    """Return (rect, axis) describing the empty gap between two bounding
    boxes, where rect = (x0, y0, x1, y1) and axis is 'vertical' if the
    boxes sit side-by-side (so the gap is a vertical strip) or
    'horizontal' if stacked. Returns (None, None) if the boxes overlap
    directly (no gap to inspect -- they should just be merged)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    if x1 + w1 <= x2:
        gx0, gx1 = x1 + w1, x2
    elif x2 + w2 <= x1:
        gx0, gx1 = x2 + w2, x1
    else:
        gx0 = gx1 = None
    gy0 = max(y1, y2)
    gy1 = min(y1 + h1, y2 + h2)
    if gx0 is not None and gy1 > gy0:
        return (gx0, gy0, gx1, gy1), 'vertical'

    if y1 + h1 <= y2:
        gy0b, gy1b = y1 + h1, y2
    elif y2 + h2 <= y1:
        gy0b, gy1b = y2 + h2, y1
    else:
        gy0b = gy1b = None
    gx0b = max(x1, x2)
    gx1b = min(x1 + w1, x2 + w2)
    if gy0b is not None and gx1b > gx0b:
        return (gx0b, gy0b, gx1b, gy1b), 'horizontal'

    return None, None


def _has_strong_boundary_edge(gray, gap_rect, axis,
                               frac_thresh=GAP_EDGE_FRAC_THRESH,
                               grad_thresh=GAP_EDGE_GRAD_THRESH):
    """Read the real Sobel gradient inside the gap between two contour
    fragments to tell apart two different situations that look
    identical if you only measure pixel distance:

      - A real seam between two separate physical objects (e.g. two
        boxes pushed together) -- this shows up as a strong, CONTINUOUS
        edge line running most of the length of the gap.
      - Fragmentation of a single object by an internal edge (printed
        text, tape, a shadow) that happened to also break the outer
        contour -- the gap here has weak or patchy gradient, because
        it's still the same surface/material on both sides.

    Returns True only for the first case (real boundary -- do not merge).
    """
    x0, y0, x1, y1 = gap_rect
    h_img, w_img = gray.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)
    if x1 <= x0 or y1 <= y0:
        return False

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    if axis == 'vertical':
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        mag = np.abs(gx)
        line_max = mag.max(axis=1)  # strongest edge per row
    else:
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.abs(gy)
        line_max = mag.max(axis=0)  # strongest edge per column

    if line_max.size == 0:
        return False
    strong_frac = float(np.mean(line_max > grad_thresh))
    return strong_frac > frac_thresh


def _merge_nearby_contours(contours, gray, gap_px=CONTOUR_MERGE_GAP_PX):
    """Merge contours whose bounding boxes are within gap_px of each
    other AND whose gap does NOT contain a strong, continuous gradient
    edge (see _has_strong_boundary_edge). Handles a single physical box
    getting split into multiple contours by a strong internal edge
    (printed warning text, a tape seam, a shadow) that the Canny+close
    stage couldn't fully bridge -- without this, the largest-first
    contour walk in _all_valid_contours() would only ever pick up one
    fragment, leaving the rest to be reported as a spurious second
    object.

    Critically, this does NOT merge two genuinely separate physical
    objects sitting close together (e.g. two boxes pushed side by
    side): the real seam between them shows up as a strong continuous
    gradient edge, which _has_strong_boundary_edge detects, so those
    contours are left distinct even though they're within gap_px.
    """
    if len(contours) <= 1:
        return contours

    boxes = [cv2.boundingRect(c) for c in contours]

    def _mergeable(b1, b2, gap):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        near = not (x1 - gap > x2 + w2 or x2 - gap > x1 + w1 or
                    y1 - gap > y2 + h2 or y2 - gap > y1 + h1)
        if not near:
            return False
        gap_rect, axis = _gap_rect_between(b1, b2)
        if gap_rect is None:
            return True  # boxes already touch/overlap -- same object
        if _has_strong_boundary_edge(gray, gap_rect, axis):
            return False  # real seam -- keep as separate objects
        return True

    groups = [[i] for i in range(len(contours))]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(_mergeable(boxes[a], boxes[b], gap_px)
                       for a in groups[i] for b in groups[j]):
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break

    merged = []
    for g in groups:
        pts = np.vstack([contours[i] for i in g])
        merged.append(cv2.convexHull(pts))
    return merged


def _snap_box_to_gradient(gray_full, x, y, w, h, search_px=EDGE_SNAP_SEARCH_PX):
    """Refine a coarse bounding box to the true object edges using the
    Sobel gradient of the FULL-RESOLUTION frame (the mask/contour
    pipeline runs on a 0.25-scale image for speed, so its box can be a
    few real-frame pixels loose or tight). For each of the 4 sides,
    search a small band around the current position and snap to the
    column/row with the strongest total gradient energy -- i.e. the
    precise physical edge of the object, not the coarse mask boundary.
    This directly improves measurement accuracy in cm.
    """
    H, W = gray_full.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return x, y, w, h

    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    mag_x = np.abs(gx)  # vertical edges -- used to snap left/right sides
    mag_y = np.abs(gy)  # horizontal edges -- used to snap top/bottom sides

    def best_col(cx, lo, hi, y0_, y1_):
        lo = max(0, lo); hi = min(W - 1, hi)
        if hi <= lo:
            return cx
        band = mag_x[y0_:y1_, lo:hi + 1]
        if band.size == 0:
            return cx
        energy = band.sum(axis=0)
        return lo + int(np.argmax(energy))

    def best_row(cy, lo, hi, x0_, x1_):
        lo = max(0, lo); hi = min(H - 1, hi)
        if hi <= lo:
            return cy
        band = mag_y[lo:hi + 1, x0_:x1_]
        if band.size == 0:
            return cy
        energy = band.sum(axis=1)
        return lo + int(np.argmax(energy))

    left = best_col(x0, x0 - search_px, x0 + search_px, y0, y1)
    right = best_col(x1, x1 - search_px, x1 + search_px, y0, y1)
    top = best_row(y0, y0 - search_px, y0 + search_px, x0, x1)
    bottom = best_row(y1, y1 - search_px, y1 + search_px, x0, x1)

    if right <= left:
        left, right = x0, x1
    if bottom <= top:
        top, bottom = y0, y1

    return left, top, right - left, bottom - top


def _grabcut_refine(bgr, x, y, w, h, pad_frac=GRABCUT_PAD_FRAC, iters=GRABCUT_ITERS):
    """Grow the coarse edge-derived box into a precise object mask using
    GrabCut, seeded from the box itself.

    Pure edge detection (Canny/Sobel) cannot recover a boundary where
    the object and background have near-identical color/brightness --
    there is no gradient to find because the pixels genuinely don't
    change much there. GrabCut instead models the foreground/background
    color distributions and iteratively grows the foreground region
    until it hits a real color/texture change, which is the correct
    tool for exactly this failure mode (a light-grey box against a
    light-grey wall/background) rather than continuing to tune Canny
    thresholds or morphology kernel sizes.

    Seeding:
      - Sure foreground: a shrunk-in core of the detected box (very
        likely to be genuinely inside the object).
      - Probable foreground: the rest of the detected box.
      - Everything in the padded search window outside the box:
        probable background, letting GrabCut grow outward if the color
        statistics support it.
    """
    H, W = bgr.shape[:2]
    pad_x = max(10, int(w * pad_frac))
    pad_y = max(10, int(h * pad_frac))
    gx0 = max(0, x - pad_x)
    gy0 = max(0, y - pad_y)
    gx1 = min(W, x + w + pad_x)
    gy1 = min(H, y + h + pad_y)

    roi = bgr[gy0:gy1, gx0:gx1]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return x, y, w, h

    mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)

    # Probable foreground = the original detected box footprint
    px0, py0 = max(0, x - gx0), max(0, y - gy0)
    px1, py1 = min(roi.shape[1], x - gx0 + w), min(roi.shape[0], y - gy0 + h)
    if px1 > px0 and py1 > py0:
        mask[py0:py1, px0:px1] = cv2.GC_PR_FGD

    # Sure foreground = a shrunk-in core of that box
    fx0 = px0 + int((px1 - px0) * 0.2)
    fy0 = py0 + int((py1 - py0) * 0.2)
    fx1 = px0 + int((px1 - px0) * 0.8)
    fy1 = py0 + int((py1 - py0) * 0.8)
    if fx1 > fx0 and fy1 > fy0:
        mask[fy0:fy1, fx0:fx1] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return x, y, w, h

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return x, y, w, h
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.5 * w * h:
        # GrabCut collapsed to something much smaller than the original
        # detection -- likely a bad seed on this frame; keep the
        # edge-based box rather than trust a degenerate result.
        return x, y, w, h
    rx, ry, rw, rh = cv2.boundingRect(c)
    return gx0 + rx, gy0 + ry, rw, rh


def _get_raw_mask(bgr):
    """Structural-edge segmentation with the camera's own OSD band
    blanked out first, so its full-width timestamp/label text can never
    seed or bridge a false contour to the frame border.

    VERIFIED against a real captured frame from this exact camera: the
    previous approach (broad HSV color fill + Canny) fused the box to
    the frame border via the camera's baked-in timestamp/label text --
    those overlays create solid full-width edges near the top and
    bottom of every raw frame, and any morphological closing bridges
    right through them, producing area_ratio ~= 1.0 with all 4 edges
    touched every time (permanent "Searching Target..."). The broad
    color mask was a second, independent problem: the light floor and
    striped background shared enough hue/saturation with the tan box
    to light up just as brightly as the box itself.

    Fix: blank the OSD bands, drop the color mask, rely on tuned edges
    + modest morphology, then merge nearby contour fragments so a
    single box with a strong internal edge (printed text, tape seam)
    doesn't get reported as two separate objects.
    """
    h_orig, w_orig = bgr.shape[:2]
    small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25)
    sh, sw = small.shape[:2]

    # Structural edges
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)

    # Blank the camera's own OSD band (top timestamp, bottom label) so
    # it can never seed or bridge a contour to the frame border.
    roi_mask = np.zeros_like(edges)
    y0 = int(sh * OSD_TOP_BAND_PCT)
    y1 = int(sh * (1.0 - OSD_BOTTOM_BAND_PCT))
    roi_mask[y0:y1, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    # Modest dilate/close -- enough to bridge small gaps in the outline
    # (tape seams, faint contrast patches), not enough to fuse the
    # object with the background.
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    closed = cv2.bitwise_and(closed, roi_mask)  # keep clipped to the OSD-safe region

    # Find raw contour fragments, drop tiny noise, then merge fragments
    # that likely belong to the same physical object (e.g. a box split
    # by a strong internal edge such as printed warning text).
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
    cnts = _merge_nearby_contours(cnts, gray, gap_px=CONTOUR_MERGE_GAP_PX)

    # Fill the internal area to seal the block(s)
    filled = np.zeros_like(closed)
    for c in cnts:
        cv2.drawContours(filled, [cv2.convexHull(c)], -1, 255, -1)

    mask_full = cv2.resize(filled, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return mask_full


def _all_valid_contours(mask, bgr, frame_area):
    """Walk contours largest-first and return the first one that passes
    the size + border-bleed sanity checks, instead of only ever checking
    the single largest contour (which, if it was a background-fused
    blob, killed detection entirely with no fallback).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h_img, w_img = bgr.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = max(w * h, 1e-6)

        # Strict Area Filter: object must occupy a sane fraction of the
        # frame -- eliminates tiny noise (foam tube) and full-frame
        # background-fused blobs alike.
        area_ratio = rect_area / frame_area
        if area_ratio < 0.10 or area_ratio > MAX_MATTRESS_AREA_RATIO:
            continue

        # Border-bleed guard: if the box hugs 3+ frame edges it's almost
        # certainly a background-fused blob rather than the real object
        # -- skip it and try the next-largest contour instead of giving
        # up on the whole frame.
        touches = sum([
            x <= 2,
            y <= 2,
            (x + w) >= w_img - 2,
            (y + h) >= h_img - 2,
        ])
        if touches >= 3:
            continue

        # Extract ROI for gradient classification features
        roi_x, roi_y = max(0, x), max(0, y)
        roi_w, roi_h = min(w, w_img - roi_x), min(h, h_img - roi_y)
        roi = bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        features = compute_gradient_features(roi)
        return [(cv2.convexHull(c), features)]

    return []


class MeasurementSmoother:
    def __init__(self, history=SMOOTH_HISTORY, max_deviation=SMOOTH_MAX_DEVIATION, warmup=SMOOTH_WARMUP):
        self.history = history
        self.max_deviation = max_deviation
        self.warmup = warmup
        self.w_buf = deque(maxlen=history)
        self.h_buf = deque(maxlen=history)

    def reset(self):
        self.w_buf.clear()
        self.h_buf.clear()

    def update(self, w_cm, h_cm):
        if w_cm is None or h_cm is None or w_cm <= 0 or h_cm <= 0:
            return None, None, False

        if len(self.w_buf) >= self.warmup:
            med_w = float(np.median(self.w_buf))
            med_h = float(np.median(self.h_buf))
            if med_w > 0 and med_h > 0:
                dev_w = abs(w_cm - med_w) / med_w
                dev_h = abs(h_cm - med_h) / med_h
                if dev_w > self.max_deviation or dev_h > self.max_deviation:
                    return round(med_w, 1), round(med_h, 1), True

        self.w_buf.append(w_cm)
        self.h_buf.append(h_cm)
        stable = len(self.w_buf) >= self.warmup
        return round(float(np.median(self.w_buf)), 1), round(float(np.median(self.h_buf)), 1), stable


_smoother = MeasurementSmoother()


def process_frame_tight_geometry(img, px_cm_x, px_cm_y, distance_cm=200.0, frame_index=0):
    """
    Detects objects and draws YOLO-style axis-aligned bounding boxes.
    """
    if img is None or img.size == 0:
        return None, None, img, None, False

    annotated = img.copy()
    h_orig, w_orig = img.shape[:2]
    frame_area = float(w_orig * h_orig)

    global _missed_frames, _last_primary_box, _last_primary_contour
    global _last_primary_features, _last_primary_w_cm, _last_primary_h_cm, _last_primary_stable

    try:
        # Generate the unified, lighting-invariant structural mask
        mask = _get_raw_mask(img)

        if DEBUG_SHOW_RAW_MASK:
            # Short-circuit the pipeline to output the raw binary mask for debugging
            annotated = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(annotated, "DEBUG MASK ACTIVE (Toggle DEBUG_SHOW_RAW_MASK=False to disable)", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            return None, None, annotated, None, False

        # Find the best valid contour (largest-first, with fallback)
        all_found = _all_valid_contours(mask, img, frame_area)

        if not all_found:
            _missed_frames += 1
            if _missed_frames < MAX_MISSED_FRAMES and _last_primary_contour is not None:
                # HYSTERESIS: Use the last known good target to bridge temporary detection gaps
                all_found = [(_last_primary_contour, _last_primary_features)]
            else:
                # No qualifying contours after timeout -- report searching
                _smoother.reset()
                _last_primary_box = None
                cv2.putText(annotated, "Searching Target...", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                return None, None, annotated, None, False
        else:
            _missed_frames = 0  # Reset timeout

        primary_w_cm = None
        primary_h_cm = None
        primary_box = None
        primary_stable = False

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for idx, (contour, features) in enumerate(all_found):
            x, y, w_px, h_px = cv2.boundingRect(contour)
            # Snap the coarse mask-derived box onto the true full-resolution
            # gradient edge for a precise, less noisy measurement.
            x, y, w_px, h_px = _snap_box_to_gradient(gray_full, x, y, w_px, h_px)
            if USE_GRABCUT_REFINE:
                # Then grow into any remaining low-contrast region (object
                # and background too similar in color/brightness for edge
                # detection to see) using color-based GrabCut segmentation.
                x, y, w_px, h_px = _grabcut_refine(img, x, y, w_px, h_px)

            color = _OBJECT_COLORS[idx % len(_OBJECT_COLORS)]
            is_primary = (idx == 0)

            # Determine label based on gradient edge density
            edge_dens = features['edge_density']
            if edge_dens > 0.11:
                label = "CARDBOARD"
            else:
                label = "MATTRESS/FOAM"

            if is_primary:
                label = f"*{label}*"

            # Draw YOLO-style straight bounding box
            thickness = 3 if is_primary else 2
            cv2.rectangle(annotated, (x, y), (x + w_px, y + h_px), color, thickness)

            # Draw semi-transparent fill ("gradient overlay") to match the reference picture
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, y), (x + w_px, y + h_px), color, -1)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            # Compute dimensions in cm
            if px_cm_x > 0 and px_cm_y > 0:
                w_cm_raw = (w_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_x * edge_correction
                h_cm_raw = (h_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_y * edge_correction

                w_cm_obj = round(w_cm_raw, 1)
                h_cm_obj = round(h_cm_raw, 1)
                w_in_obj = round(w_cm_obj / 2.54, 1)
                h_in_obj = round(h_cm_obj / 2.54, 1)

                if is_primary:
                    nW_cm, nH_cm, stable = _smoother.update(w_cm_obj, h_cm_obj)
                    if nW_cm is not None:
                        primary_w_cm = nW_cm
                        primary_h_cm = nH_cm
                        primary_stable = stable
                    elif _last_primary_w_cm is not None:
                        # Fallback to last known smoothed dims during hysteresis
                        primary_w_cm = _last_primary_w_cm
                        primary_h_cm = _last_primary_h_cm
                        primary_stable = _last_primary_stable

                    if primary_w_cm is not None:
                        w_cm_obj = primary_w_cm
                        h_cm_obj = primary_h_cm

                    # Save state for temporal hysteresis
                    _last_primary_contour = contour
                    _last_primary_features = features
                    _last_primary_w_cm = primary_w_cm
                    _last_primary_h_cm = primary_h_cm
                    _last_primary_stable = primary_stable
                    primary_box = (x, y, x + w_px, y + h_px)
                    _last_primary_box = primary_box

                diag_cm = round(math.hypot(w_cm_obj, h_cm_obj), 1)

                # Draw YOLO-style solid label background above box for dimensions and gradient
                suffix = "" if (is_primary and primary_stable) or not is_primary else " ~"
                full_label = f"{label} | Grad: {edge_dens:.2f} | W:{w_cm_obj}cm H:{h_cm_obj}cm{suffix}"
                (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                # Make sure the label doesn't go off the top of the screen
                label_y_top = max(0, y - th - 10)
                label_y_bottom = max(th + 10, y)
                cv2.rectangle(annotated, (x, label_y_top), (x + tw + 10, label_y_bottom), color, -1)

                # Text with thin black outline for readability against bright colors
                cv2.putText(annotated, full_label, (x + 5, label_y_bottom - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, full_label, (x + 5, label_y_bottom - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw Diagonal label at bottom of box
                metrics_text = f'Diag: {diag_cm} cm'
                (mw, mh), _ = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x, y + h_px), (x + mw + 10, y + h_px + mh + 10), color, -1)
                cv2.putText(annotated, metrics_text, (x + 5, y + h_px + mh + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, metrics_text, (x + 5, y + h_px + mh + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Show total objects detected count
        obj_text = f"Objects: {len(all_found)}"
        cv2.putText(annotated, obj_text, (30, h_orig - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, obj_text, (30, h_orig - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        return primary_w_cm, primary_h_cm, annotated, primary_box, primary_stable

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[dimensions] Error during processing: {str(e)}")

    return None, None, annotated, None, False


# ---------------------------------------------------------------------------
# Background OCR Thread — runs read_banner_fast() decoupled from capture loop
# (unchanged from the YOLO rig -- detection method doesn't affect OCR)
# ---------------------------------------------------------------------------
_ocr_latest_frame = None
_ocr_frame_lock   = threading.Lock()
_ocr_result       = ""
_ocr_result_lock  = threading.Lock()

def _ocr_worker():
    global _ocr_result, _ocr_latest_frame
    while True:
        frame_to_process = None
        with _ocr_frame_lock:
            if _ocr_latest_frame is not None:
                frame_to_process = _ocr_latest_frame.copy()
                _ocr_latest_frame = None
        if frame_to_process is None:
            time.sleep(0.1)
            continue
        try:
            if BANNER_OCR_AVAILABLE:
                sku, raw_text, _ = read_banner_fast(frame_to_process)
                if sku:
                    label = sku.replace("_", " ").upper()
                elif raw_text and len(raw_text.strip()) > 2:
                    words = re.findall(r"[A-Za-z0-9]{3,}", raw_text)
                    label = " ".join(words).upper()[:40]
                else:
                    label = ""
                if label:
                    with _ocr_result_lock:
                        _ocr_result = label
                    print(f"[ocr_worker] Detected: {label}")
        except Exception as exc:
            print(f"[ocr_worker] {exc}")
        time.sleep(0.5)

_ocr_thread = threading.Thread(target=_ocr_worker, daemon=True)
_ocr_thread.start()


class StreamerServer:
    def __init__(self, width=1920, height=1080, fps=30, webcam_index=8):
        self.width = width
        self.height = height
        self.fps = fps
        self.webcam_index = webcam_index

        self.picam2 = None
        self.cap = None
        self.camera_type = "Offline"

        self.latest_jpeg = None
        self.latest_dims = {
            "width_cm": None, "width_in": None,
            "height_cm": None, "height_in": None,
            "ocr_text": "", "status": "Initializing",
            "camera_type": "Offline", "distance_cm": CAMERA_DISTANCE_CM
        }
        self.latest_ocr = ""
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0

        self.fps_counter = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()

        self._init_camera()

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # 1. Try RTSP Stream
        if isinstance(self.webcam_index, str) and any(self.webcam_index.startswith(proto) for proto in ["rtsp://", "rtmp://", "http://", "https://"]):
            try:
                print(f"[cctv_stream] Connecting to RTSP Stream: {self.webcam_index} using FFMPEG backend...")
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.camera_type = "RTSP CCTV"
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cctv_stream] Connected to RTSP camera successfully at {actual_w}x{actual_h}")
                    return
            except Exception as e:
                print(f"[cctv_stream] RTSP stream connection failed: {e}")

        # 2. Try USB Webcam
        try:
            idx = self.webcam_index if isinstance(self.webcam_index, int) else 0
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
            cap = cv2.VideoCapture(idx, backend)

            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap = cap
                self.camera_type = f"USB Cam (Index {idx})"
                return
        except Exception as e:
            print(f"[cctv_stream] USB webcam initialization failed: {e}")

        # 3. Fallback: Picamera2
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            self.picam2 = picam
            self.camera_type = "Picamera2 (IMX296)"
        except Exception as e:
            print(f"[cctv_stream] Picamera2 fallback failed: {e}")
            self.camera_type = "Offline"

    def _reconnect_rtsp(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        _smoother.reset()
        self._init_camera()

    def _worker_loop(self):
        global pixels_per_cm_x, pixels_per_cm_y
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        consecutive_failures = 0

        while self.running:
            raw_bgr = None
            if self.picam2:
                try:
                    rgb = self.picam2.capture_array()
                    raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    consecutive_failures = 0
                except Exception:
                    time.sleep(0.005)
                    continue
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    raw_bgr = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= RTSP_RECONNECT_AFTER_FAILURES:
                        print("[cctv_stream] RTSP stream unresponsive — attempting reconnect...")
                        self._reconnect_rtsp()
                        consecutive_failures = 0
                    time.sleep(0.005)
                    continue
            else:
                time.sleep(0.01)
                continue

            if raw_bgr is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            w_cm, h_cm, annotated, target_box, stable = process_frame_tight_geometry(
                raw_bgr, pixels_per_cm_x, pixels_per_cm_y,
                distance_cm=CAMERA_DISTANCE_CM, frame_index=self.frame_count,
            )

            if self.frame_count % 5 == 0:
                with _ocr_frame_lock:
                    global _ocr_latest_frame
                    _ocr_latest_frame = raw_bgr.copy()

            with _ocr_result_lock:
                self.latest_ocr = _ocr_result

            if self.latest_ocr:
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            fps_txt = f"FPS: {self.current_fps} | Distance: {CAMERA_DISTANCE_CM} cm"
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', annotated, jpeg_params)
            if ret:
                w_in = round(w_cm / 2.54, 1) if w_cm is not None else None
                h_in = round(h_cm / 2.54, 1) if h_cm is not None else None
                if w_cm is not None:
                    status = "Target Detected" if stable else "Stabilizing..."
                else:
                    status = "Searching Target..."
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_dims = {
                        "width_cm": w_cm,
                        "width_in": w_in,
                        "height_cm": h_cm,
                        "height_in": h_in,
                        "ocr_text": self.latest_ocr,
                        "status": status,
                        "camera_type": self.camera_type,
                        "distance_cm": CAMERA_DISTANCE_CM
                    }

            time.sleep(0.001)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_dims(self):
        with self.lock:
            return self.latest_dims

    def release(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap:
            self.cap.release()


streamer = StreamerServer(width=1920, height=1080, fps=30, webcam_index=DEFAULT_RTSP)


def generate_feed():
    last_sent = None
    while True:
        jpeg = streamer.get_jpeg()
        if jpeg is not None and jpeg != last_sent:
            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)

# ==============================================================================
# Web UI Dashboard
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CCTV Mattress Color-Contour Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: rgba(22, 28, 45, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --primary: #06b6d4;
            --magenta: #e81b84;
            --amber: #f59e0b;
            --green: #10b981;
            --text: #f8fafc;
            --dim: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); padding: 20px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }

        .header { width: 100%; max-width: 1200px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 16px 24px; background: var(--panel); border: 1px solid var(--border); border-radius: 16px; backdrop-filter: blur(12px); }
        .header h1 { font-size: 22px; font-weight: 800; }
        .header h1 span { color: var(--primary); }
        .camera-badge { color: var(--green); font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 12px; border-radius: 20px; background: rgba(16, 185, 129, 0.08); display: flex; align-items: center; gap: 6px; }

        .layout { width: 100%; max-width: 1200px; display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }

        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; backdrop-filter: blur(12px); }
        .video-box { width: 100%; aspect-ratio: 16/10; background: #000; border-radius: 14px; overflow: hidden; display: flex; justify-content: center; align-items: center; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .video-box img { width: 100%; height: 100%; object-fit: contain; }

        .stat-group { display: flex; flex-direction: column; gap: 14px; }
        .stat-card { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); padding: 14px; border-radius: 14px; }
        .stat-label { font-size: 11px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
        .stat-val { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--primary); }
        .stat-val.magenta { color: var(--magenta); }
        .stat-val.amber { color: var(--amber); font-size: 18px; font-weight: 700; word-break: break-word; }

        .calib-box { margin-top: 10px; padding-top: 15px; border-top: 1px solid var(--border); }
        .calib-input { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
        .calib-input input { background: #0f172a; border: 1px solid var(--border); color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 14px; }
        .calib-input button { background: var(--primary); color: #fff; border: none; padding: 12px; border-radius: 8px; font-weight: 700; cursor: pointer; }
        .calib-input button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📐 CCTV <span>Color-Contour Dimension Stream</span></h1>
        <div class="camera-badge" id="cameraBadge">● {{ camera_type }} ACTIVE</div>
    </div>

    <div class="layout">
        <div class="card">
            <div class="video-box">
                <img src="/video_feed" alt="Live Camera Stream">
            </div>
        </div>

        <div class="card">
            <div class="stat-group">
                <div class="stat-card">
                    <div class="stat-label">Live Width</div>
                    <div class="stat-val" id="valWidth">—</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Live Height / Length</div>
                    <div class="stat-val magenta" id="valHeight">—</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Detected Brand / OCR</div>
                    <div class="stat-val amber" id="valOcr">Reading...</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Camera Distance</div>
                    <div class="stat-val" style="color: #38bdf8;">{{ distance_cm }} cm</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Status</div>
                    <div class="stat-val" id="valStatus" style="font-size: 16px; color: var(--green);">Searching Target...</div>
                </div>

                <div class="calib-box">
                    <div class="stat-label">Dual-Axis Distance Calibration</div>
                    <div style="font-size: 12px; color: var(--dim); margin-bottom: 6px;">Preset: Known 47cm (W) x 46cm (H) @ 200cm</div>
                    <div class="calib-input">
                        <input type="number" id="knownW" value="47" placeholder="Known Width cm (47)">
                        <input type="number" id="knownH" value="46" placeholder="Known Length cm (46)">
                        <button onclick="calibrateBoth()">Calibrate Dual Axes</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateMetrics() {
            fetch('/api/dims')
                .then(res => res.json())
                .then(data => {
                    if (data.width_cm !== null) {
                        document.getElementById('valWidth').innerText = data.width_cm + ' cm (' + data.width_in + ' in)';
                    } else {
                        document.getElementById('valWidth').innerText = '—';
                    }

                    if (data.height_cm !== null) {
                        document.getElementById('valHeight').innerText = data.height_cm + ' cm (' + data.height_in + ' in)';
                    } else {
                        document.getElementById('valHeight').innerText = '—';
                    }

                    document.getElementById('valOcr').innerText = data.ocr_text ? data.ocr_text : 'No text detected';
                    document.getElementById('valStatus').innerText = data.status;
                })
                .catch(err => console.error("Metrics error:", err));
        }

        function calibrateBoth() {
            const knownW = document.getElementById('knownW').value;
            const knownH = document.getElementById('knownH').value;
            fetch(`/api/calibrate?known_width_cm=${knownW || '47'}&known_length_cm=${knownH || '46'}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Calibration Updated:\\nWidth Scale: ' + data.px_cm_x + ' px/cm\\nLength Scale: ' + data.px_cm_y + ' px/cm');
                    } else {
                        alert('❌ Calibration failed: ' + data.error);
                    }
                });
        }

        setInterval(updateMetrics, 500);
        updateMetrics();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        pixels_per_cm=round(pixels_per_cm_x, 2),
        camera_type=streamer.camera_type,
        distance_cm=CAMERA_DISTANCE_CM
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/dims')
def api_dims():
    return jsonify(streamer.get_dims())

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    global pixels_per_cm_x, pixels_per_cm_y
    try:
        known_w_cm = float(request.args.get('known_width_cm', 47) or 47)
        known_h_cm = float(request.args.get('known_length_cm', 46) or 46)

        dims = streamer.get_dims()
        curr_width_cm = dims.get("width_cm")
        curr_height_cm = dims.get("height_cm")

        if curr_width_cm is None or curr_height_cm is None or curr_width_cm <= 0:
            return jsonify({"success": False, "error": "No valid mattress target detected to calibrate against"}), 422

        with _calib_lock:
            pixel_w = curr_width_cm * pixels_per_cm_x
            pixel_h = curr_height_cm * pixels_per_cm_y

            pixels_per_cm_x = round(pixel_w / known_w_cm, 3)
            pixels_per_cm_y = round(pixel_h / known_h_cm, 3)

        _smoother.reset()

        print(f"[calibrate] Updated Scale Ratios at Distance {CAMERA_DISTANCE_CM}cm: X={pixels_per_cm_x} px/cm, Y={pixels_per_cm_y} px/cm")
        return jsonify({"success": True, "px_cm_x": pixels_per_cm_x, "px_cm_y": pixels_per_cm_y})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" 🚀 PINHOLE DISTANCE-AWARE CCTV COLOR-CONTOUR STREAM ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()