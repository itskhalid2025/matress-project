"""
mattress/config.py — every tunable in one place.
Values marked PLACEHOLDER are uncalibrated until scripts/separability.py output
is used to set them; values marked CALIBRATED were set from measured probes on
the reference dataset (2026-07, unit1, rot000/090/180).
"""

# ---- Scale enforcement (G2) -------------------------------------------------
CAPTURE_W = 1920          # enforced native width  — all reference data is 1920x1080
CAPTURE_H = 1080          # enforced native height
CAM_TOP_INDEX = 2         # cam2 = top view = texture identity (indices shuffle on reboot)

# ---- Cropping (G8) ----------------------------------------------------------
# Fixed crop box for the controlled line / POC frames (documented switch, NOT the
# banner defence — banner exclusion is appearance-based below).
FIXED_CROP = (150, 930, 300, 1620)   # y0, y1, x0, x1

# ---- Deterministic sampling (G1) ---------------------------------------------
PATCH_SIZE = 128          # native-pixel patch. NEVER RESIZE on the feature path.
PATCH_STRIDE = 128        # deterministic grid stride
INTERIOR_MARGIN = 0.15    # coarse pre-filter margin inside the crop (rim/piping)

# ---- Quality gates & exclusion masks (G3, G4) --------------------------------
# CALIBRATED 2026-07: min cropped-gray Laplacian variance over all 147 good
# reference frames = 122 (p05=163, median=308). Gate at half the observed
# minimum so borderline-good frames pass; recalibrate the low side once
# genuinely blurry rig captures exist.
SHARPNESS_GATE = 60.0
GLARE_V_THRESH = 240      # near-clipping value  (plastic-wrap glare)
GLARE_S_THRESH = 15       # near-white saturation
GLARE_REJECT_FRAC = 0.15  # patch rejected above this glare fraction

# Sash (banner) = large connected blob of saturated+bright pixels whose hue
# DIFFERS from the frame's own dominant fabric hue (frame-adaptive).
# CALIBRATED: gravite fabric saturation shifted between capture sessions
# (blue-pixel sat p90: 114 on rot000 vs 174 on rot180) and overlaps the sash's
# sat range (p50 179) — so a fixed saturation gate provably cannot separate
# sash from fabric. Hue does: fabric blue ~105 vs sash orange ~12 (gravite),
# and the rule generalises (maxi_plush: warm fabric vs magenta 152 sash;
# near-white covers have no dominant hue, so ANY saturated blob is sash-like).
SASH_S_THRESH = 90        # candidate floor (kept low; hue rule does the work)
SASH_V_THRESH = 90
SASH_HUE_DELTA = 25       # circular hue distance from dominant fabric hue
FABRIC_HUE_MIN_SAT = 60   # pixels counted toward the dominant-hue estimate
# Only trust a dominant FABRIC hue when saturated pixels are fabric-scale — a
# sash is never most of the cover. CALIBRATED in-crop (s>60 fraction):
# gravite >=34.5%, maxi_plush >=9.7% (coloured fabrics) vs ortholex <=8.4%,
# maxi_pro <=7.0% (near-white; there ANY saturated blob is sash-like).
FABRIC_HUE_MIN_FRAC = 0.09
SASH_MIN_AREA = 5000      # px, connected-component gate (sash blobs 5.6k-135k px)
SASH_DILATE = 15          # px, catch sash edge/shadow

# Dark-ink / background blobs (grey-sash text, dark gaps at crop edge).
# CALIBRATED: maxi_pro (no sash in crop) tops out at 4.5k px dark blobs (stitch
# shadows); real sash/background dark masses run 23k-56k px.
DARK_V_THRESH = 110
DARK_MIN_AREA = 8000
DARK_DILATE = 10

EXCLUDE_REJECT_FRAC = 0.05  # patch rejected if sash/dark overlap exceeds this (strict)
MIN_CONTRAST_THRESH = 5.0   # gray std; rejects blank/blown patches

# ---- Feature blocks (G5) ------------------------------------------------------
COLOR_H_BINS = 8
COLOR_S_BINS = 8
LBP_P = 8
LBP_R = 1                  # uniform LBP, P=8 -> 10 histogram bins (values 0..9)
GLCM_LEVELS = 32           # quantized (gray//8) — classical practice, big speedup
GLCM_DISTANCES = [1, 2]
GLCM_PROPS = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
# Macro-pattern cue = FFT radial/angular profile (explicitly allowed by spec).
# CALIBRATED: angular anisotropy in these bands separates ortholex (oriented
# hexagons, ratio 4.4-6.4) from maxi_pro (isotropic mottle, 2.2-4.1); the
# wavelength-16-32px band is the strongest and was missed by the old Gabor freqs.
MACRO_BANDS = [(2, 4), (4, 8), (8, 16), (16, 32)]  # FFT radius bands on a 128px patch
MACRO_N_ANGLES = 18
STATS_Z_CLIP = 3.0          # z-scored stats blocks clipped to +/-3 then scaled

# Block weights — applied AFTER each block is brought to comparable unit scale.
WEIGHT_COLOR = 1.0          # PLACEHOLDER
WEIGHT_LBP = 1.0            # PLACEHOLDER
WEIGHT_GLCM = 1.0           # PLACEHOLDER
WEIGHT_MACRO = 1.0          # PLACEHOLDER

# ---- Matching guards (G7) ------------------------------------------------------
# CALIBRATED 2026-07 on current data: known val patches' best-distance p99=0.32
# (max ~0.35) vs unknown-proxy patches starting at 0.38 (side-camera frames used
# as out-of-set proxy; real unknown TOP-view covers still needed). At 0.35:
# 100% known patches kept, 100% unknown proxy rejected. Margin set below known
# p10 (0.069) to flag only truly ambiguous patches; frame vote absorbs the rest.
# RE-CALIBRATE whenever enrollment, lighting, or the SKU set changes.
REJECTION_MAX_DIST = 0.35
MARGIN_GUARD = 0.03
VOTE_MIN_FRAC = 0.5         # winning vote fraction below this -> NEEDS_REVIEW
VOTE_MIN_DECIDED = 3        # min decided patches required
VOTE_MIN_FRAC_DECIDED = 0.7 # min fraction of decided patches agreeing
VOTE_DISSENT_MAX_FRAC = 0.25 # max dissenting vote fraction allowed

# ================================================================================
# CLAIM LAYER (QR + label OCR + reconciliation) — Milestone 2, mattress/claim.py
# The QR/label are CLAIMS, never identity. Fabric verdict is always authoritative;
# reconcile.py never lets a claim override it.
# ================================================================================
CAM_SIDE_INDEX = 0          # side view = label + QR (VERIFY with --list; shuffles)

# CALIBRATED 2026-07: this webcam delivers visibly SOFTER frames at 1920x1080
# than at 1280x720 (measured: Laplacian-variance sharpness 233 vs 2171 on the
# same QR region at the same instant — ~9x drop). Almost certainly a USB
# bandwidth/MJPG-compression limit at the higher resolution on this rig (same
# class of issue as the earlier SSH MJPEG-tunnel saturation). This does NOT
# conflict with the fabric-side 1920x1080 requirement: nothing in the claim
# layer (QR/OCR) needs pixel-scale consistency with the enrolled fabric
# signatures, so the side camera is free to run at whatever resolution this
# hardware decodes best at. Use these for side-camera captures ONLY — never
# for the top/fabric camera.
CLAIM_CAPTURE_W = 1280
CLAIM_CAPTURE_H = 720

# QR decode retry ladder (classical, cheap, in this order until one decodes).
QR_RETRY_GRAYSCALE = True
QR_RETRY_UPSCALE = 2.0      # try a 2x upscale if direct + grayscale both fail
QR_RETRY_CLAHE = True       # local contrast boost as a last resort
QR_RETRY_OTSU = True       # NEW 2026-07: needed for Basler acA4600-10uc frames -- see notes below

# SKU name -> key mapping lives in mattress/claim.py:_match_sku_in_text() (it
# needs word-token logic to split the near-identical maxi pair — 'Maxi Plush'
# vs 'Maxiplush Pro' — which a flat dict cannot do safely). Confirmed product
# names from decoded QRs: 'Gravite', 'Ortho Lex'. Provisional from label text
# (QRs not yet decodable, Task C1): 'Maxi Plush', 'Maxiplush Pro'. Re-verify the
# maxi pair against real QR productName strings once their side frames decode.

# Label OCR (Task C3). Tesseract = classical/offline, the safe default under the
# no-deep-learning constraint (most modern OCR engines are DL-based — confirm
# Tesseract's acceptability, or drop OCR and rely on QR + CODE39 barcode alone).
OCR_ENGINE = 'tesseract'
OCR_LANG = 'eng'
OCR_MIN_CONF = 40           # tesseract per-word confidence (0-100) to keep a token

# PLACEHOLDER: label region within the side crop, as fractions of the frame
# (x0, y0, x1, y1). Calibrate once the side camera framing is fixed (Task C1) —
# current default is a generous centre-right guess, NOT yet validated.
LABEL_CROP_FRAC = (0.35, 0.10, 1.00, 0.90)
# EMPIRICALLY DETERMINED 2026-07 (gravite sample, brute-force rotation sweep):
# the label reads correctly at 90 deg clockwise, no flip. Re-verify if side
# camera mounting changes. None = no rotation; also: cv2.ROTATE_180,
# cv2.ROTATE_90_COUNTERCLOCKWISE.
import cv2 as _cv2
LABEL_ROTATE_CODE = _cv2.ROTATE_90_CLOCKWISE
