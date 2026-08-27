# ================================================================================
# CLAIM LAYER (QR + label OCR + reconciliation) — Milestone 2, mattress/claim.py
# The QR/label are CLAIMS, never identity. Fabric verdict is always authoritative;
# reconcile.py never lets a claim override it.
# ================================================================================
CAM_SIDE_INDEX = 0          # Now using the high-quality camera

# Restored to 1920x1080. Since you are using the better camera that was previously
# handling the top view, it should not suffer from the 720p MJPG compression 
# blurring issue that the old side camera had.
CLAIM_CAPTURE_W = 1920
CLAIM_CAPTURE_H = 1080

# QR decode retry ladder (classical, cheap, in this order until one decodes).
QR_RETRY_GRAYSCALE = True
QR_RETRY_UPSCALE = 2.0      # try a 2x upscale if direct + grayscale both fail
QR_RETRY_CLAHE = True       # local contrast boost as a last resort
QR_RETRY_OTSU = False       # DISABLED: Set to False if this is a standard auto-exposing webcam

# Label OCR (Task C3). 
OCR_ENGINE = 'tesseract'
OCR_LANG = 'eng'
OCR_MIN_CONF = 40           # tesseract per-word confidence (0-100) to keep a token

# Label crop: capture the full label region (left half of frame, full height).
# The VARIETY column appears around x: 10%-50%, y: 10%-90% in the live feed.
LABEL_CROP_FRAC = (0.05, 0.05, 0.65, 0.95)

import cv2 as _cv2
# RESET: Standard webcams are typically mounted upright. Set to None to prevent 
# sideways OCR reading, unless you purposefully mounted it sideways again.
LABEL_ROTATE_CODE = None