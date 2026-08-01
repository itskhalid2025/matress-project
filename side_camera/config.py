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

# PLACEHOLDER: You will need to manually recalibrate this for the new camera's Field of View.
# Take a test frame and measure the (x0, y0, x1, y1) fractions of where the label sits.
LABEL_CROP_FRAC = (0.35, 0.10, 1.00, 0.90)

import cv2 as _cv2
# RESET: Standard webcams are typically mounted upright. Set to None to prevent 
# sideways OCR reading, unless you purposefully mounted it sideways again.
LABEL_ROTATE_CODE = None