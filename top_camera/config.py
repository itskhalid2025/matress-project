"""
config.py — Mattress Top Camera QC Configuration Tunables.

Contains all camera, cropping, dimension-checking, and banner-checking constants.
Values are documented and set based on previous project calibrations.
"""

# ==============================================================================
# Camera & Resolution Tunables
# ==============================================================================
# The default webcam device index (e.g. /dev/video2). Note that webcam indices
# can shuffle across reboots. This is modifiable at runtime via the dashboard.
CAM_INDEX = 0

# Enforced capture resolution. Reference images and templates expect a 1080p frame.
CAPTURE_W = 1920
CAPTURE_H = 1080

# ==============================================================================
# Cropping ROI (Region of Interest)
# ==============================================================================
# Fixed crop box to isolate the mattress cover on the conveyor/line.
# Format: (y0, y1, x0, x1) in pixels
FIXED_CROP = (150, 930, 300, 1620)

# ==============================================================================
# Dimension Checking Calibration
# ==============================================================================
# Ratio of pixels per centimeter. Set to None initially to require calibration,
# or default to a safe starting placeholder (e.g. 15.0).
PIXELS_PER_CM = 15.0

# Multiplier to account for contour smoothing / edge offsets.
EDGE_CORRECTION_FACTOR = 1.1

# Minimum area of a contour (in pixels) to be considered a mattress target.
# Minimum bounds protect against speckle noise; maximum bounds prevent background bleed.
MIN_CONTOUR_AREA = 50000

# ==============================================================================
# Banner Checking Thresholds
# ==============================================================================
# Sharpness gate based on Laplacian variance. Frames below this are marked blurry.
SHARPNESS_GATE = 60.0

# Glare detection parameters for plastic-wrap reflections.
GLARE_V_THRESH = 240      # brightness value floor
GLARE_S_THRESH = 15       # saturation ceiling (near-white)
GLARE_REJECT_FRAC = 0.15  # reject patch if glare fraction exceeds this

# Colored sash detection (HSV thresholds)
SASH_S_THRESH = 90        # saturation floor
SASH_V_THRESH = 90        # brightness floor
SASH_HUE_DELTA = 25       # circular hue delta from dominant fabric hue

FABRIC_HUE_MIN_SAT = 60   # minimum saturation to count towards dominant hue
FABRIC_HUE_MIN_FRAC = 0.09 # minimum fraction of pixels to claim fabric hue
SASH_MIN_AREA = 5000      # minimum size in pixels for a valid sash blob
SASH_DILATE = 15          # dilation size to cover edges/shadows

# Dark-ink markings detection
DARK_V_THRESH = 110       # brightness ceiling (dark print/ink)
DARK_MIN_AREA = 8000      # minimum size of dark blob
DARK_DILATE = 10          # dilation size for dark region masking

# OCR constraints
OCR_LANG = 'eng'
OCR_MIN_CONF = 40         # minimum Tesseract word confidence (0-100)
