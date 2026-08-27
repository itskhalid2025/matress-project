import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Local Model Paths inside side_cam_final/models/
TEXTURE_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
WECHAT_MODEL_DIR = os.path.join(MODELS_DIR, "wechat_qrcode")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "bestdimension.pt")

if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
DB_PATH = os.path.join(BASE_DIR, "database.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Texture Classes
CLASS_NAMES = [
    "Dual harmony",
    "Gravite",
    "Maxi plush",
    "Maxi pro",
    "Memorise",
    "Ortholex",
    "Purity plus",
    "Velvet"
]

# Camera Indices & Stream Sources
CAMERA_QR_INDEX = 0
CAMERA_BILL_TEXTURE_INDEX = 1
CAMERA_TOP_INDEX = 2

# CCTV Top Camera RTSP Stream URL
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)

# OCR Settings
OCR_CONFIDENCE_THRESHOLD = 0.25

# Expected Dimensions (in cm) for Tolerance Checking
EXPECTED_DIMENSIONS = {
    "maxiplus": {"length_cm": 190.0, "width_cm": 160.0, "thickness_cm": 20.0},
    "maxi plush": {"length_cm": 190.0, "width_cm": 160.0, "thickness_cm": 20.0},
    "maxi pro": {"length_cm": 198.0, "width_cm": 180.0, "thickness_cm": 25.0},
    "dual harmony": {"length_cm": 198.0, "width_cm": 150.0, "thickness_cm": 15.0},
    "gravite": {"length_cm": 190.0, "width_cm": 120.0, "thickness_cm": 18.0},
    "memorise": {"length_cm": 200.0, "width_cm": 180.0, "thickness_cm": 25.0},
    "ortholex": {"length_cm": 190.0, "width_cm": 160.0, "thickness_cm": 20.0},
    "purity plus": {"length_cm": 198.0, "width_cm": 180.0, "thickness_cm": 25.0},
    "velvet": {"length_cm": 190.0, "width_cm": 140.0, "thickness_cm": 15.0},
    "default": {"length_cm": 190.0, "width_cm": 160.0, "thickness_cm": 20.0}
}

DIMENSION_TOLERANCE_CM = 10.0

# Banner Checking & Sash Thresholds (used by banner.py)
SHARPNESS_GATE = 60.0
GLARE_V_THRESH = 240
GLARE_S_THRESH = 15
GLARE_REJECT_FRAC = 0.15
SASH_S_THRESH = 90
SASH_V_THRESH = 90
SASH_HUE_DELTA = 25
FABRIC_HUE_MIN_SAT = 60
FABRIC_HUE_MIN_FRAC = 0.09
SASH_MIN_AREA = 5000
SASH_DILATE = 15
DARK_V_THRESH = 110
DARK_MIN_AREA = 8000
DARK_DILATE = 10
OCR_LANG = 'eng'
OCR_MIN_CONF = 40


