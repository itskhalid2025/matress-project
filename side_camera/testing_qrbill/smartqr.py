import os
import platform
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# QR detector cascade -- the full cascade runs ONLY when triggered (button
# click or 's' key), not every frame. The live preview only runs a cheap,
# decode-free presence check so it stays lightweight while you aim.
#
#   Pass 1 (cheap): ArUco -> classic OpenCV -> pyzbar, across preprocessing
#                    variants
#   Pass 2 (only if pass 1 finds nothing): WeChatQRCode DNN detector
#           -- includes a super-resolution model that zooms in on small/far
#              QR codes before decoding
#   Pass 3 (only if pass 2 also finds nothing): center-crop + 2x digital
#           zoom, then rerun passes 1 and 2 on the zoomed image
# ---------------------------------------------------------------------------
try:
    aruco_detector = cv2.QRCodeDetectorAruco()
except AttributeError:
    aruco_detector = None

classic_detector = cv2.QRCodeDetector()

# --- WeChatQRCode setup ----------------------------------------------------
# 4 model files, downloaded from:
#   git clone -b wechat_qrcode_20210119 https://github.com/opencv/opencv_3rdparty.git wechat_model
# Expects these exact filenames (no "_2021nov" suffix -- that was wrong
# in an earlier version of this script):
#     detect.prototxt
#     detect.caffemodel
#     sr.prototxt
#     sr.caffemodel
# Place them in a "models/wechat_qrcode/" folder next to this script.
# Also requires: pip install opencv-contrib-python
WECHAT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "wechat_qrcode")

wechat_detector = None
try:
    _detect_proto = os.path.join(WECHAT_MODEL_DIR, "detect.prototxt")
    _detect_model = os.path.join(WECHAT_MODEL_DIR, "detect.caffemodel")
    _sr_proto = os.path.join(WECHAT_MODEL_DIR, "sr.prototxt")
    _sr_model = os.path.join(WECHAT_MODEL_DIR, "sr.caffemodel")
    if all(os.path.isfile(p) for p in (_detect_proto, _detect_model, _sr_proto, _sr_model)):
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            _detect_proto, _detect_model, _sr_proto, _sr_model
        )
        print("WeChatQRCode detector loaded (far/small-code fallback enabled).")
    else:
        print("WeChatQRCode model files not found in", WECHAT_MODEL_DIR)
        print("Far-away QR fallback will be limited to digital zoom only.")
except AttributeError:
    print("cv2.wechat_qrcode_WeChatQRCode not available.")
    print("Install with: pip install opencv-contrib-python")
    wechat_detector = None


def preprocess_variants(gray):
    """A few different views of the same frame. Different ones win depending
    on glare, curvature, and distance -- trying several beats picking one
    fixed preprocessing and hoping it always works."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return [gray, clahe.apply(gray), sharp]


def _scan_fast_cascade(variants):
    """Pass 1: ArUco -> classic -> pyzbar, across preprocessing variants."""
    for variant in variants:
        if aruco_detector is not None:
            ok, decoded_info, points, _ = aruco_detector.detectAndDecodeMulti(variant)
            if ok:
                hits = [(t, p.astype(int)) for t, p in zip(decoded_info, points) if t]
                if hits:
                    return hits

        ok, decoded_info, points, _ = classic_detector.detectAndDecodeMulti(variant)
        if ok:
            hits = [(t, p.astype(int)) for t, p in zip(decoded_info, points) if t]
            if hits:
                return hits

        codes = decode(variant, symbols=[ZBarSymbol.QRCODE])
        if codes:
            hits = []
            for c in codes:
                x, y, w, h = c.rect
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                hits.append((c.data.decode("utf-8"), pts))
            return hits
    return []


def _scan_wechat(gray_or_bgr):
    """Pass 2: WeChatQRCode DNN detector (with built-in super-resolution)."""
    if wechat_detector is None:
        return []

    img = gray_or_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res, points = wechat_detector.detectAndDecode(img)
    hits = []
    for text, pts in zip(res, points):
        if text:
            hits.append((text, pts.astype(int)))
    return hits


def _zoom_center(gray, scale=2.0, crop_frac=0.5):
    """Crop the center crop_frac of the frame and upscale by `scale`."""
    h, w = gray.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = gray[y0:y0 + ch, x0:x0 + cw]
    zoomed = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    return zoomed, x0, y0, scale


def scan_qr(gray):
    """Full decode cascade. Only called on-demand (trigger)."""
    variants = preprocess_variants(gray)

    hits = _scan_fast_cascade(variants)
    if hits:
        return hits

    hits = _scan_wechat(gray)
    if hits:
        return hits

    zoomed, x0, y0, scale = _zoom_center(gray)
    zoomed_variants = preprocess_variants(zoomed)

    hits = _scan_fast_cascade(zoomed_variants)
    if not hits:
        hits = _scan_wechat(zoomed)

    if hits:
        remapped = []
        for text, pts in hits:
            full_pts = (pts.astype(float) / scale) + np.array([x0, y0])
            remapped.append((text, full_pts.astype(int)))
        return remapped

    return []


def detect_qr_presence(gray):
    """
    Cheap, decode-free localization used ONLY to draw a live aiming box.
    Runs every frame, but does no decoding, no WeChat DNN, no zoom fallback --
    just a single fast check on the plain image so the preview stays responsive.
    """
    try:
        ok, points = classic_detector.detectMulti(gray)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    return [p.astype(int) for p in points]


trigger_scan = False


def click_button(event, x, y, flags, param):
    global trigger_scan
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 170 and 10 <= y <= 50:
            trigger_scan = True


# ---------------------------------------------------------------------------
# Camera Setup
# ---------------------------------------------------------------------------
backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

print("Camera Started")
print(f"Resolution : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"ArUco QR detector available: {aruco_detector is not None}")
print(f"WeChatQRCode detector available: {wechat_detector is not None}")

cv2.namedWindow("QR Scanner")
cv2.setMouseCallback("QR Scanner", click_button)

print("Live preview shows a yellow aiming box when a QR is in frame (no decoding yet).")
print("Click 'Scan QR' or press 's' to decode. Press 'q' to quit.")

# Stores latest QR information
qr_data = {
    "product_name": "",
    "batch_no": "",
    "inventory_item_id": ""
}

while True:

    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- live aiming box only: cheap presence check, NO decoding ----
    for pts in detect_qr_presence(gray):
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), 2)
        cv2.putText(frame, "QR in frame", (pts[0][0], pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ---- "Scan QR" button overlay ----
    cv2.rectangle(frame, (10, 10), (170, 50), (0, 255, 0), -1)
    cv2.putText(frame, "Scan QR", (30, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # ---- on-demand: full decode cascade, once ----
    if trigger_scan:
        print("\n--- Scanning for QR code(s)... ---")
        clean_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
        hits = scan_qr(clean_gray)

        annotated = clean_frame.copy()
        if not hits:
            print("No QR code decoded. Check distance/focus.")
        for text, pts in hits:
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)

            parsed = urlparse(text)
            params = parse_qs(parsed.query)

            qr_data["product_name"] = params.get("productName", [""])[0].replace("_", " ")
            qr_data["batch_no"] = params.get("batchNo", [""])[0]
            qr_data["inventory_item_id"] = params.get("inventoryItemId", [""])[0]

            print("========== QR DATA ==========")
            print("Product Name      :", qr_data["product_name"])
            print("Batch No          :", qr_data["batch_no"])
            print("Inventory Item ID :", qr_data["inventory_item_id"])
            print("=============================")

        cv2.imshow("QR Scanner", annotated)
        cv2.waitKey(2000)

        trigger_scan = False

    cv2.imshow("QR Scanner", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        trigger_scan = True

cap.release()
cv2.destroyAllWindows()
