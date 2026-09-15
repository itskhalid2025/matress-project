import os
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs
from config import WECHAT_MODEL_DIR

# Import pyzbar safely
try:
    from pyzbar.pyzbar import decode, ZBarSymbol
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

# ArUco and Classic QR Detectors
try:
    aruco_detector = cv2.QRCodeDetectorAruco()
except AttributeError:
    aruco_detector = None

classic_detector = cv2.QRCodeDetector()

# WeChatQRCode setup
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
        print("[INFO] WeChatQRCode DNN detector loaded successfully.")
    else:
        print(f"[WARN] WeChatQRCode model files missing in {WECHAT_MODEL_DIR}")
except Exception as e:
    print(f"[WARN] WeChatQRCode detector init failed: {e}")
    wechat_detector = None


def preprocess_qr_variants(gray):
    """Produces CLAHE and sharpened variants for difficult lighting/blur."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return [gray, clahe.apply(gray), sharp]


def remap_rotated_pts(pts, angle, orig_h, orig_w):
    """Remaps polygon coordinates from rotated image space back to original image space."""
    if pts is None or len(pts) == 0:
        return pts
    if angle == 0:
        return np.array(pts, dtype=np.int32)

    pts = np.array(pts, dtype=np.float32)
    remapped = np.zeros_like(pts)

    if angle == 90:  # Clockwise rotation
        remapped[:, 0] = pts[:, 1]
        remapped[:, 1] = orig_h - 1 - pts[:, 0]
    elif angle == 180:  # 180 deg rotation
        remapped[:, 0] = orig_w - 1 - pts[:, 0]
        remapped[:, 1] = orig_h - 1 - pts[:, 1]
    elif angle == 270:  # Counter-clockwise rotation
        remapped[:, 0] = orig_w - 1 - pts[:, 1]
        remapped[:, 1] = pts[:, 0]

    return remapped.astype(np.int32)


def _scan_single_orientation(gray):
    """Cascade QR decode on a single image orientation."""
    variants = preprocess_qr_variants(gray)

    # Pass 1: Classical cascade
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

        if HAS_PYZBAR:
            codes = decode(variant, symbols=[ZBarSymbol.QRCODE])
            if codes:
                hits = []
                for c in codes:
                    x, y, w, h = c.rect
                    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                    hits.append((c.data.decode("utf-8"), pts))
                return hits

    # Pass 2: WeChatQRCode DNN
    if wechat_detector is not None:
        img = gray
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res, points = wechat_detector.detectAndDecode(img)
        hits = []
        for text, pts in zip(res, points):
            if text:
                hits.append((text, pts.astype(int)))
        if hits:
            return hits

    return []


def scan_qr(gray):
    """
    Multi-Orientation QR Scanner (0 deg, 90 deg, 180 deg, 270 deg).
    Ensures sideways or upside-down QR codes are detected instantly.
    """
    orig_h, orig_w = gray.shape[:2]

    rotations = [
        (0, gray),
        (90, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(gray, cv2.ROTATE_180)),
        (270, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

    for angle, rot_gray in rotations:
        hits = _scan_single_orientation(rot_gray)
        if hits:
            remapped_hits = []
            for text, pts in hits:
                remapped_pts = remap_rotated_pts(pts, angle, orig_h, orig_w)
                remapped_hits.append((text, remapped_pts))
            return remapped_hits

    return []


def detect_qr_presence(gray):
    """Multi-Orientation presence check for live stream aiming box."""
    orig_h, orig_w = gray.shape[:2]

    rotations = [
        (0, gray),
        (90, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(gray, cv2.ROTATE_180)),
        (270, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

    for angle, rot_gray in rotations:
        pts_list = []
        try:
            ok, points = classic_detector.detectMulti(rot_gray)
            if ok and points is not None:
                for p in points:
                    remapped_pts = remap_rotated_pts(p, angle, orig_h, orig_w)
                    pts_list.append(remapped_pts)
                return pts_list
        except cv2.error:
            pass

        if HAS_PYZBAR:
            try:
                codes = decode(rot_gray, symbols=[ZBarSymbol.QRCODE])
                for c in codes:
                    x, y, w, h = c.rect
                    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                    remapped_pts = remap_rotated_pts(pts, angle, orig_h, orig_w)
                    pts_list.append(remapped_pts)
                if pts_list:
                    return pts_list
            except Exception:
                pass

    return []


def parse_qr_payload(text):
    """Extract URL parameters or fallback to raw payload text."""
    text = text.strip()
    parsed = urlparse(text)
    params = parse_qs(parsed.query)

    product_name = params.get("productName", [""])[0].replace("_", " ")
    batch_no = params.get("batchNo", [""])[0]
    inventory_item_id = params.get("inventoryItemId", [""])[0]

    if not product_name and not batch_no and not inventory_item_id:
        product_name = text if text else "Not Detected"
        batch_no = "N/A"
        inventory_item_id = "N/A"
    else:
        product_name = product_name if product_name else "N/A"
        batch_no = batch_no if batch_no else "N/A"
        inventory_item_id = inventory_item_id if inventory_item_id else "N/A"

    return {
        "raw_text": text,
        "product_name": product_name,
        "batch_no": batch_no,
        "inventory_item_id": inventory_item_id
    }


def draw_qr_overlay(img, pts, qr_data):
    """Renders green bounding polygon and an on-screen QR metadata card."""
    pts_int = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts_int], True, (0, 255, 0), 3)

    min_x = max(10, int(np.min(pts_int[:, 0])))
    min_y = int(np.min(pts_int[:, 1]))

    lines = [
        "QR DETECTED",
        f"Product : {qr_data['product_name']}",
        f"Batch   : {qr_data['batch_no']}",
        f"ItemID  : {qr_data['inventory_item_id']}"
    ]

    card_h = len(lines) * 24 + 12
    card_w = 400
    card_x0 = min_x
    card_y0 = max(60, min_y - card_h - 10)

    overlay = img.copy()
    cv2.rectangle(overlay, (card_x0, card_y0), (card_x0 + card_w, card_y0 + card_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (card_x0, card_y0), (card_x0 + card_w, card_y0 + card_h), (0, 255, 0), 2)

    for idx, line in enumerate(lines):
        color = (0, 255, 255) if idx == 0 else (255, 255, 255)
        scale = 0.6 if idx == 0 else 0.5
        thickness = 2 if idx == 0 else 1
        ty = card_y0 + 22 + (idx * 24)
        cv2.putText(img, line, (card_x0 + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
