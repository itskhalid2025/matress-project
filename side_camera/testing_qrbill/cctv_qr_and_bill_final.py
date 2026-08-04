import os
import sys
import time
import argparse
import datetime
import cv2
import numpy as np
import easyocr
from urllib.parse import urlparse, parse_qs

# Force TCP transport for RTSP stream stability
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RTSP_RECONNECT_AFTER_FAILURES = 60  # consecutive failed reads before reconnect
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)

# Results directory setup
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import pyzbar safely if installed
try:
    from pyzbar.pyzbar import decode, ZBarSymbol
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

# ---------------------------------------------------------------------------
# Detector Initialization
# ---------------------------------------------------------------------------
try:
    aruco_detector = cv2.QRCodeDetectorAruco()
except AttributeError:
    aruco_detector = None

classic_detector = cv2.QRCodeDetector()

# --- WeChatQRCode setup ---
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
        print("[INFO] WeChatQRCode DNN detector loaded successfully.")
    else:
        print(f"[WARN] WeChatQRCode model files missing in {WECHAT_MODEL_DIR}")
        print("[WARN] Far-away QR fallback will use digital zoom & classic cascade.")
except Exception as e:
    print(f"[WARN] WeChatQRCode detector init failed: {e}")
    wechat_detector = None


def preprocess_qr_variants(gray):
    """Produces CLAHE and sharpened variants for difficult lighting/blur."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return [gray, clahe.apply(gray), sharp]


def _scan_fast_cascade(variants):
    """Pass 1: ArUco -> OpenCV classic -> pyzbar across image variants."""
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
    return []


def _scan_wechat(gray_or_bgr):
    """Pass 2: WeChatQRCode DNN detector."""
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
    """Center-crop frame and upscale for digital zoom fallback."""
    h, w = gray.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = gray[y0:y0 + ch, x0:x0 + cw]
    zoomed = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    return zoomed, x0, y0, scale


def scan_qr(gray):
    """Full multi-stage QR decode cascade."""
    variants = preprocess_qr_variants(gray)

    # Pass 1: Classical cascade
    hits = _scan_fast_cascade(variants)
    if hits:
        return hits

    # Pass 2: WeChatQRCode DNN
    hits = _scan_wechat(gray)
    if hits:
        return hits

    # Pass 3: Center digital zoom
    zoomed, x0, y0, scale = _zoom_center(gray)
    zoomed_variants = preprocess_qr_variants(zoomed)

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
    """Fast presence check for live aiming box."""
    pts_list = []
    try:
        ok, points = classic_detector.detectMulti(gray)
        if ok and points is not None:
            for p in points:
                pts_list.append(p.astype(int))
            return pts_list
    except cv2.error:
        pass

    if HAS_PYZBAR:
        try:
            codes = decode(gray, symbols=[ZBarSymbol.QRCODE])
            for c in codes:
                x, y, w, h = c.rect
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                pts_list.append(pts)
            if pts_list:
                return pts_list
        except Exception:
            pass

    return pts_list


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
    cv2.polylines(img, [pts], True, (0, 255, 0), 3)

    min_x = max(10, int(np.min(pts[:, 0])))
    min_y = int(np.min(pts[:, 1]))

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

    # Semi-transparent dark card background
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


def print_qr_terminal(qr_data):
    """Prints formatted QR data to the terminal."""
    print("\n==================================================")
    print("                QR CODE DETECTED                  ")
    print("==================================================")
    print(f" Raw Payload       : {qr_data['raw_text']}")
    print(f" Product Name      : {qr_data['product_name']}")
    print(f" Batch Number      : {qr_data['batch_no']}")
    print(f" Inventory Item ID : {qr_data['inventory_item_id']}")
    print("==================================================\n")


# ---------------------------------------------------------------------------
# OCR Setup
# ---------------------------------------------------------------------------
print("[INFO] Initializing EasyOCR Reader...")
reader = easyocr.Reader(['en'], gpu=True)


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


trigger_process = False


def click_button(event, x, y, flags, param):
    global trigger_process
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 190 and 10 <= y <= 50:
            trigger_process = True


def main():
    global trigger_process

    parser = argparse.ArgumentParser(description="CCTV QR Code & Bill OCR Scanner")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP stream URL (defaults to MATTRESS_RTSP_URL env var or default RTSP URL)")
    parser.add_argument("--win-width", type=int, default=1024, help="Display window width in pixels (default: 1024)")
    parser.add_argument("--win-height", type=int, default=576, help="Display window height in pixels (default: 576)")
    args = parser.parse_args()

    window_name = "CCTV QR & Bill Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.win_width, args.win_height)
    cv2.setMouseCallback(window_name, click_button)

    print(f"\n--- CCTV SYSTEM READY ---")
    print(f"RTSP URL: {args.rtsp}")
    print("1. Live preview displays yellow box when QR is in frame.")
    print("2. Click 'Process' or press 's' to scan QR and Bill OCR.")
    print("3. Press 'q' to quit.\n")

    print(f"[CCTV] Connecting to RTSP stream: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    consecutive_failures = 0

    while True:
        if cap is None or not cap.isOpened():
            print("[CCTV] RTSP stream disconnected. Retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures % 10 == 0:
                print(f"[CCTV] Failed frame read count: {consecutive_failures}/{RTSP_RECONNECT_AFTER_FAILURES}")
            if consecutive_failures >= RTSP_RECONNECT_AFTER_FAILURES:
                print("[CCTV] RTSP stream unresponsive — attempting reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
            time.sleep(0.005)
            continue

        consecutive_failures = 0

        clean_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Live aiming box presence check
        for pts in detect_qr_presence(gray):
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), 2)
            cv2.putText(frame, "QR in frame - Click Process", (pts[0][0], max(20, pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # "Process" UI Button
        cv2.rectangle(frame, (10, 10), (190, 50), (0, 255, 0), -1)
        cv2.putText(frame, "Process", (45, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if trigger_process:
            annotated = clean_frame.copy()
            clean_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)

            # 1. QR Decode
            print("\n--- Processing QR Code(s)... ---")
            qr_hits = scan_qr(clean_gray)
            if not qr_hits:
                print("No QR code decoded. Check distance/focus.")
            else:
                for text, pts in qr_hits:
                    qr_data = parse_qr_payload(text)
                    draw_qr_overlay(annotated, pts, qr_data)
                    print_qr_terminal(qr_data)

            # 2. Bill OCR
            print("--- Processing Bill Text (OCR)... ---")
            p = preprocess_for_ocr(clean_frame)
            results = reader.readtext(p, rotation_info=[90, 180, 270])

            if not results:
                print("No OCR text found.")
            else:
                print("--- OCR Output ---")
                for (bbox, text_res, prob) in results:
                    clean_text = text_res.strip()
                    if len(clean_text) <= 1 and not clean_text.isdigit():
                        continue
                    if prob < 0.25:
                        continue

                    print(f"  [{prob:.2f}] {clean_text}")

                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(annotated, [pts], True, (0, 0, 255), 2)
                    cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save result image to results folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"[INFO] Result image saved to: {save_path}")

            cv2.imshow(window_name, annotated)
            cv2.waitKey(4000)
            trigger_process = False

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            trigger_process = True

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
