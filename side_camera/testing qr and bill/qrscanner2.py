import platform
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from urllib.parse import urlparse, parse_qs

# ==============================
# QR detector cascade
# ==============================
# cv2.QRCodeDetectorAruco (OpenCV >= 4.8, ships in plain opencv-python, no
# extra install) tends to find QR codes the classic detector misses --
# especially at an angle or partially occluded. Falls back cleanly on
# older OpenCV.
try:
    aruco_detector = cv2.QRCodeDetectorAruco()
except AttributeError:
    aruco_detector = None

classic_detector = cv2.QRCodeDetector()


def preprocess_variants(gray):
    """A few different views of the same frame. Different ones win depending
    on glare, curvature, and distance -- trying several beats picking one
    fixed preprocessing and hoping it always works."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return [gray, clahe.apply(gray), sharp]


def scan_qr(gray):
    """Try each preprocessing variant against the detector cascade
    (ArUco -> classic OpenCV -> pyzbar), stopping at the first hit.
    Returns a list of (text, 4x2 int array of corner points)."""
    for variant in preprocess_variants(gray):

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


# ==============================
# Camera Setup
# ==============================
# CAP_DSHOW is Windows-only; pick V4L2 automatically on Linux (Raspberry Pi
# included) so this doesn't silently fail to open the camera later.
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

seen = set()

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

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for text, pts in scan_qr(gray):

        # Draw QR outline
        for i in range(4):
            cv2.line(display, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 3)

        parsed = urlparse(text)
        params = parse_qs(parsed.query)

        product_name = params.get("productName", [""])[0].replace("_", " ")
        batch_no = params.get("batchNo", [""])[0]
        inventory_item_id = params.get("inventoryItemId", [""])[0]

        qr_data["product_name"] = product_name
        qr_data["batch_no"] = batch_no
        qr_data["inventory_item_id"] = inventory_item_id

        if text not in seen:
            seen.add(text)

            print("\n========== QR DATA ==========")
            print("Product Name      :", qr_data["product_name"])
            print("Batch No          :", qr_data["batch_no"])
            print("Inventory Item ID :", qr_data["inventory_item_id"])
            print("=============================")

    cv2.imshow("QR Scanner", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        seen.clear()
        print("QR cache cleared.")

cap.release()
cv2.destroyAllWindows()