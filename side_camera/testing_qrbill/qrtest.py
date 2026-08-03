import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from urllib.parse import urlparse, parse_qs

# ==============================
# Camera Setup
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

print("Camera Started") 
print(f"Resolution : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

qr_detector = cv2.QRCodeDetector()

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

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    decoded = False

    # =====================================================
    # OpenCV QR Detection
    # =====================================================
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(sharp)

    if retval:

        for text, pts in zip(decoded_info, points):

            if text == "":
                continue

            pts = pts.astype(int)

            # Draw ONLY QR outline
            for i in range(4):
                cv2.line(
                    display,
                    tuple(pts[i]),
                    tuple(pts[(i + 1) % 4]),
                    (0, 255, 0),
                    3
                )

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

            decoded = True

    # =====================================================
    # pyzbar Fallback
    # =====================================================
    if not decoded:

        codes = decode(sharp, symbols=[ZBarSymbol.QRCODE])

        for code in codes:

            text = code.data.decode("utf-8")

            x, y, w, h = code.rect

            # Draw ONLY rectangle
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

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