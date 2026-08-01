"""
qrtest.py — Reusable QR Detection Module for Side Camera Inspection.

Adapts QR detection logic from side_camera/testing-qr/qrtest.py.
Extracts:
  - Product Name
  - Batch Number
  - Inventory Item ID
"""

import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs

# Import pyzbar safely if installed
try:
    from pyzbar.pyzbar import decode, ZBarSymbol
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


def inspect_qr_code(image_or_path):
    """
    Analyzes an image (numpy array or file path) to extract QR code metadata.
    
    Returns dict:
      {
        "qr_found": bool,
        "product_name": str,
        "batch_no": str,
        "inventory_item_id": str,
        "raw_text": str
      }
    """
    if isinstance(image_or_path, str):
        frame = cv2.imread(image_or_path)
        if frame is None:
            return {
                "qr_found": False,
                "product_name": "N/A",
                "batch_no": "N/A",
                "inventory_item_id": "N/A",
                "raw_text": ""
            }
    else:
        frame = image_or_path

    if frame is None or frame.size == 0:
        return {
            "qr_found": False,
            "product_name": "N/A",
            "batch_no": "N/A",
            "inventory_item_id": "N/A",
            "raw_text": ""
        }

    # Preprocessing (Grayscale -> Gaussian Blur -> Sharpening)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    # 1. OpenCV QR Detector
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(sharp)

    if retval:
        for text in decoded_info:
            if text and text.strip():
                result = _parse_qr_payload(text)
                if result["qr_found"]:
                    return result

    # Also try OpenCV single detector on sharp and gray frames
    try:
        decoded_info_single, _, _ = qr_detector.detectAndDecode(gray)
        if decoded_info_single and decoded_info_single.strip():
            result = _parse_qr_payload(decoded_info_single)
            if result["qr_found"]:
                return result
    except Exception:
        pass

    # 2. pyzbar Fallback
    if HAS_PYZBAR:
        # Try sharp image first
        codes = decode(sharp, symbols=[ZBarSymbol.QRCODE])
        if not codes:
            # Fallback to gray or original frame
            codes = decode(gray, symbols=[ZBarSymbol.QRCODE])
            
        for code in codes:
            try:
                text = code.data.decode("utf-8")
                if text and text.strip():
                    result = _parse_qr_payload(text)
                    if result["qr_found"]:
                        return result
            except Exception:
                pass

    return {
        "qr_found": False,
        "product_name": "Not Detected",
        "batch_no": "Not Detected",
        "inventory_item_id": "Not Detected",
        "raw_text": ""
    }


def _parse_qr_payload(text):
    """
    Parses QR URL payload to extract productName, batchNo, and inventoryItemId.
    """
    text = text.strip()
    parsed = urlparse(text)
    params = parse_qs(parsed.query)

    product_name = params.get("productName", [""])[0].replace("_", " ")
    batch_no = params.get("batchNo", [""])[0]
    inventory_item_id = params.get("inventoryItemId", [""])[0]

    # If parameters were not present in URL query format, attempt raw text or fallback
    qr_found = True if (product_name or batch_no or inventory_item_id or text) else False

    return {
        "qr_found": qr_found,
        "product_name": product_name if product_name else (text if text else "N/A"),
        "batch_no": batch_no if batch_no else "N/A",
        "inventory_item_id": inventory_item_id if inventory_item_id else "N/A",
        "raw_text": text
    }


def print_qr_results(qr_data):
    """Prints QR results to terminal formatted as requested."""
    print("\n=========================")
    print("QR RESULT\n")
    print("Product Name:")
    print(qr_data.get("product_name", "Not Detected"))
    print("\nBatch Number:")
    print(qr_data.get("batch_no", "Not Detected"))
    print("\nInventory Item ID:")
    print(qr_data.get("inventory_item_id", "Not Detected"))
    print("=========================\n")


if __name__ == "__main__":
    # Simple standalone test
    import sys
    if len(sys.argv) > 1:
        res = inspect_qr_code(sys.argv[1])
        print_qr_results(res)

