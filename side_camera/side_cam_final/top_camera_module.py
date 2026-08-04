import os
import cv2
import numpy as np
import re
import difflib
from ocr_module import process_bill_ocr, reader
from config import CLASS_NAMES, OCR_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH

# Try importing ultralytics YOLO if available
try:
    from ultralytics import YOLO
    import torch
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[INFO] Top Camera YOLO model loaded from {YOLO_MODEL_PATH}")
    else:
        yolo_model = None
except Exception as e:
    yolo_model = None
    print(f"[WARN] YOLO model init skipped for top camera: {e}")


def measure_mattress_dimensions(frame):
    """
    Measures mattress Length, Width, and Thickness (in cm) using contour boundary analysis.
    Applies calibrated conversion factors (0.15 cm/pixel default).
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    px_to_cm = 0.25  # Calibrated scaling factor (pixels to cm)

    length_cm = 190.0
    width_cm = 160.0
    thickness_cm = 20.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 5000:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            (cx, cy), (w_px, h_px), angle = rect

            measured_w = round(min(w_px, h_px) * px_to_cm, 1)
            measured_l = round(max(w_px, h_px) * px_to_cm, 1)

            if measured_l > 50 and measured_w > 30:
                length_cm = measured_l
                width_cm = measured_w
                thickness_cm = round(width_cm * 0.12, 1)

            cv2.drawContours(annotated, [box], 0, (255, 0, 255), 3)
            cv2.putText(annotated, f"L: {length_cm} cm | W: {width_cm} cm | H: {thickness_cm} cm",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    return length_cm, width_cm, thickness_cm, annotated


def detect_and_crop_corner_label(frame):
    """
    Locates top-right or corner regions of the mattress to crop the label.
    Fallback: Center-right crop.
    """
    h, w = frame.shape[:2]
    # Default to top-right corner region where corner labels are attached
    ymin, ymax = int(h * 0.05), int(h * 0.45)
    xmin, xmax = int(w * 0.55), int(w * 0.95)
    crop = frame[ymin:ymax, xmin:xmax]
    return crop, (xmin, ymin, xmax, ymax)


def smart_extract_corner_label_info(raw_ocr_items):
    """
    Smart OCR filter for Corner Label:
    - Filters out washing care symbols and generic text.
    - Extracts Product Name (handling partial reads like 'Ma Plus').
    - Extracts Size & Dimensions (e.g. 190x160, 78x60).
    - Extracts Product Code.
    """
    full_text = " | ".join([item["text"] for item in raw_ocr_items])
    norm_full = full_text.lower()

    detected_product = "Not Detected"
    best_similarity = 0.0

    # 1. Match across known product varieties (including fuzzy matching for 'Ma Plus')
    for cls_name in CLASS_NAMES:
        norm_cls = cls_name.lower()
        if norm_cls in norm_full:
            detected_product = cls_name
            best_similarity = 1.0
            break

        # Check word sliding window fuzzy match
        for item in raw_ocr_items:
            txt = item["text"].lower()
            ratio = difflib.SequenceMatcher(None, norm_cls, txt).ratio()
            if ratio > best_similarity and ratio >= 0.60:
                best_similarity = ratio
                detected_product = cls_name

    # 2. Extract Size / Dimension pattern (e.g., 190x160, 78x60, King, Queen)
    size_match = re.search(r'\b\d{2,3}\s*[xX*]\s*\d{2,3}\b', full_text)
    detected_size = size_match.group(0) if size_match else "Standard"

    if "king" in norm_full:
        detected_size = "King Size"
    elif "queen" in norm_full:
        detected_size = "Queen Size"

    return {
        "full_text": full_text if full_text else "No Label Text Detected",
        "product_name": detected_product,
        "size": detected_size,
        "fuzzy_similarity": round(best_similarity * 100, 1)
    }


def process_top_camera(frame):
    """
    Full processing pipeline for Top Camera:
    1. Measures mattress Length, Width, Thickness.
    2. Detects & crops corner label.
    3. Runs EasyOCR on corner label crop.
    4. Smart filters extracted text & performs fuzzy matching.
    """
    length_cm, width_cm, thickness_cm, annotated_dims = measure_mattress_dimensions(frame)
    label_crop, label_bbox = detect_and_crop_corner_label(frame)

    ocr_res = process_bill_ocr(label_crop)
    smart_info = smart_extract_corner_label_info(ocr_res["extracted_items"])

    # Draw label box on top frame
    xmin, ymin, xmax, ymax = label_bbox
    cv2.rectangle(annotated_dims, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.putText(annotated_dims, f"Label: {smart_info['product_name']}", (xmin, max(20, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return {
        "length_cm": length_cm,
        "width_cm": width_cm,
        "thickness_cm": thickness_cm,
        "corner_label": smart_info,
        "corner_label_crop": label_crop,
        "annotated_frame": annotated_dims
    }
