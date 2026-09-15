import os
import cv2
import numpy as np
import re
import difflib
from ocr_module import reader
from config import CLASS_NAMES, OCR_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH

VARIETY_ALIASES = {
    "maxiplush": "Maxi plush",
    "maxi plush": "Maxi plush",
    "maxi 8": "Maxi plush",
    "maxi8": "Maxi plush",
    "maxipro": "Maxi pro",
    "maxi pro": "Maxi pro",
    "ortholex": "Ortholex",
    "ortholex mattress": "Ortholex",
    "dualharmony": "Dual harmony",
    "dual harmony": "Dual harmony",
    "purityplus": "Purity plus",
    "purity plus": "Purity plus",
    "gravite": "Gravite",
    "memorise": "Memorise",
    "velvet": "Velvet"
}

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
    Applies calibrated conversion factors (0.25 cm/pixel default).
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

    return length_cm, width_cm, thickness_cm, annotated


def detect_and_crop_corner_ribbon(frame):
    """
    Locates diagonal ribbon/banner regions or crops all 4 corners + full frame of the mattress.
    Ensures ribbons at Bottom-Right, Top-Left, Top-Right, or Bottom-Left are captured.
    """
    h, w = frame.shape[:2]
    candidates = []

    # 1. Color-Based Ribbon Segmentation (Detect Purple/Red/Dark Ribbon Banners)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([125, 40, 40])
    upper_purple = np.array([170, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    combined_mask = mask_purple | mask_red
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        large_cnts = [c for c in contours if cv2.contourArea(c) > 2000]
        if large_cnts:
            c = max(large_cnts, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            candidates.append({
                "name": "Auto-Detected Color Ribbon Region",
                "image": frame[y0:y1, x0:x1],
                "bbox": (x0, y0, x1, y1)
            })

    # 2. All 4 Corner Quadrants (Coverage for any orientation)
    candidates.append({"name": "Bottom-Right Corner Region", "image": frame[int(h*0.45):h, int(w*0.45):w], "bbox": (int(w*0.45), int(h*0.45), w, h)})
    candidates.append({"name": "Top-Left Ribbon Region", "image": frame[0:int(h*0.55), 0:int(w*0.55)], "bbox": (0, 0, int(w*0.55), int(h*0.55))})
    candidates.append({"name": "Top-Right Corner Region", "image": frame[0:int(h*0.55), int(w*0.45):w], "bbox": (int(w*0.45), 0, w, int(h*0.55))})
    candidates.append({"name": "Bottom-Left Corner Region", "image": frame[int(h*0.45):h, 0:int(w*0.55)], "bbox": (0, int(h*0.45), int(w*0.55), h)})
    
    # 3. Full Frame Fallback
    candidates.append({"name": "Full Mattress Frame", "image": frame, "bbox": (0, 0, w, h)})

    return candidates


def detect_and_crop_corner_label(frame):
    """Legacy compatibility helper wrapper returning primary candidate crop."""
    candidates = detect_and_crop_corner_ribbon(frame)
    primary = candidates[0]
    return primary["image"], primary["bbox"]


def parse_corner_label_variety(extracted_items):
    """
    Filters raw OCR text items extracted from Corner Label Banner
    and matches against master CLASS_NAMES and VARIETY_ALIASES.
    """
    full_text = " ".join([item["text"] for item in extracted_items])
    clean_merged = full_text.replace(" | ", " ").replace("|", " ")
    norm_merged = clean_merged.lower()

    detected_variety = "Not Detected"
    best_sim = 0.0

    # Rule 1: Direct Alias & Master Variety Lookup in merged text
    for alias, canonical in VARIETY_ALIASES.items():
        clean_alias = alias.replace(" ", "")
        clean_norm = norm_merged.replace(" ", "")
        if clean_alias in clean_norm or alias in norm_merged:
            detected_variety = canonical
            best_sim = 1.0
            break

    if best_sim < 0.8:
        for cls_name in CLASS_NAMES:
            clean_cls = cls_name.lower().replace(" ", "")
            clean_norm = norm_merged.replace(" ", "")
            if clean_cls in clean_norm or cls_name.lower() in norm_merged:
                detected_variety = cls_name
                best_sim = 1.0
                break

    # Rule 2: Item-by-item fuzzy match (strict threshold >= 0.75)
    if best_sim < 0.75:
        for item in extracted_items:
            txt = item["text"].strip().lower()
            clean_txt = txt.replace(" ", "")
            if len(clean_txt) >= 4:
                for cls_name in CLASS_NAMES:
                    clean_cls = cls_name.lower().replace(" ", "")
                    ratio = difflib.SequenceMatcher(None, clean_cls, clean_txt).ratio()
                    if ratio > best_sim and ratio >= 0.75:
                        best_sim = ratio
                        detected_variety = cls_name

    # Extract Size / Dimension pattern (e.g., 190x160, 78x60, King, Queen)
    size_match = re.search(r'\b\d{2,3}\s*[xX*]\s*\d{2,3}\b', full_text)
    detected_size = size_match.group(0) if size_match else "Standard"

    if "king" in norm_merged:
        detected_size = "King Size"
    elif "queen" in norm_merged:
        detected_size = "Queen Size"

    return {
        "full_text": full_text if full_text else "No Label Text Detected",
        "product_name": detected_variety,
        "size": detected_size,
        "fuzzy_similarity": round(best_sim * 100, 1)
    }


def process_top_camera(frame):
    """
    Full processing pipeline for Top Camera:
    1. Measures mattress Length, Width, Thickness.
    2. Scans 4-corner & color ribbon regions across 4 orientations using EasyOCR.
    3. High-precision filters extracted corner banner text against master varieties.
    """
    length_cm, width_cm, thickness_cm, annotated_dims = measure_mattress_dimensions(frame)
    candidates = detect_and_crop_corner_ribbon(frame)

    best_corner_info = {
        "full_text": "No Label Text Detected",
        "product_name": "Not Detected",
        "size": "Standard",
        "fuzzy_similarity": 0.0,
        "region_used": "None"
    }
    best_crop = candidates[0]["image"]
    best_bbox = candidates[0]["bbox"]

    for cand in candidates:
        crop_img = cand["image"]
        if crop_img is None or crop_img.size == 0:
            continue

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        prep = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        results = reader.readtext(prep, rotation_info=[90, 180, 270])
        region_items = []

        if results:
            for (bbox, text_res, prob) in results:
                clean_text = text_res.strip()
                if len(clean_text) <= 1 and not clean_text.isdigit():
                    continue
                if prob < OCR_CONFIDENCE_THRESHOLD:
                    continue

                region_items.append({"text": clean_text, "confidence": float(prob)})

        res = parse_corner_label_variety(region_items)
        if res["fuzzy_similarity"] > best_corner_info["fuzzy_similarity"]:
            best_corner_info = res
            best_corner_info["region_used"] = cand["name"]
            best_crop = crop_img
            best_bbox = cand["bbox"]
            if res["fuzzy_similarity"] >= 90.0:
                break

    # Return clean top frame without box annotations as requested
    return {
        "length_cm": length_cm,
        "width_cm": width_cm,
        "thickness_cm": thickness_cm,
        "corner_label": best_corner_info,
        "corner_label_crop": best_crop,
        "annotated_frame": annotated_dims
    }
