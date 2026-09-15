import re
import cv2
import difflib
import numpy as np
import easyocr
import platform
from config import OCR_CONFIDENCE_THRESHOLD, CLASS_NAMES

VARIETY_ALIASES = {
    "maxiplush": "Maxi plush",
    "maxi plush": "Maxi plush",
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

print("[INFO] Initializing High-Precision EasyOCR Reader...")
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("[INFO] EasyOCR initialized with GPU support.")
except Exception as e:
    print(f"[WARN] GPU initialization failed ({e}), falling back to CPU for EasyOCR.")
    reader = easyocr.Reader(['en'], gpu=False)


# =====================================================================
# OPTIMIZATION 1: AUTO-DESKEW & ORIENTATION PRE-ALIGNMENT
# =====================================================================
def auto_deskew_and_align(img):
    """
    Detects sticker orientation using contour aspect ratio of detected sticker region.
    If sticker region width > height, auto-rotates 90 deg counterclockwise.
    """
    h, w = img.shape[:2]
    rotated_img = img.copy()
    rotation_angle = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.15)]
        if large_contours:
            c = max(large_contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh) if bh > 0 else 1.0

            if aspect_ratio > 1.25:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotation_angle = 90

    return rotated_img, rotation_angle


# =====================================================================
# OPTIMIZATION 2: CELL-BASED HORIZONTAL LINE SEGMENTATION
# =====================================================================
def segment_table_cells(img):
    """
    Uses OpenCV Morphological line detection to locate horizontal divider lines.
    Slices the bill sticker into isolated row cell strips before running OCR.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    kernel_len = max(20, w // 4)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    y_coords = [0, h]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > (w * 0.4):
            y_coords.append(y + bh // 2)

    y_coords = sorted(list(set(y_coords)))
    filtered_y = [y_coords[0]]
    for y in y_coords[1:]:
        if (y - filtered_y[-1]) > 25:
            filtered_y.append(y)
    
    if filtered_y[-1] < h:
        filtered_y.append(h)

    cell_strips = []
    for i in range(len(filtered_y) - 1):
        y1, y2 = filtered_y[i], filtered_y[i + 1]
        if (y2 - y1) >= 20:
            cell_strips.append({"y_range": (y1, y2), "image": img[y1:y2, 0:w]})

    return cell_strips if len(cell_strips) > 1 else [{"y_range": (0, h), "image": img}]


# =====================================================================
# OPTIMIZATION 3: EXACT LABEL ANCHORING & DUAL-UNIT CROSS-VALIDATION
# =====================================================================
def validate_dual_dimensions(inches_str, metric_str):
    """
    Cross-validates Product Code Inches (e.g. 75X60X5) against Metric (e.g. 1.905 m X 1.524 m X 15 cm).
    """
    if not inches_str or not metric_str or inches_str == "Not Detected" or metric_str == "Not Detected":
        return {"valid": True, "notes": "Single format present"}

    try:
        inch_nums = [float(x) for x in re.findall(r'\d+', inches_str)]
        if len(inch_nums) >= 2:
            exp_l_m = round((inch_nums[0] * 2.54) / 100.0, 3)
            exp_w_m = round((inch_nums[1] * 2.54) / 100.0, 3)

            metric_nums = [float(x) for x in re.findall(r'\d+\.?\d*', metric_str)]
            if len(metric_nums) >= 2:
                act_l_m = metric_nums[0]
                act_w_m = metric_nums[1]

                l_match = abs(exp_l_m - act_l_m) <= 0.08
                w_match = abs(exp_w_m - act_w_m) <= 0.08

                if l_match and w_match:
                    return {
                        "valid": True,
                        "notes": f"Validated: {inch_nums[0]}\"x{inch_nums[1]}\" ({exp_l_m}m x {exp_w_m}m) matches metric read"
                    }
                else:
                    return {
                        "valid": False,
                        "notes": f"Discrepancy: Inches code {inches_str} predicts {exp_l_m}m x {exp_w_m}m, read {act_l_m}m x {act_w_m}m"
                    }
    except Exception:
        pass

    return {"valid": True, "notes": "Validation format bypassed"}


def parse_anchored_side_bill_metadata(raw_text_items, full_text):
    """
    Targeted Regex Anchored Extraction for Variety, Metric Dims, Inches Code, and Price/MRP.
    """
    clean_text_merged = full_text.replace(" | ", " ").replace("|", " ")
    norm_merged = clean_text_merged.lower()

    # --- 1. VARIETY EXTRACTION ---
    extracted_variety = "Not Detected"
    best_var_sim = 0.0

    # Rule A: Direct Master Class & Alias Lookup
    for alias, canonical_name in VARIETY_ALIASES.items():
        if alias in norm_merged:
            extracted_variety = canonical_name
            best_var_sim = 1.0
            break

    if best_var_sim < 0.8:
        for cls_name in CLASS_NAMES:
            if cls_name.lower() in norm_merged:
                extracted_variety = cls_name
                best_var_sim = 1.0
                break

    # Rule B: Anchor Keyword Regex
    if best_var_sim < 0.8:
        anchor_var = re.search(r'(?:variety|commodity|name of the commodity)\s*[:=\-]?\s*([a-zA-Z0-9\s]+)', clean_text_merged, re.IGNORECASE)
        if anchor_var:
            cand = anchor_var.group(1).strip()
            if cand and len(cand) >= 2:
                norm_cand = cand.lower()
                for cls_name in CLASS_NAMES:
                    norm_cls = cls_name.lower()
                    if norm_cls in norm_cand or (len(norm_cand) >= 3 and norm_cand in norm_cls):
                        extracted_variety = cls_name
                        best_var_sim = 1.0
                        break

    # Rule C: Sliding Fuzzy Match
    if best_var_sim < 0.6:
        for item in raw_text_items:
            clean_txt = item["text"].strip().lower()
            if len(clean_txt) >= 3:
                for cls_name in CLASS_NAMES:
                    ratio = difflib.SequenceMatcher(None, cls_name.lower(), clean_txt).ratio()
                    if ratio > best_var_sim and ratio >= 0.60:
                        best_var_sim = ratio
                        extracted_variety = cls_name

    # --- 2. METRIC DIMENSIONS EXTRACTION ---
    extracted_metric_dim = "Not Detected"
    metric_match = re.search(
        r'\b\d+\.?\d*\s*m\s*[xX*]\s*\d+\.?\d*(?:\s*m)?(?:\s*[xX*]\s*\d+\.?\d*\s*(?:cm|m))?\b',
        clean_text_merged,
        re.IGNORECASE
    )
    if metric_match:
        extracted_metric_dim = metric_match.group(0).strip()
    else:
        for item in raw_text_items:
            m_sub = re.search(r'\b\d+\.?\d*\s*m\s*[xX*]\s*\d+\.?\d*', item["text"], re.IGNORECASE)
            if m_sub:
                extracted_metric_dim = item["text"].strip()
                break

    # --- 3. INCHES PRODUCT CODE EXTRACTION ---
    extracted_inches_code = "Not Detected"
    code_match = re.search(r'(?:product code|code)\s*[:=\-]?\s*(\d{2,3}\s*[xX*]\s*\d{2,3}\s*[xX*]\s*\d{1,2})', clean_text_merged, re.IGNORECASE)
    if code_match:
        extracted_inches_code = code_match.group(1).strip()
    else:
        inch_match = re.search(r'\b\d{2}\s*[xX*]\s*\d{2}\s*[xX*]\s*\d{1,2}\b', clean_text_merged)
        if inch_match:
            extracted_inches_code = inch_match.group(0).strip()

    # --- 4. MRP / PRICE EXTRACTION ---
    extracted_mrp = "Not Detected"
    mrp_standalone = re.search(r'\b(?:rs\.?|inr|₹)\s*[:=\-]?\s*([0-9,]+(?:\.\d{2})?)\b', clean_text_merged, re.IGNORECASE)
    if mrp_standalone:
        val = mrp_standalone.group(1).strip()
        clean_num = val.replace(",", "")
        try:
            if float(clean_num) > 100:
                extracted_mrp = f"Rs. {val}"
        except ValueError:
            pass

    if extracted_mrp == "Not Detected":
        mrp_anchor = re.search(r'(?:mrp|price)\s*[:=\-]?\s*(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.\d{2})?)', clean_text_merged, re.IGNORECASE)
        if mrp_anchor:
            val = mrp_anchor.group(1).strip()
            clean_num = val.replace(",", "")
            try:
                if float(clean_num) > 100:
                    extracted_mrp = f"Rs. {val}"
            except ValueError:
                pass

    unit_val_result = validate_dual_dimensions(extracted_inches_code, extracted_metric_dim)

    return {
        "variety": extracted_variety,
        "metric_dimension": extracted_metric_dim,
        "inches_code": extracted_inches_code,
        "price_mrp": extracted_mrp,
        "match_confidence": round(best_var_sim * 100, 1),
        "dual_unit_validation": unit_val_result
    }


def process_bill_ocr(frame):
    """
    Runs 3-Optimization Advanced Side Bill OCR Engine:
    - Auto-Deskew & Orientation Pre-alignment
    - Cell-Based Horizontal Line Segmentation
    - Anchored Metadata Parsing (Variety, Metric Dims, Inches Code, Price/MRP)
    """
    annotated = frame.copy()
    aligned_frame, rotation_angle = auto_deskew_and_align(frame)
    cells = segment_table_cells(aligned_frame)

    extracted_items = []
    total_conf = 0.0
    rot_info = [90, 180, 270] if len(cells) <= 1 else []

    for cell in cells:
        cell_crop = cell["image"]
        y1, y2 = cell["y_range"]
        results = reader.readtext(cell_crop, rotation_info=rot_info)

        if results:
            for (bbox, text_res, prob) in results:
                clean_text = text_res.strip()
                if len(clean_text) <= 1 and not clean_text.isdigit():
                    continue
                if prob < OCR_CONFIDENCE_THRESHOLD:
                    continue

                extracted_items.append({
                    "text": clean_text,
                    "confidence": float(prob)
                })
                total_conf += float(prob)

                pts = np.array(bbox, dtype=np.int32)
                pts[:, 1] += y1
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    avg_conf = (total_conf / len(extracted_items)) if extracted_items else 0.0
    full_extracted_text = " | ".join([item["text"] for item in extracted_items])
    parsed_fields = parse_anchored_side_bill_metadata(extracted_items, full_extracted_text)

    # Draw live overlay box on top left
    cv2.rectangle(annotated, (10, 10), (450, 140), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (450, 140), (0, 255, 255), 2)
    cv2.putText(annotated, f"VARIETY  : {parsed_fields['variety']} ({parsed_fields['match_confidence']}%)",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated, f"METRIC   : {parsed_fields['metric_dimension']}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(annotated, f"PRICE/MRP: {parsed_fields['price_mrp']}",
                (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return {
        "extracted_items": extracted_items,
        "full_text": full_extracted_text,
        "parsed_fields": parsed_fields,
        "avg_confidence": round(avg_conf * 100, 2),
        "annotated_frame": annotated
    }


def detect_ocr_presence_fast(frame):
    """
    Fast lightweight text region detector (<2ms) for live video stream aiming.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_boxes = []
    h, w = gray.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        area = bw * bh
        if area > 300 and area < (h * w * 0.25) and aspect > 1.2 and bh > 10:
            text_boxes.append((x, y, bw, bh))
            
    return text_boxes
