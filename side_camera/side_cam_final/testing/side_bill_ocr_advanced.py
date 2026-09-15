import re
import cv2
import time
import difflib
import numpy as np
import easyocr
import platform

# Master Variety List & Aliases
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
    "purity plus": "Purity plus"
}


# =====================================================================
# OPTIMIZATION 1: AUTO-DESKEW & ORIENTATION PRE-ALIGNMENT
# =====================================================================
def auto_deskew_and_align(img):
    """
    Detects sticker orientation using contour aspect ratio.
    If image is horizontal (Width > Height), auto-rotates 90 deg counterclockwise
    so the sticker is upright (0 deg) BEFORE passing to EasyOCR.
    """
    h, w = img.shape[:2]
    rotated_img = img.copy()
    rotation_angle = 0

    # Convert to grayscale & find prominent rectangular boundaries
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rect_found = False
    if contours:
        # Filter for large contours resembling bill sticker
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.10)]
        if large_contours:
            c = max(large_contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh) if bh > 0 else 1.0

            # If width > height by 1.2x, sticker is horizontal (rotated 90 deg)
            if aspect_ratio > 1.2:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotation_angle = 90
                rect_found = True

    # Fallback aspect ratio check on frame dimensions if contour check didn't trigger
    if not rect_found and (w / float(h)) > 1.3:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotation_angle = 90

    return rotated_img, rotation_angle


# =====================================================================
# OPTIMIZATION 2: CELL-BASED HORIZONTAL LINE SEGMENTATION
# =====================================================================
def segment_table_cells(img):
    """
    Uses OpenCV Morphological line detection to locate black horizontal divider lines.
    Slices the bill sticker into isolated row cell strips before running OCR.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Morphological horizontal line detection
    kernel_len = max(20, w // 4)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # Binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Find row dividing Y-coordinates
    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    y_coords = [0, h]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > (w * 0.4):  # Line must stretch across at least 40% of sticker width
            y_coords.append(y + bh // 2)

    y_coords = sorted(list(set(y_coords)))
    
    # Filter lines that are too close (< 20px apart)
    filtered_y = [y_coords[0]]
    for y in y_coords[1:]:
        if (y - filtered_y[-1]) > 25:
            filtered_y.append(y)
    
    if filtered_y[-1] < h:
        filtered_y.append(h)

    # Crop cell strips
    cell_strips = []
    for i in range(len(filtered_y) - 1):
        y1, y2 = filtered_y[i], filtered_y[i + 1]
        if (y2 - y1) >= 20:
            strip = img[y1:y2, 0:w]
            cell_strips.append({
                "y_range": (y1, y2),
                "image": strip
            })

    return cell_strips if len(cell_strips) > 1 else [{"y_range": (0, h), "image": img}]


# =====================================================================
# OPTIMIZATION 3: EXACT LABEL ANCHORING & DUAL-UNIT CROSS-VALIDATION
# =====================================================================
def validate_dual_dimensions(inches_str, metric_str):
    """
    Cross-validates Product Code Inches (e.g. 75X60X6) against Metric (e.g. 1.905 m X 1.524 m X 15 cm).
    Converts 75" -> 1.905m (190.5cm), 60" -> 1.524m (152.4cm), 6" -> 15cm (15.24cm).
    """
    if not inches_str or not metric_str or inches_str == "Not Detected" or metric_str == "Not Detected":
        return {"valid": True, "notes": "Single format present"}

    try:
        inch_nums = [float(x) for x in re.findall(r'\d+', inches_str)]
        if len(inch_nums) >= 2:
            exp_l_m = round((inch_nums[0] * 2.54) / 100.0, 3)  # 75" -> 1.905 m
            exp_w_m = round((inch_nums[1] * 2.54) / 100.0, 3)  # 60" -> 1.524 m

            # Extract metric numbers
            metric_nums = [float(x) for x in re.findall(r'\d+\.?\d*', metric_str)]
            if len(metric_nums) >= 2:
                act_l_m = metric_nums[0]
                act_w_m = metric_nums[1]

                l_match = abs(exp_l_m - act_l_m) <= 0.05
                w_match = abs(exp_w_m - act_w_m) <= 0.05

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
    except Exception as e:
        pass

    return {"valid": True, "notes": "Validation format bypassed"}


def parse_anchored_side_bill_metadata(raw_text_items, full_text):
    """
    Targeted Regex Anchored Extraction for Variety, Dimensions (Metric & Inches), and MRP.
    Cleans pipe separators and applies non-empty string checks.
    """
    # 0. Create clean continuous text without pipe separators
    clean_text_merged = full_text.replace(" | ", " ").replace("|", " ")
    norm_merged = clean_text_merged.lower()

    # --- 1. VARIETY EXTRACTION ---
    extracted_variety = "Not Detected"
    best_var_sim = 0.0

    # Rule 1: Direct Master Class Name & Alias Lookup in clean text
    # (Matches e.g. "ORTHOLEX", "ORTHOLEX MATTRESS", "MAXI PRO", "DUAL HARMONY")
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

    # Rule 2: Anchor Keyword Regex (e.g. VARIETY: ORTHOLEX, NAME OF THE COMMODITY: ORTHOLEX MATTRESS)
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

    # Rule 3: Sliding Fuzzy Match across items
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
    # Matches e.g., "1.905 m X 1.524 m X 15 cm", "1.905m x 1.524m x 15cm", "1.905 m X1.524 m X 15 cm"
    metric_match = re.search(
        r'\b\d+\.?\d*\s*m\s*[xX*]\s*\d+\.?\d*(?:\s*m)?(?:\s*[xX*]\s*\d+\.?\d*\s*(?:cm|m))?\b',
        clean_text_merged,
        re.IGNORECASE
    )
    if metric_match:
        extracted_metric_dim = metric_match.group(0).strip()
    else:
        # Fallback: scan individual text items for dimension strings
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

    # Pattern A: Currency symbol / prefix e.g. "Rs. 27000.00", "Rs.27000.00", "Rs 27000", "₹27000"
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
        # Pattern B: Keyword anchored e.g. "MRP: 27000.00", "PRICE: 27000"
        mrp_anchor = re.search(r'(?:mrp|price)\s*[:=\-]?\s*(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.\d{2})?)', clean_text_merged, re.IGNORECASE)
        if mrp_anchor:
            val = mrp_anchor.group(1).strip()
            clean_num = val.replace(",", "")
            try:
                if float(clean_num) > 100:
                    extracted_mrp = f"Rs. {val}"
            except ValueError:
                pass

    # Dual-Unit Validation
    unit_val_result = validate_dual_dimensions(extracted_inches_code, extracted_metric_dim)

    return {
        "variety": extracted_variety,
        "metric_dimension": extracted_metric_dim,
        "inches_code": extracted_inches_code,
        "price_mrp": extracted_mrp,
        "match_confidence": round(best_var_sim * 100, 1),
        "dual_unit_validation": unit_val_result
    }


# =====================================================================
# ADVANCED SIDE BILL OCR ENGINE (COMBINING ALL 3 OPTIMIZATIONS)
# =====================================================================
class AdvancedSideBillOCR:
    def __init__(self, use_gpu=True):
        print("[INFO] Initializing Advanced Side Bill OCR Engine (3 Optimizations Active)...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            print("[INFO] EasyOCR initialized with GPU support.")
        except Exception as e:
            print(f"[WARN] GPU init failed ({e}), falling back to CPU...")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def process(self, input_img):
        """
        Executes Full 3-Optimization Pipeline:
        1. Auto-Deskew & Orientation Pre-Alignment
        2. Cell-Based Horizontal Line Segmentation
        3. Single-Pass EasyOCR & Anchored Dual-Unit Extraction
        """
        start_t = time.time()

        # Step 1: Auto-Deskew & Orientation Check
        aligned_img, rotation_angle = auto_deskew_and_align(input_img)

        # Step 2: Cell-Based Horizontal Line Segmentation
        cells = segment_table_cells(aligned_img)

        # Step 3: Run EasyOCR (with 4-rotation fallback if single cell)
        all_raw_items = []
        annotated = aligned_img.copy()
        rot_info = [90, 180, 270] if len(cells) <= 1 else []

        for cell in cells:
            cell_crop = cell["image"]
            y1, y2 = cell["y_range"]
            
            results = self.reader.readtext(cell_crop, rotation_info=rot_info)

            for (bbox, text_res, prob) in results:
                clean_text = text_res.strip()
                if len(clean_text) <= 1 and not clean_text.isdigit():
                    continue
                if prob < 0.25:
                    continue

                all_raw_items.append({
                    "text": clean_text,
                    "confidence": float(prob)
                })

                # Adjust bbox coordinates back to aligned image space
                pts = np.array(bbox, dtype=np.int32)
                pts[:, 1] += y1
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        full_text = " | ".join([item["text"] for item in all_raw_items])
        parsed = parse_anchored_side_bill_metadata(all_raw_items, full_text)
        elapsed = round(time.time() - start_t, 2)

        # Draw summary panel overlay on top-left
        cv2.rectangle(annotated, (10, 10), (520, 160), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (520, 160), (0, 255, 255), 2)
        cv2.putText(annotated, f"VARIETY      : {parsed['variety']} ({parsed['match_confidence']}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"METRIC DIMS  : {parsed['metric_dimension']}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"INCHES CODE  : {parsed['inches_code']}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(annotated, f"PRICE / MRP  : {parsed['price_mrp']}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return {
            "parsed_fields": parsed,
            "rotation_angle_applied": rotation_angle,
            "cells_segmented": len(cells),
            "processing_time_sec": elapsed,
            "annotated_frame": annotated,
            "raw_text": full_text
        }


# Interactive Live Camera & Static Image Test Runner
def run_live_camera_test(camera_index=1):
    engine = AdvancedSideBillOCR()

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print(f"[WARN] Camera Index {camera_index} failed, trying Index 0...")
        cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("[ERROR] No USB camera available for live test!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 65)
    print("ADVANCED SIDE BILL OCR ENGINE - ALL 3 OPTIMIZATIONS ACTIVE")
    print("  1. Auto-Deskew & Orientation Pre-Alignment")
    print("  2. Cell-Based Horizontal Line Segmentation")
    print("  3. Exact Label Anchoring & Dual-Unit Validation")
    print("-" * 65)
    print("Press 'SPACE' or 'p' to PROCESS & EXTRACT current frame.")
    print("Press 'q' or 'ESC' to QUIT.")
    print("=" * 65 + "\n")

    latest_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (10, 10), (540, 50), (20, 20, 20), -1)
        cv2.putText(display_frame, "Press 'SPACE' to PROCESS (All 3 Optimizations) | 'q' to QUIT",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        if latest_annotated is not None:
            thumb = cv2.resize(latest_annotated, (320, 180))
            display_frame[520:700, 940:1260] = thumb
            cv2.rectangle(display_frame, (940, 520), (1260, 700), (0, 255, 0), 2)

        cv2.imshow("Advanced Side Bill OCR (3 Optimizations Active)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('p'), ord(' ')):
            print("\n[INFO] Processing frame through 3-Optimization Pipeline...")
            res = engine.process(frame)
            parsed = res["parsed_fields"]
            latest_annotated = res["annotated_frame"]

            print("=" * 65)
            print(f"EXTRACTION COMPLETE (Took {res['processing_time_sec']}s | Rotated: {res['rotation_angle_applied']} deg | Cells: {res['cells_segmented']}):")
            print(f"  > VARIETY           : {parsed['variety']} (Match Confidence: {parsed['match_confidence']}%)")
            print(f"  > METRIC DIMS       : {parsed['metric_dimension']}")
            print(f"  > INCHES CODE       : {parsed['inches_code']}")
            print(f"  > PRICE / MRP       : {parsed['price_mrp']}")
            print(f"  > DUAL-UNIT VALID   : {parsed['dual_unit_validation']['notes']}")
            print("=" * 65 + "\n")

            cv2.imshow("Advanced OCR Extraction Result", latest_annotated)

        elif key in (ord('q'), 27):
            print("[INFO] Exiting advanced live test...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    cam_idx = 1
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_idx = int(sys.argv[1])
    
    print("=" * 65)
    print("LAUNCHING ADVANCED SIDE BILL OCR LIVE CAMERA TEST")
    print(f"Target Camera Index: {cam_idx}")
    print("=" * 65)

    run_live_camera_test(camera_index=cam_idx)
