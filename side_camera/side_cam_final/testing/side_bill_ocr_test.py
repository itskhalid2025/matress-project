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
    "dualharmony": "Dual harmony",
    "dual harmony": "Dual harmony",
    "purityplus": "Purity plus",
    "purity plus": "Purity plus"
}


def preprocess_bill_image(img):
    """Enhanced preprocessing for high-contrast OCR extraction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def parse_side_bill_metadata(raw_text_items, full_text):
    """
    High-Precision Parser for Side Bill OCR:
    Extracts Variety, Dimensions, and Price (MRP) from bill OCR text stream.
    """
    norm_full = full_text.lower()

    # --- 1. VARIETY EXTRACTION ---
    extracted_variety = "Not Detected"
    best_var_sim = 0.0

    # Strategy A: Anchor Keyword Search (e.g. VARIETY: MAXI PRO, COMMODITY: MAXIPLUSH)
    anchor_match = re.search(r'(?:variety|commodity|model|item)\s*[:=\-]?\s*([a-zA-Z0-9\s]+)', full_text, re.IGNORECASE)
    if anchor_match:
        cand_text = anchor_match.group(1).strip().lower()
        for cls_name in CLASS_NAMES:
            norm_cls = cls_name.lower()
            if norm_cls in cand_text or cand_text in norm_cls:
                extracted_variety = cls_name
                best_var_sim = 1.0
                break

    # Strategy B: Direct & Alias Matching across extracted items
    if best_var_sim < 0.8:
        for alias, canonical_name in VARIETY_ALIASES.items():
            clean_alias = alias.replace(" ", "")
            clean_full = norm_full.replace(" ", "")
            if clean_alias in clean_full:
                extracted_variety = canonical_name
                best_var_sim = 1.0
                break

    # Strategy C: Fuzzy Matching across items
    if best_var_sim < 0.6:
        for item in raw_text_items:
            txt = item["text"].strip()
            clean_txt = txt.lower()
            for cls_name in CLASS_NAMES:
                ratio = difflib.SequenceMatcher(None, cls_name.lower(), clean_txt).ratio()
                if ratio > best_var_sim and ratio >= 0.60:
                    best_var_sim = ratio
                    extracted_variety = cls_name

    # --- 2. DIMENSION EXTRACTION ---
    extracted_dimension = "Not Detected"

    # Pattern A: Metric dimensions e.g., "1.905 m X 0.915 m", "1.905m x 0.915m"
    metric_match = re.search(r'\b\d+\.?\d*\s*m\s*[xX*]\s*\d+\.?\d*\s*m(?:\s*[xX*]\s*\d+\.?\d*\s*m)?\b', full_text)
    if metric_match:
        extracted_dimension = metric_match.group(0).strip()
    else:
        # Pattern B: Standard inch/cm dimensions e.g., "75x36x8", "190x160", "78 * 60 * 10"
        dim_match = re.search(r'\b\d{2,3}\s*[xX*]\s*\d{2,3}(?:\s*[xX*]\s*\d{1,2})?\b', full_text)
        if dim_match:
            extracted_dimension = dim_match.group(0).strip()
        else:
            # Pattern C: Keyword anchored e.g., "Dimension: 75x36x8"
            anchor_dim = re.search(r'(?:dimension|dim|size)\s*[:=\-]?\s*([0-9\.\s[xX*m]+)', full_text, re.IGNORECASE)
            if anchor_dim:
                extracted_dimension = anchor_dim.group(1).strip()

    # --- 3. PRICE / MRP EXTRACTION ---
    extracted_price = "Not Detected"

    # Pattern A: Currency anchored e.g. "MRP: Rs. 14,999", "PRICE: 12500", "₹ 15999", "RS 14999.00"
    mrp_match = re.search(r'(?:mrp|price|amount|rs\.?|inr|₹)\s*[:=\-]?\s*(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.\d{2})?)', full_text, re.IGNORECASE)
    if mrp_match:
        val = mrp_match.group(1).strip()
        clean_num = val.replace(",", "")
        try:
            if float(clean_num) > 100:
                extracted_price = f"Rs. {val}"
        except ValueError:
            pass

    return {
        "variety": extracted_variety,
        "dimension": extracted_dimension,
        "price": extracted_price,
        "match_confidence": round(best_var_sim * 100, 1)
    }


class HighPrecisionSideBillOCR:
    def __init__(self, use_gpu=True):
        print("[INFO] Initializing High-Precision Side Bill EasyOCR Reader...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            print("[INFO] EasyOCR loaded successfully with GPU.")
        except Exception as e:
            print(f"[WARN] Failed GPU init ({e}), using CPU...")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def process_frame(self, frame):
        """Processes frame, extracts text items, parses fields, and draws annotations."""
        preprocessed = preprocess_bill_image(frame)
        results = self.reader.readtext(preprocessed, rotation_info=[90, 180, 270])

        annotated = frame.copy()
        extracted_items = []

        if results:
            for (bbox, text_res, prob) in results:
                clean_text = text_res.strip()
                if len(clean_text) <= 1 and not clean_text.isdigit():
                    continue
                if prob < 0.25:
                    continue

                extracted_items.append({
                    "text": clean_text,
                    "confidence": float(prob)
                })

                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        full_text = " | ".join([item["text"] for item in extracted_items])
        parsed_fields = parse_side_bill_metadata(extracted_items, full_text)

        # Draw summary overlay card on top left
        cv2.rectangle(annotated, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (450, 140), (0, 255, 255), 2)
        cv2.putText(annotated, f"VARIETY  : {parsed_fields['variety']} ({parsed_fields['match_confidence']}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"DIMENSION: {parsed_fields['dimension']}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"PRICE/MRP: {parsed_fields['price']}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return {
            "parsed_fields": parsed_fields,
            "raw_text": full_text,
            "items_count": len(extracted_items),
            "annotated_frame": annotated
        }


def run_live_camera_test(camera_index=1):
    """Live interactive camera testing mode."""
    ocr_engine = HighPrecisionSideBillOCR()

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

    print("\n" + "=" * 60)
    print("LIVE SIDE BILL OCR TEST STARTED")
    print("Press 'SPACE' or 'p' to PROCESS & EXTRACT current frame.")
    print("Press 'q' or 'ESC' to QUIT.")
    print("=" * 60 + "\n")

    latest_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            time.sleep(0.1)
            continue

        display_frame = frame.copy()

        # Display live helper instructions
        cv2.rectangle(display_frame, (10, 10), (500, 50), (20, 20, 20), -1)
        cv2.putText(display_frame, "Press 'SPACE' to PROCESS & EXTRACT | 'q' to QUIT",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if latest_annotated is not None:
            # Show small preview thumbnail of last processed result
            thumb = cv2.resize(latest_annotated, (320, 180))
            display_frame[520:700, 940:1260] = thumb
            cv2.rectangle(display_frame, (940, 520), (1260, 700), (0, 255, 0), 2)

        cv2.imshow("Live Side Bill OCR Test (Press SPACE to Process)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('p'), ord(' ')):
            print("\n[INFO] Processing image frame...")
            start_t = time.time()
            res = ocr_engine.process_frame(frame)
            elapsed = round(time.time() - start_t, 2)

            parsed = res["parsed_fields"]
            latest_annotated = res["annotated_frame"]

            print("=" * 60)
            print(f"EXTRACTION COMPLETE (Took {elapsed}s):")
            print(f"  > VARIETY   : {parsed['variety']} (Confidence: {parsed['match_confidence']}%)")
            print(f"  > DIMENSION : {parsed['dimension']}")
            print(f"  > PRICE/MRP : {parsed['price']}")
            print(f"  > ITEMS     : {res['items_count']} text blocks extracted")
            print(f"  > RAW TEXT  : {res['raw_text'][:120]}...")
            print("=" * 60 + "\n")

            cv2.imshow("Processed Extraction Result", latest_annotated)

        elif key in (ord('q'), 27):
            print("[INFO] Exiting live test...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    cam_idx = 1
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_idx = int(sys.argv[1])
    run_live_camera_test(camera_index=cam_idx)
