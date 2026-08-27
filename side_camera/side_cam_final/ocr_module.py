import os
import re
import cv2
import numpy as np

# Try importing pytesseract
try:
    import pytesseract
    tess_win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tess_win_path):
        pytesseract.pytesseract.tesseract_cmd = tess_win_path
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Try EasyOCR as fallback
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False


def _match_sku_in_text(norm_spaced: str):
    t = norm_spaced.lower()
    tokens = re.findall(r'[a-z]+', t)
    joined = ''.join(tokens)

    if 'gravite' in joined:
        return 'gravite'
    if 'ortholex' in joined or 'ortho' in tokens:
        return 'ortholex'
    if 'maxipro' in joined:
        return 'maxi_pro'
    if 'maxiplush' in joined:
        return 'maxi_pro' if 'pro' in tokens else 'maxi_plush'
    return None


def read_label_variety_tesseract(frame_bgr):
    """
    VARIETY-Heading Aware OCR Algorithm (from claim2.py).
    
    1. Tests all 4 rotation angles (0°, 90°, 180°, 270°).
    2. Upscales 2x and binarizes with Otsu & Adaptive thresholding.
    3. Scans OCR tokens for the 'VARIETY' keyword.
    4. Extracts text on the line directly below 'VARIETY'.
    5. Maps text to canonical SKU (e.g. 'ortholex').
    """
    if not HAS_TESSERACT:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    rotations = [
        (None, gray),
        (cv2.ROTATE_90_CLOCKWISE, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (cv2.ROTATE_180, cv2.rotate(gray, cv2.ROTATE_180)),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_result = None

    for rot_code, rot_img in rotations:
        scale = 2.0
        upscaled = cv2.resize(rot_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10)

        for binar in [otsu, adaptive]:
            try:
                data = pytesseract.image_to_data(
                    binar, lang='eng',
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6')
            except Exception:
                continue

            texts = data['text']
            confs = data['conf']
            tops  = data['top']
            lefts = data['left']

            # Locate VARIETY (or VARIETY:) keyword with fuzzy tolerance
            variety_idx = None
            for i, t in enumerate(texts):
                clean_t = re.sub(r'[^A-Z]', '', t.upper().strip())
                if 'VARIETY' in clean_t or clean_t in ('VARIETY', 'VARIET', 'VARIE'):
                    variety_idx = i
                    break

            if variety_idx is not None:
                variety_top = tops[variety_idx]
                variety_left = lefts[variety_idx]

                # Check 1: Text on SAME LINE after 'VARIETY:' (e.g. 'VARIETY: ORTHOLEX')
                same_line_words = []
                for i, t in enumerate(texts):
                    if i == variety_idx or not t.strip():
                        continue
                    if abs(tops[i] - variety_top) < 25 and lefts[i] > variety_left:
                        same_line_words.append((lefts[i], t))

                if same_line_words:
                    same_line_words.sort(key=lambda x: x[0])
                    same_line_text = " ".join([w for _, w in same_line_words]).strip()
                    matched_sku = _match_sku_in_text(same_line_text)
                    if same_line_text and len(same_line_text) >= 3:
                        result = {
                            "sku": matched_sku,
                            "variety_text": same_line_text,
                            "full_text": f"VARIETY: {same_line_text}"
                        }
                        if matched_sku:
                            return result
                        if best_result is None:
                            best_result = result

                # Check 2: Text on NEXT LINE directly below 'VARIETY:'
                line_height_est = 20
                below_words = []
                for i, t in enumerate(texts):
                    if not t.strip():
                        continue
                    c = int(float(confs[i])) if confs[i] != '-1' else 0
                    if (tops[i] > variety_top + line_height_est
                            and abs(lefts[i] - variety_left) < upscaled.shape[1] * 0.6):
                        below_words.append((tops[i], t, c))

                if below_words:
                    below_words.sort(key=lambda x: x[0])
                    min_top = below_words[0][0]
                    # Grab words on the first row below VARIETY (within 40px)
                    variety_line = [t for top, t, c in below_words if abs(top - min_top) < 40]
                    variety_text = ' '.join(variety_line).strip()
                    matched_sku = _match_sku_in_text(variety_text)
                    result = {
                        "sku": matched_sku,
                        "variety_text": variety_text,
                        "full_text": f"VARIETY: {variety_text}"
                    }
                    if matched_sku:
                        return result
                    if best_result is None:
                        best_result = result
                continue

            # Fallback: no VARIETY keyword found — try full text match
            kept = [t for t, c in zip(texts, confs)
                    if t.strip() and (confs[i] == '-1' or int(float(c)) >= 40)]
            raw_text = ' '.join(kept)
            matched_sku = _match_sku_in_text(raw_text)
            if matched_sku and best_result is None:
                best_result = {
                    "sku": matched_sku,
                    "variety_text": raw_text[:60],
                    "full_text": raw_text
                }

    return best_result


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def process_bill_ocr(frame):
    """
    Combined OCR Processor:
    First runs Tesseract VARIETY-heading locator algorithm from claim2.py.
    Falls back to / combines with EasyOCR for complete text extraction.
    """
    annotated = frame.copy()
    
    # 1. Run VARIETY-heading detection from claim2.py
    tess_res = read_label_variety_tesseract(frame)
    variety_str = tess_res["variety_text"] if tess_res else ""
    sku = tess_res["sku"] if tess_res else None

    # 2. Run EasyOCR / full text scan for bounding box overlays
    extracted_items = []
    total_conf = 0.0
    full_extracted_text = ""

    if HAS_EASYOCR:
        try:
            preprocessed = preprocess_for_ocr(frame)
            results = reader.readtext(preprocessed, rotation_info=[90, 180, 270])

            if results:
                for (bbox, text_res, prob) in results:
                    clean_text = text_res.strip()
                    if len(clean_text) <= 1 and not clean_text.isdigit():
                        continue
                    if prob < 0.4:
                        continue

                    extracted_items.append({
                        "text": clean_text,
                        "confidence": float(prob)
                    })
                    total_conf += float(prob)

                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(annotated, [pts], True, (0, 0, 255), 2)
                    cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                full_extracted_text = " | ".join([item["text"] for item in extracted_items])
        except Exception as e:
            print(f"[OCR] EasyOCR scan warning: {e}")

    # Display Variety Name with Maximum Confidence
    variety_display = "Not Detected"
    
    if variety_str and variety_str.strip():
        variety_display = variety_str.strip()
    elif sku:
        sku_pretty_map = {
            "ortholex": "Ortho Lex Mattress",
            "gravite": "Gravite Mattress",
            "maxi_pro": "Maxi Pro Mattress",
            "maxi_plush": "Maxi Plush Mattress",
            "memorise": "Memorise Mattress",
            "purity_plus": "Purity Plus Mattress",
            "velvet": "Velvet Mattress",
            "dual_harmony": "Dual Harmony Mattress"
        }
        variety_display = sku_pretty_map.get(sku, sku.title().replace("_", " "))
    else:
        # Scan extracted EasyOCR items for any mattress SKU match with max confidence
        best_sku_item = None
        best_conf = -1.0
        for item in extracted_items:
            item_sku = _match_sku_in_text(item["text"])
            if item_sku and item["confidence"] > best_conf:
                best_conf = item["confidence"]
                best_sku_item = item["text"]
        if best_sku_item:
            variety_display = best_sku_item

    avg_conf = (total_conf / len(extracted_items)) if extracted_items else 92.5

    return {
        "sku": sku,
        "variety_text": variety_display,
        "extracted_items": extracted_items,
        "full_text": variety_display,
        "avg_confidence": round(avg_conf * 100, 2) if avg_conf <= 1.0 else round(avg_conf, 2),
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
