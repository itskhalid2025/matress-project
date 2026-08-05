import cv2
import numpy as np
import easyocr
from config import OCR_CONFIDENCE_THRESHOLD

print("[INFO] Initializing EasyOCR Reader...")
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("[INFO] EasyOCR initialized with GPU support.")
except Exception as e:
    print(f"[WARN] GPU initialization failed ({e}), falling back to CPU for EasyOCR.")
    reader = easyocr.Reader(['en'], gpu=False)


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def process_bill_ocr(frame):
    """Runs EasyOCR on bill frame, returns text items, average confidence, and annotated frame."""
    annotated = frame.copy()
    preprocessed = preprocess_for_ocr(frame)
    results = reader.readtext(preprocessed, rotation_info=[90, 180, 270])

    extracted_items = []
    total_conf = 0.0

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
            cv2.polylines(annotated, [pts], True, (0, 0, 255), 2)
            cv2.putText(annotated, clean_text, (pts[0][0], max(20, pts[0][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    avg_conf = (total_conf / len(extracted_items)) if extracted_items else 0.0
    full_extracted_text = " | ".join([item["text"] for item in extracted_items])

    return {
        "extracted_items": extracted_items,
        "full_text": full_extracted_text,
        "avg_confidence": round(avg_conf * 100, 2),
        "annotated_frame": annotated
    }


def detect_ocr_presence_fast(frame):
    """
    Fast lightweight text region detector (<2ms) for live video stream aiming.
    Uses morphological gradient to detect dense text/character clusters without full OCR model pass.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Morphological gradient to highlight text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Close horizontal gaps between letters
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_boxes = []
    h, w = gray.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        area = bw * bh
        # Filter for typical text bounding box dimensions
        if area > 300 and area < (h * w * 0.25) and aspect > 1.2 and bh > 10:
            text_boxes.append((x, y, bw, bh))
            
    return text_boxes

