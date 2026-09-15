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


def preprocess_corner_label_image(img):
    """Preprocessing optimized for diagonal corner label banners and printed text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def detect_and_crop_corner_ribbon(frame):
    """
    Locates diagonal ribbon/banner regions or crops all 4 corners + full frame of the mattress.
    Ensures ribbons at Top-Left, Top-Right, Bottom-Left, or Bottom-Right are captured.
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

    # Rule 2: Item-by-item fuzzy match (strict threshold >= 0.75 for short strings)
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

    return {
        "variety": detected_variety,
        "confidence": round(best_sim * 100, 1),
        "raw_text": full_text
    }


class HighPrecisionCornerLabelOCR:
    def __init__(self, use_gpu=True):
        print("[INFO] Initializing High-Precision Corner Label OCR Engine...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            print("[INFO] EasyOCR initialized with GPU support.")
        except Exception as e:
            print(f"[WARN] GPU initialization failed ({e}), falling back to CPU...")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def process_frame(self, frame):
        """
        Processes top camera frame, scans all 4 corners + color ribbon regions across 4 orientations,
        extracts diagonal banner text, and filters against master variety names.
        """
        start_t = time.time()
        annotated = frame.copy()
        candidates = detect_and_crop_corner_ribbon(frame)

        best_result = {
            "variety": "Not Detected",
            "confidence": 0.0,
            "raw_text": "",
            "region_used": "None"
        }
        all_extracted_items = []

        for cand in candidates:
            crop_img = cand["image"]
            if crop_img is None or crop_img.size == 0:
                continue

            preprocessed = preprocess_corner_label_image(crop_img)

            # Scan with 4-orientation rotation info (0, 90, 180, 270) to catch diagonal text
            results = self.reader.readtext(preprocessed, rotation_info=[90, 180, 270])

            region_items = []
            if results:
                for (bbox, text_res, prob) in results:
                    clean_text = text_res.strip()
                    if len(clean_text) <= 1 and not clean_text.isdigit():
                        continue
                    if prob < 0.25:
                        continue

                    item_info = {"text": clean_text, "confidence": float(prob)}
                    region_items.append(item_info)
                    all_extracted_items.append(item_info)

                    # Draw bounding boxes on annotated frame
                    bx0, by0 = cand["bbox"][0], cand["bbox"][1]
                    pts = np.array(bbox, dtype=np.int32)
                    pts[:, 0] += bx0
                    pts[:, 1] += by0
                    cv2.polylines(annotated, [pts], True, (255, 0, 255), 2)

            res = parse_corner_label_variety(region_items)
            if res["confidence"] > best_result["confidence"]:
                best_result = res
                best_result["region_used"] = cand["name"]
                if res["confidence"] >= 90.0:
                    break

        elapsed = round(time.time() - start_t, 2)

        # Draw summary overlay on top-left of annotated frame
        cv2.rectangle(annotated, (10, 10), (480, 80), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (480, 80), (255, 0, 255), 2)
        cv2.putText(annotated, f"CORNER LABEL: {best_result['variety']} ({best_result['confidence']}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        cv2.putText(annotated, f"REGION USED : {best_result['region_used']}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return {
            "parsed_info": best_result,
            "processing_time_sec": elapsed,
            "all_extracted_items": all_extracted_items,
            "annotated_frame": annotated
        }


# Interactive Live Camera & Image Test Runner
def run_live_corner_test(camera_index=2):
    engine = HighPrecisionCornerLabelOCR()

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print(f"[WARN] Camera Index {camera_index} failed, trying Index 0...")
        cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("[ERROR] No camera available for live test!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 65)
    print("HIGH-PRECISION CORNER LABEL OCR TEST (ALL 4 CORNERS ACTIVE)")
    print("Press 'SPACE' or 'p' to PROCESS & EXTRACT corner label.")
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
        cv2.putText(display_frame, "Press 'SPACE' to PROCESS CORNER LABEL | 'q' to QUIT",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)

        if latest_annotated is not None:
            dh, dw = display_frame.shape[:2]
            tw, th = min(320, dw // 3), min(180, dh // 3)
            if dh > th and dw > tw:
                thumb = cv2.resize(latest_annotated, (tw, th))
                display_frame[dh-th:dh, dw-tw:dw] = thumb
                cv2.rectangle(display_frame, (dw-tw, dh-th), (dw, dh), (255, 0, 255), 2)

        cv2.imshow("Corner Label Banner OCR Test (All 4 Corners)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('p'), ord(' ')):
            print("\n[INFO] Processing corner label frame...")
            res = engine.process_frame(frame)
            parsed = res["parsed_info"]
            latest_annotated = res["annotated_frame"]

            print("=" * 65)
            print(f"CORNER LABEL EXTRACTION COMPLETE (Took {res['processing_time_sec']}s):")
            print(f"  > DETECTED VARIETY : {parsed['variety']} (Match Confidence: {parsed['confidence']}%)")
            print(f"  > REGION USED      : {parsed['region_used']}")
            print(f"  > RAW TEXT         : {parsed['raw_text']}")
            print("=" * 65 + "\n")

            cv2.imshow("Corner Label Result", latest_annotated)

        elif key in (ord('q'), 27):
            print("[INFO] Exiting corner label test...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    print("=" * 65)
    print("RUNNING CORNER LABEL OCR TEST SUITE (SAMPLE DATA)")
    print("=" * 65)

    sample_items = [
        {"text": "MAXI PLUSH", "confidence": 0.95},
        {"text": "FOAM", "confidence": 0.88}
    ]

    res = parse_corner_label_variety(sample_items)
    print("\n--- SAMPLE RESULT ---")
    print(f"  > DETECTED VARIETY : {res['variety']} (Confidence: {res['confidence']}%)")
    print(f"  > RAW TEXT         : {res['raw_text']}")
    print("=" * 65)

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        cam_idx = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2
        run_live_corner_test(camera_index=cam_idx)
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        run_live_corner_test(camera_index=int(sys.argv[1]))
    else:
        run_live_corner_test(camera_index=0)
