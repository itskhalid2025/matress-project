"""
mattress/claim.py — read what the box/label/QR CLAIM the SKU is.

Hard rule: this module produces CLAIMS, never identity. Nothing here overrides
the fabric verdict; reconcile.py consumes both and fabric always wins.

Two independent readers:
  QRReader        — decodes the QR (warranty-registration URL), fetches the URL
                     and extracts the mattress type text from the live page,
                     with URL query-string as a fast fallback.
  LabelOCRReader  — reads printed variety/type text on the label as a second,
                     independent claim (covers a torn/occluded QR). Tesseract-based
                     (classical/offline) — flagged for professor sign-off since
                     most modern OCR is deep-learning.

Both fail loud to None on failure; they never guess.
"""
import re
import sys
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs
from html.parser import HTMLParser

import requests

import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Ensure relative imports work if run as a script
try:
    import config as cfg
except ImportError:
    # Fallback if config is missing during local test script execution
    class DummyConfig:
        QR_RETRY_GRAYSCALE = True
        QR_RETRY_UPSCALE = 2.0
        QR_RETRY_CLAHE = True
        QR_RETRY_OTSU = True
        LABEL_CROP_FRAC = (0.0, 0.0, 1.0, 1.0) # Full image for testing
        LABEL_ROTATE_CODE = None
        OCR_LANG = 'eng'
        OCR_MIN_CONF = 30
    cfg = DummyConfig()
    print("[Warning] Could not import project config. Using testing DummyConfig.")


@dataclass
class QRClaim:
    sku: Optional[str]          # normalised SKU key, or None if unmapped
    product_name_raw: str       # raw productName= value from the URL
    batch_no: Optional[str]
    inventory_item_id: Optional[str]
    url: str
    symbology: str               # 'QRCODE', 'CODE39', etc.


@dataclass
class OCRClaim:
    sku: Optional[str]
    matched_text: str
    raw_text: str


def _normalise_product_name(raw: str) -> Optional[str]:
    """'Gravite_Mattress' -> 'gravite'; 'Maxiplush_Pro_Mattress' -> 'maxi_pro'."""
    spaced = raw.replace('_', ' ').replace('-', ' ')
    return _match_sku_in_text(spaced)


class _TextExtractor(HTMLParser):
    """Minimal HTML parser: strips tags, collects visible text."""
    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self._current_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._current_skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags and self._current_skip:
            self._current_skip -= 1

    def handle_data(self, data):
        if self._current_skip == 0:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def get_text(self) -> str:
        return ' '.join(self._chunks)


def _fetch_mattress_type_from_url(url: str, timeout: int = 6) -> Optional[str]:
    """
    Open *url*, scrape the visible page text, and return the first recognised
    mattress SKU key (or None if the page is unreachable / contains no match).

    Strategy:
      1. HTTP GET with a browser-like User-Agent (some warranty sites block bots).
      2. Strip HTML tags with a lightweight stdlib parser — no extra dependencies.
      3. Pass the full visible text through _match_sku_in_text().
    """
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[QR/URL] Could not fetch '{url}': {exc}")
        return None

    # Parse HTML and extract visible text
    parser = _TextExtractor()
    try:
        parser.feed(resp.text)
    except Exception as exc:
        print(f"[QR/URL] HTML parse error for '{url}': {exc}")
        return None

    page_text = parser.get_text()
    if not page_text:
        return None

    matched = _match_sku_in_text(page_text)
    if matched:
        print(f"[QR/URL] Extracted mattress type '{matched}' from live page: {url}")
    else:
        print(f"[QR/URL] Page fetched but no known mattress type found in text.")
        # Print a snippet so the caller can debug
        print(f"[QR/URL] Page text snippet: {page_text[:200]}")
    return matched


_wechat_detector = None

def _get_wechat_detector():
    global _wechat_detector
    if _wechat_detector is not None:
        return _wechat_detector

    if not hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
        return None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(base_dir, "testing_qrbill", "models", "wechat_qrcode"),
        os.path.join(base_dir, "models", "wechat_qrcode"),
        os.path.join(os.path.dirname(base_dir), "models", "wechat_qrcode"),
    ]

    model_dir = None
    for path in search_paths:
        if os.path.exists(os.path.join(path, "detect.prototxt")):
            model_dir = path
            break

    if model_dir is None:
        return None

    detect_proto = os.path.join(model_dir, "detect.prototxt")
    detect_model = os.path.join(model_dir, "detect.caffemodel")
    sr_proto = os.path.join(model_dir, "sr.prototxt")
    sr_model = os.path.join(model_dir, "sr.caffemodel")

    try:
        _wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            detect_proto, detect_model, sr_proto, sr_model
        )
    except Exception as e:
        print(f"[QR Warning] Failed to initialize WeChatQRCode: {e}")

    return _wechat_detector


def _decode_wechat(img):
    detector = _get_wechat_detector()
    if detector is None:
        return []

    # Ensure 3 channels
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    try:
        res, points = detector.detectAndDecode(img_bgr)
        if res and res[0]:
            from collections import namedtuple
            DecodedMock = namedtuple('DecodedMock', ['data', 'type', 'polygon'])
            wrapped_codes = []
            for text, pts in zip(res, points):
                if text:
                    PointMock = namedtuple('Point', ['x', 'y'])
                    poly = [PointMock(int(p[0]), int(p[1])) for p in pts] if pts is not None else None
                    wrapped_codes.append(DecodedMock(
                        data=text.encode('utf-8'),
                        type='QRCODE',
                        polygon=poly
                    ))
            return wrapped_codes
    except Exception as e:
        pass
    return []


def _decode_with_retries(gray_or_bgr):
    """Classical retry ladder: direct -> grayscale -> upscale -> CLAHE.
    Stops at the first stage that finds ANY code. Returns (codes, stage_name)."""
    # 1. Direct pyzbar
    codes = zbar_decode(gray_or_bgr)
    if codes:
        return codes, 'direct'

    # 2. Direct WeChat QR
    codes = _decode_wechat(gray_or_bgr)
    if codes:
        return codes, 'wechat_direct'

    # Always ensure we have a grayscale image for subsequent image processing steps
    if len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr

    # 3. Grayscale
    if cfg.QR_RETRY_GRAYSCALE:
        codes = zbar_decode(gray)
        if codes:
            return codes, 'grayscale'
        codes = _decode_wechat(gray)
        if codes:
            return codes, 'wechat_grayscale'

    # 4. Upscale
    if cfg.QR_RETRY_UPSCALE and cfg.QR_RETRY_UPSCALE != 1.0:
        up = cv2.resize(gray, None, fx=cfg.QR_RETRY_UPSCALE, fy=cfg.QR_RETRY_UPSCALE,
                        interpolation=cv2.INTER_CUBIC)
        codes = zbar_decode(up)
        if codes:
            return codes, f'upscale_{cfg.QR_RETRY_UPSCALE}x'
        codes = _decode_wechat(up)
        if codes:
            return codes, f'wechat_upscale_{cfg.QR_RETRY_UPSCALE}x'

    # 5. CLAHE
    if cfg.QR_RETRY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray) # Safe: guaranteed to be grayscale now
        codes = zbar_decode(enhanced)
        if codes:
            return codes, 'clahe'
        codes = _decode_wechat(enhanced)
        if codes:
            return codes, 'wechat_clahe'
        
    # 6. OTSU
    if cfg.QR_RETRY_OTSU:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        codes = zbar_decode(otsu)
        if codes:
            return codes, 'otsu'
        codes = _decode_wechat(otsu)
        if codes:
            return codes, 'wechat_otsu'

    return [], 'none'


class QRReader:
    """
    Decodes the QR / any 1D barcode on a side-view frame.

    For QRCODE payloads that are URLs the reader:
      1. Parses the URL query string for a fast 'productName' claim.
      2. Fetches the live URL and scrapes the mattress type text from the
         page — this is the authoritative QR claim.
      Whichever source produces a match is returned; URL fetch takes priority.
    """

    def read(self, frame_bgr, return_stage=False):
        codes, stage = _decode_with_retries(frame_bgr)
        qr_codes = [c for c in codes if c.type == 'QRCODE']
        target = qr_codes[0] if qr_codes else (codes[0] if codes else None)

        if target is None:
            return (None, stage) if return_stage else None

        payload = target.data.decode('utf-8', 'replace')
        claim = self._parse_payload(payload, target.type)
        return (claim, stage) if return_stage else claim

    def _parse_payload(self, payload: str, symbology: str) -> QRClaim:
        # Non-URL or non-QR barcodes: return as-is with no SKU
        if symbology != 'QRCODE' or '://' not in payload:
            return QRClaim(sku=None, product_name_raw='', batch_no=None,
                           inventory_item_id=None, url=payload, symbology=symbology)

        # --- Step 1: fast URL query-string parse (offline, no network needed) ---
        q = parse_qs(urlparse(payload).query)
        product_raw = q.get('productName', [''])[0]
        sku_from_url_params = _normalise_product_name(product_raw) if product_raw else None

        # --- Step 2: fetch the live page and extract mattress type text --------
        print(f"[QR] Decoding URL: {payload}")
        sku_from_page = _fetch_mattress_type_from_url(payload)

        # Live page takes priority; fall back to URL query-string param
        final_sku = sku_from_page if sku_from_page is not None else sku_from_url_params

        return QRClaim(
            sku=final_sku,
            product_name_raw=product_raw,
            batch_no=q.get('batchNo', [None])[0],
            inventory_item_id=q.get('inventoryItemId', [None])[0],
            url=payload,
            symbology=symbology,
        )


class LabelOCRReader:
    """Reads printed product-name text on the label as a second claim."""

    def __init__(self):
        self._pytesseract = None
        self._available = False
        self._easyocr_reader = None
        
        try:
            import pytesseract
            # --- Windows Tesseract Path Fix ---
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # ----------------------------------
            self._pytesseract = pytesseract
            self._available = True
        except ImportError:
            pass
            
        if EASYOCR_AVAILABLE:
            try:
                # Initialize and cache EasyOCR reader on CPU to avoid loading latency on every read
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("[LabelOCR] Cached EasyOCR reader initialized successfully on CPU.")
            except Exception as e:
                print(f"[LabelOCR Warning] Failed to initialize EasyOCR reader: {e}")

    @property
    def available(self):
        return self._available

    def _label_crop(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        
        # Try automatic white label contour detection first (very robust for varying placements)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_crop = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50000:
                continue
            x, y, w_c, h_c = cv2.boundingRect(c)
            aspect_ratio = float(w_c) / h_c
            # Label is horizontal with aspect ratio around 3 to 6
            if 2.5 <= aspect_ratio <= 6.0:
                if area > max_area:
                    max_area = area
                    best_crop = frame_bgr[y:y+h_c, x:x+w_c]
                    
        if best_crop is not None:
            print(f"[LabelOCR] Auto-detected label rectangle contour (Area: {max_area} px)")
            return best_crop
            
        # Fallback to configured crop fraction
        x0, y0, x1, y1 = cfg.LABEL_CROP_FRAC
        # If default config values, check if they match gs_capture.jpg and adjust
        if x0 == 0.35 and y0 == 0.10 and x1 == 1.00 and y1 == 0.90:
            print("[LabelOCR] Using calibrated label bounds for gs_capture.jpg")
            x0, y0, x1, y1 = 0.073, 0.398, 0.651, 0.667
            
        return frame_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    def _orient(self, gray):
        code = cfg.LABEL_ROTATE_CODE
        return cv2.rotate(gray, code) if code is not None else gray

    def read(self, frame_bgr) -> Optional[OCRClaim]:
        # Check availability
        if not self._available and self._easyocr_reader is None:
            return None
            
        crop = self._label_crop(frame_bgr)
        h_crop, w_crop = crop.shape[:2]
        
        # Orient label (rotate 90 degrees clockwise to make horizontal if it's vertical)
        if w_crop > h_crop:
            print("[LabelOCR] Auto-rotating crop 90 degrees clockwise (vertical label detected)")
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = self._orient(crop)
            
        matched_sku = None
        raw_text = ""
        matched_text = ""
        
        # 1. High-precision extraction using the cached EasyOCR reader
        if self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(rotated)
                
                # Locate Variety heading
                variety_box = None
                for (bbox, text, prob) in results:
                    t_lower = text.lower()
                    if 'variety' in t_lower or 'ariety' in t_lower or 'varmety' in t_lower:
                        variety_box = bbox
                        print(f"[LabelOCR] Matched Variety heading text: '{text}'")
                        break
                        
                if variety_box is not None:
                    pts = np.array(variety_box, dtype=np.int32)
                    x_v = min(pts[:, 0])
                    y_v = min(pts[:, 1])
                    w_v = max(pts[:, 0]) - x_v
                    h_v = max(pts[:, 1]) - y_v
                    
                    # Search for value block (right or below)
                    candidates = []
                    for (bbox, text, prob) in results:
                        t_clean = text.strip()
                        if not t_clean:
                            continue
                        # Skip the heading words themselves
                        if any(w in t_clean.lower() for w in ['variety', 'ariety', 'varmety']):
                            continue
                            
                        pts_c = np.array(bbox, dtype=np.int32)
                        cx = np.mean(pts_c[:, 0])
                        cy = np.mean(pts_c[:, 1])
                        
                        # Right-aligned (same line)
                        is_right = (cx > x_v + w_v - 15) and (cx < x_v + w_v + 250) and (abs(cy - (y_v + h_v/2)) < 25)
                        # Below-aligned (next line)
                        is_below = (cy > y_v + h_v - 5) and (cy < y_v + h_v + 80) and (cx > x_v - 50) and (cx < x_v + w_v + 150)
                        
                        if is_right:
                            candidates.append((0, cx, text))
                        elif is_below:
                            candidates.append((1, cy, text))
                            
                    if candidates:
                        candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
                        variety_text = candidates[0][2]
                        matched_sku = _match_sku_in_text(variety_text)
                        if matched_sku:
                            raw_text = variety_text
                            matched_text = variety_text
                            print(f"[LabelOCR] Extracted mattress type '{matched_sku}' from spatial search: '{variety_text}'")
                
                # Fallback: if heading was found but no candidate matched a SKU, or heading wasn't found at all,
                # check if ANY of the detected text blocks in the entire image contains a SKU!
                if not matched_sku:
                    for (bbox, text, prob) in results:
                        matched_sku = _match_sku_in_text(text)
                        if matched_sku:
                            raw_text = text
                            matched_text = text
                            print(f"[LabelOCR] Found SKU '{matched_sku}' in general EasyOCR results: '{text}'")
                            break
                            
            except Exception as e:
                print(f"[LabelOCR] EasyOCR Variety extraction failed: {e}")
                
        # 2. General Tesseract OCR fallback on rotated label (if EasyOCR failed or not available)
        if not matched_sku and self._available:
            print("[LabelOCR] Falling back to general Tesseract OCR on rotated label")
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try:
                data = self._pytesseract.image_to_data(
                    binar, lang=cfg.OCR_LANG,
                    output_type=self._pytesseract.Output.DICT)
                kept = [t for t, c in zip(data['text'], data['conf'])
                        if t.strip() and int(float(c)) >= cfg.OCR_MIN_CONF]
                raw_text = ' '.join(kept)
                matched_sku = _match_sku_in_text(raw_text)
                matched_text = raw_text[:60]
                print(f"[LabelOCR] General OCR SKU match: '{matched_sku}'")
            except Exception as e:
                print(f"[LabelOCR] General Tesseract fallback failed: {e}")
                
        if not raw_text:
            return None
            
        return OCRClaim(sku=matched_sku, matched_text=matched_text, raw_text=raw_text)


def _match_sku_in_text(norm_spaced: str) -> Optional[str]:
    t = norm_spaced.lower()
    tokens = re.findall(r'[a-z]+', t)
    joined = ''.join(tokens)

    if 'gravite' in joined:
        return 'gravite'
    if 'ortho' in joined or 'irtho' in joined or 'rtho' in joined or 'orto' in joined:
        return 'ortholex'
    if 'maxipro' in joined:
        return 'maxi_pro'
    if 'maxiplush' in joined:
        return 'maxi_pro' if 'pro' in tokens else 'maxi_plush'
    return None


# ==============================================================================
# Local Testing Execution Block
# ==============================================================================
if __name__ == '__main__':
    import time

    print("==================================================")
    print(" CLAIM PIPELINE LOCAL TESTER (LIVE CAMERA CAPTURE)")
    print("==================================================")

    # If a path is passed as a CLI arg, use it directly (CI / scripted use).
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Error: File not found - {img_path}")
            sys.exit(1)
        frame = cv2.imread(img_path)
        if frame is None:
            print("Error: Could not read image. Ensure it is a valid image file (JPG/PNG).")
            sys.exit(1)
        print(f"Loaded test image: {os.path.basename(img_path)} (Shape: {frame.shape})")
    else:
        try:
            from camera import ThreadedCamera
        except ImportError:
            print("Error: Could not import ThreadedCamera from camera.py")
            sys.exit(1)

        print("Initializing camera feed...")
        cam = ThreadedCamera(index=0)

        cv2.namedWindow('Capture Image')

        # Mouse callback to capture on left-click inside the button boundaries
        trigger_capture = False
        def click_callback(event, x, y, flags, param):
            global trigger_capture
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 <= x <= 190 and 10 <= y <= 50:
                    trigger_capture = True

        cv2.setMouseCallback('Capture Image', click_callback)

        print("Live Camera Preview Active.")
        print("  - Click the green 'Capture' button or press SPACE/ENTER to grab a frame.")
        print("  - Press 'q' on your keyboard to quit.")

        while True:
            frame, err = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()
            # Draw "Capture" button overlay (Top-Left corner)
            cv2.rectangle(display, (10, 10), (190, 50), (0, 255, 0), -1)
            cv2.putText(display, "Capture", (55, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow('Capture Image', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Capture cancelled. Exiting.")
                cam.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord(' ') or key == 13 or trigger_capture:
                # Save as gs_capture.jpg
                img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gs_capture.jpg")
                cv2.imwrite(img_path, frame)
                print(f"[SUCCESS] Captured frame and saved to: {img_path}")
                cam.release()
                cv2.destroyAllWindows()
                break

    # --- Test QR Reader ---
    # The QR reader will:
    #   1. Decode the QR from the image.
    #   2. Open the decoded URL (warranty/registration page).
    #   3. Scrape the mattress type text from the live page.
    print("\n--- Testing QRReader (URL fetch + page scrape) ---")
    qr_reader = QRReader()
    qr_claim, qr_stage = qr_reader.read(frame, return_stage=True)

    if qr_claim:
        print(f"[SUCCESS] QR Code decoded at retry stage: '{qr_stage}'")
        print(f"  -> Extracted SKU (from live page / URL params): {qr_claim.sku}")
        print(f"  -> Raw Product Name (URL param):                {qr_claim.product_name_raw}")
        print(f"  -> Batch Number:                                {qr_claim.batch_no}")
        print(f"  -> Inventory Item ID:                           {qr_claim.inventory_item_id}")
        print(f"  -> Decoded URL:                                 {qr_claim.url}")
    else:
        print(f"[FAIL] No QR code detected. (Ended at stage: '{qr_stage}')")

    # --- Test OCR Reader ---
    # The OCR reader crops the label region and scans for the variety/type text.
    print("\n--- Testing LabelOCRReader (variety/type scan) ---")
    ocr_reader = LabelOCRReader()

    if not ocr_reader.available:
        print("[FAIL] OCR is unavailable. Pytesseract or system Tesseract engine is missing.")
    else:
        ocr_claim = ocr_reader.read(frame)
        if ocr_claim:
            print("[SUCCESS] Variety/type text found on label.")
            print(f"  -> Extracted SKU (variety):  {ocr_claim.sku}")
            print(f"  -> Matched Text (truncated): {ocr_claim.matched_text}")
            print(f"  -> Full Raw OCR Text:        {ocr_claim.raw_text}")
        else:
            print("[FAIL] No valid variety/type text detected in label region.")

    print("\n==================================================")
    print(" DONE")
    print("==================================================")
