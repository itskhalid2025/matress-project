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
from pyzbar.pyzbar import decode as zbar_decode

# Ensure relative imports work if run as a script
try:
    import config as cfg
except ImportError:
    # Fallback if config is missing during local test script execution
    # Updated to match the new webcam/1080p configurations
    class DummyConfig:
        QR_RETRY_GRAYSCALE = True
        QR_RETRY_UPSCALE = 2.0
        QR_RETRY_CLAHE = True
        QR_RETRY_OTSU = False                     # Disabled for auto-exposing webcams
        LABEL_CROP_FRAC = (0.35, 0.10, 1.00, 0.90)
        LABEL_ROTATE_CODE = None                  # Upright webcam orientation
        OCR_LANG = 'eng'
        OCR_MIN_CONF = 40
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
        print(f"[QR/URL] Page text snippet: {page_text[:200]}")
    return matched


def _decode_with_retries(gray_or_bgr):
    """Classical retry ladder: direct -> grayscale -> upscale -> CLAHE."""
    codes = zbar_decode(gray_or_bgr)
    if codes:
        return codes, 'direct'

    if len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr

    if cfg.QR_RETRY_GRAYSCALE:
        codes = zbar_decode(gray)
        if codes:
            return codes, 'grayscale'

    if cfg.QR_RETRY_UPSCALE and cfg.QR_RETRY_UPSCALE != 1.0:
        up = cv2.resize(gray, None, fx=cfg.QR_RETRY_UPSCALE, fy=cfg.QR_RETRY_UPSCALE,
                        interpolation=cv2.INTER_CUBIC)
        codes = zbar_decode(up)
        if codes:
            return codes, f'upscale_{cfg.QR_RETRY_UPSCALE}x'

    if cfg.QR_RETRY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        codes = zbar_decode(enhanced)
        if codes:
            return codes, 'clahe'
        
    if cfg.QR_RETRY_OTSU:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        codes = zbar_decode(otsu)
        if codes:
            return codes, 'otsu'

    return [], 'none'


class QRReader:
    """Decodes the QR / any 1D barcode on a side-view frame."""

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
        if symbology != 'QRCODE' or '://' not in payload:
            return QRClaim(sku=None, product_name_raw='', batch_no=None,
                           inventory_item_id=None, url=payload, symbology=symbology)

        q = parse_qs(urlparse(payload).query)
        product_raw = q.get('productName', [''])[0]
        sku_from_url_params = _normalise_product_name(product_raw) if product_raw else None

        print(f"[QR] Decoding URL: {payload}")
        sku_from_page = _fetch_mattress_type_from_url(payload)
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
        try:
            import pytesseract
            # Windows Tesseract Executable Auto-Detection
            tess_win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tess_win_path):
                pytesseract.pytesseract.tesseract_cmd = tess_win_path
            self._pytesseract = pytesseract
            self._available = True
        except ImportError:
            self._pytesseract = None
            self._available = False

    @property
    def available(self):
        return self._available

    def _label_crop(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = cfg.LABEL_CROP_FRAC
        return frame_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    def _orient(self, gray):
        code = cfg.LABEL_ROTATE_CODE
        return cv2.rotate(gray, code) if code is not None else gray

    def read(self, frame_bgr) -> Optional[OCRClaim]:
        if not self._available:
            return None

        crop = self._label_crop(frame_bgr)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Try all 4 orientations — label may be rotated 0°, 90°, 180°, 270°
        rotations = [
            (None, gray),
            (cv2.ROTATE_90_CLOCKWISE, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
            (cv2.ROTATE_180, cv2.rotate(gray, cv2.ROTATE_180)),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        # Override: if config sets a specific rotate code, try that first
        if cfg.LABEL_ROTATE_CODE is not None:
            fixed = cv2.rotate(gray, cfg.LABEL_ROTATE_CODE)
            rotations = [(cfg.LABEL_ROTATE_CODE, fixed)] + rotations

        best_result = None

        for rot_code, rot_img in rotations:
            # Upscale for better OCR accuracy
            scale = 2.0
            upscaled = cv2.resize(rot_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Binarise — try both Otsu and adaptive
            _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 10)

            for binar in [otsu, adaptive]:
                try:
                    data = self._pytesseract.image_to_data(
                        binar, lang=cfg.OCR_LANG,
                        output_type=self._pytesseract.Output.DICT,
                        config='--psm 6')
                except Exception as e:
                    print(f"[OCR Warning] {e}")
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
                            result = OCRClaim(
                                sku=matched_sku,
                                matched_text=f"VARIETY: {same_line_text}",
                                raw_text=same_line_text
                            )
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
                        variety_line = [t for top, t, c in below_words if abs(top - min_top) < 40]
                        variety_text = ' '.join(variety_line).strip()
                        matched_sku = _match_sku_in_text(variety_text)
                        result = OCRClaim(
                            sku=matched_sku,
                            matched_text=f"VARIETY: {variety_text}",
                            raw_text=variety_text
                        )
                        if matched_sku:
                            return result
                        if best_result is None:
                            best_result = result
                    continue

                # Fallback: no VARIETY found — try full text SKU match
                kept = [t for t, c in zip(texts, confs)
                        if t.strip() and int(float(c)) >= cfg.OCR_MIN_CONF]
                raw_text = ' '.join(kept)
                matched_sku = _match_sku_in_text(raw_text)
                if matched_sku and best_result is None:
                    best_result = OCRClaim(
                        sku=matched_sku,
                        matched_text=raw_text[:60],
                        raw_text=raw_text
                    )

        return best_result


def _match_sku_in_text(norm_spaced: str) -> Optional[str]:
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


# ==============================================================================
# Local Testing Execution Block
# ==============================================================================
if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog

    print("==================================================")
    print(" CLAIM PIPELINE LOCAL TESTER")
    print("==================================================")

    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        img_path = filedialog.askopenfilename(
            title="Select a mattress image",
            filetypes=[
                ("JPEG images", "*.jpg *.jpeg"),
                ("PNG images",  "*.png"),
                ("All image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ]
        )
        root.destroy()
        if not img_path:
            print("No file selected. Exiting.")
            sys.exit(0)

    if not os.path.exists(img_path):
        print(f"Error: File not found - {img_path}")
        sys.exit(1)

    frame = cv2.imread(img_path)
    if frame is None:
        print("Error: Could not read image. Ensure it is a valid image file (JPG/PNG).")
        sys.exit(1)

    print(f"Loaded test image: {os.path.basename(img_path)} (Shape: {frame.shape})")

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