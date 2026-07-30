"""
mattress/claim.py — read what the box/label/QR CLAIM the SKU is.

Hard rule: this module produces CLAIMS, never identity. Nothing here overrides
the fabric verdict; reconcile.py consumes both and fabric always wins.

Two independent readers:
  QRReader        — decodes the QR (warranty-registration URL), extracts the
                     claimed SKU + batch + inventory id from its query string.
  LabelOCRReader  — reads printed text on the label as a second, independent
                     claim (covers a torn/occluded QR). Tesseract-based
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

import cv2
from pyzbar.pyzbar import decode as zbar_decode

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


def _decode_with_retries(gray_or_bgr):
    """Classical retry ladder: direct -> grayscale -> upscale -> CLAHE.
    Stops at the first stage that finds ANY code. Returns (codes, stage_name)."""
    codes = zbar_decode(gray_or_bgr)
    if codes:
        return codes, 'direct'

    # Always ensure we have a grayscale image for subsequent image processing steps
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
        enhanced = clahe.apply(gray) # Safe: guaranteed to be grayscale now
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
        if symbology != 'QRCODE' or '?' not in payload:
            return QRClaim(sku=None, product_name_raw='', batch_no=None,
                           inventory_item_id=None, url=payload, symbology=symbology)

        q = parse_qs(urlparse(payload).query)
        product_raw = q.get('productName', [''])[0]
        return QRClaim(
            sku=_normalise_product_name(product_raw),
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
        gray = self._orient(gray)
        _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            data = self._pytesseract.image_to_data(
                binar, lang=cfg.OCR_LANG,
                output_type=self._pytesseract.Output.DICT)
        except Exception as e:
            print(f"[OCR Warning] Tesseract engine failed to execute: {e}")
            return None 

        kept = [t for t, c in zip(data['text'], data['conf'])
                if t.strip() and int(float(c)) >= cfg.OCR_MIN_CONF]
        raw_text = ' '.join(kept)
        if not raw_text:
            return None

        matched_sku = _match_sku_in_text(raw_text)

        return OCRClaim(sku=matched_sku, matched_text=raw_text[:60], raw_text=raw_text)


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
    print("==================================================")
    print(" CLAIM PIPELINE LOCAL TESTER")
    print("==================================================")
    
    if len(sys.argv) < 2:
        print("Usage: python claim.py <path_to_test_image.jpg>")
        print("Please provide an image file containing a QR code or Mattress Label.")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: File not found - {img_path}")
        sys.exit(1)

    frame = cv2.imread(img_path)
    if frame is None:
        print("Error: Could not read image. Ensure it is a valid image file.")
        sys.exit(1)

    print(f"Loaded test image: {os.path.basename(img_path)} (Shape: {frame.shape})")

    # --- Test QR Reader ---
    print("\n--- Testing QRReader ---")
    qr_reader = QRReader()
    qr_claim, qr_stage = qr_reader.read(frame, return_stage=True)
    
    if qr_claim:
        print(f"[SUCCESS] QR Code decoded at retry stage: '{qr_stage}'")
        print(f"  -> Extracted SKU:  {qr_claim.sku}")
        print(f"  -> Raw Product:    {qr_claim.product_name_raw}")
        print(f"  -> Batch Number:   {qr_claim.batch_no}")
        print(f"  -> Decoded URL:    {qr_claim.url}")
    else:
        print(f"[FAIL] No QR code detected. (Ended at stage: '{qr_stage}')")

    # --- Test OCR Reader ---
    print("\n--- Testing LabelOCRReader ---")
    ocr_reader = LabelOCRReader()
    
    if not ocr_reader.available:
        print("[FAIL] OCR is unavailable. Pytesseract or system Tesseract engine is missing.")
    else:
        ocr_claim = ocr_reader.read(frame)
        if ocr_claim:
            print("[SUCCESS] Text found on label.")
            print(f"  -> Extracted SKU: {ocr_claim.sku}")
            print(f"  -> Raw Text:      {ocr_claim.raw_text}")
        else:
            print("[FAIL] No valid text or OCR claim detected in image.")