"""
mattress/claim.py — read what the box/label/QR CLAIM the SKU is.

Hard rule: this module produces CLAIMS, never identity. Nothing here overrides
the fabric verdict; reconcile.py consumes both and fabric always wins.

Two independent readers:
  QRReader        — decodes the QR (warranty-registration URL), extracts the
                     claimed SKU + batch + inventory id from its query string.
  LabelOCRReader   — reads printed text on the label as a second, independent
                     claim (covers a torn/occluded QR). Tesseract-based
                     (classical/offline) — flagged for professor sign-off since
                     most modern OCR is deep-learning.

Both fail loud to None on failure; they never guess.
"""
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs

import cv2
from pyzbar.pyzbar import decode as zbar_decode

try:
    import config as cfg
except ImportError:
    import mattress.config as cfg


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
    """'Gravite_Mattress' -> 'gravite'; 'Maxiplush_Pro_Mattress' -> 'maxi_pro'.
    Underscores/spaces are preserved as token separators so the maxi pair can
    be split on the standalone 'pro' token."""
    spaced = raw.replace('_', ' ').replace('-', ' ')
    return _match_sku_in_text(spaced)


def _decode_with_retries(gray_or_bgr):
    """Classical retry ladder: direct -> grayscale -> upscale -> CLAHE.
    Stops at the first stage that finds ANY code. Returns (codes, stage_name)."""
    codes = zbar_decode(gray_or_bgr)
    if codes:
        return codes, 'direct'

    if cfg.QR_RETRY_GRAYSCALE and len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
        codes = zbar_decode(gray)
        if codes:
            return codes, 'grayscale'
    else:
        gray = gray_or_bgr

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
        if symbology != 'QRCODE' or '?' not in payload:
            # non-URL payload (e.g. CODE39 batch code) — no structured SKU claim
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
    """Reads printed product-name text on the label as a second claim.
    Requires pytesseract + the tesseract binary. Fails loud to None (not an
    exception) if either is unavailable, so this stays an optional hardening
    layer rather than a hard dependency of the claim pipeline."""

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
        """Rotate the crop into reading orientation.
        EMPIRICALLY DETERMINED (2026-07, gravite sample), not auto-detected:
        Tesseract's OSD failed on this label (side-view text is small/low-res
        for OSD's script-detection to work reliably), but a brute-force sweep
        of all 4 rotations x flip showed rot=90 CLOCKWISE, no flip, reads
        cleanly ('PRODUCT CODE: 75X36X6 | NET CONTENT 1N | MONTH AND YEAR...').
        This matches the label being mounted sideways on the mattress edge as
        the side camera sees it. PLACEHOLDER: re-verify if the side camera
        mounting/orientation changes (Task C1 re-collection).
        """
        code = cfg.LABEL_ROTATE_CODE
        return cv2.rotate(gray, code) if code is not None else gray

    def read(self, frame_bgr) -> Optional[OCRClaim]:
        if not self._available:
            return None
        crop = self._label_crop(frame_bgr)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._orient(gray)
        # simple binarisation; label text is high-contrast black-on-white
        _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        data = self._pytesseract.image_to_data(
            binar, lang=cfg.OCR_LANG,
            output_type=self._pytesseract.Output.DICT)

        kept = [t for t, c in zip(data['text'], data['conf'])
                if t.strip() and int(float(c)) >= cfg.OCR_MIN_CONF]
        raw_text = ' '.join(kept)
        if not raw_text:
            return None

        matched_sku = _match_sku_in_text(raw_text)

        return OCRClaim(sku=matched_sku, matched_text=raw_text[:60], raw_text=raw_text)


def _match_sku_in_text(norm_spaced: str) -> Optional[str]:
    """Map free text (QR productName OR OCR label text) to a SKU, failing SAFE
    on ambiguity.

    Ground truth (2026-07), confirmed from DECODED QRs (not label guesses):
      gravite    -> QR 'Gravite_Mattress'
      ortholex   -> QR 'Ortho_Lex_Mattress'
      maxi_plush -> QR 'Maxi_Plush_Mattress_-_South'
      maxi_pro   -> QR 'Maxipro_Mattress_-_South'   (ONE word 'Maxipro', no 'plush')

    Note the QR wording for maxi_pro ('Maxipro') differs from the PRINTED
    label's commodity name ('MAXIPLUSH PRO MATTRESS') — the two are genuinely
    different strings for the same product. So:
      * 'maxipro' as a plain substring is UNAMBIGUOUS (maxiplush contains
        'l','u','s','h' that maxipro doesn't — no substring collision) and
        handles the QR case directly.
      * 'maxiplush' as a substring is ambiguous ONLY against the printed
        label's 'MAXIPLUSH PRO' commodity name — resolved by checking for a
        standalone 'pro' TOKEN (never matches inside 'PRODUCT', which is one
        token, not 'pro'+'duct').
    Abstain (None) when nothing matches — OCR/QR-unmapped is a secondary
    claim; fabric is authoritative, so abstaining is correct-conservative.
    """
    t = norm_spaced.lower()
    tokens = re.findall(r'[a-z]+', t)
    joined = ''.join(tokens)

    if 'gravite' in joined:
        return 'gravite'
    if 'ortholex' in joined or 'ortho' in tokens:
        return 'ortholex'
    if 'maxipro' in joined:                  # QR wording: unambiguous
        return 'maxi_pro'
    if 'maxiplush' in joined:                # printed-label wording: check 'pro'
        return 'maxi_pro' if 'pro' in tokens else 'maxi_plush'
    return None                              # nothing distinctive -> abstain
