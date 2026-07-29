"""
mattress/adapters.py — glue between subsystem outputs (texture pipeline, QR
decoder, banner OCR) and reconcile.py's plain-string inputs.

Each adapter takes a subsystem's raw/native output and returns either a
string (best-effort raw value — reconcile.normalize_sku() handles parsing)
or None (source absent / failed / rejected).
"""

try:
    from mattress.pipeline import PipelineStatus
    from mattress.identify import NEEDS_REVIEW
except ImportError:
    class PipelineStatus:
        SUCCESS = "SUCCESS"
    NEEDS_REVIEW = "NEEDS_REVIEW"


def texture_to_reconcile_input(status, result):
    """Convert IdentificationPipeline.process_frame() output to reconcile()'s
    texture arg. Verified on real Pi captures: SUCCESS -> plain canonical
    SKU string (e.g. 'gravite'); every NEEDS_REVIEW_* status -> NEEDS_REVIEW
    sentinel, mapped here to None."""
    if status == PipelineStatus.SUCCESS and result != NEEDS_REVIEW:
        return result
    return None


def qr_to_reconcile_input(qr_claim):
    """Convert a QRReader.read() result to reconcile()'s qr_payload arg.
    Deliberately ignores QRClaim.sku (computed by claim.py's own normalizer)
    and instead passes the raw productName value through, so normalization
    happens in exactly one place: reconcile.normalize_sku(). This avoids the
    two normalizers (claim.py's _match_sku_in_text vs reconcile.py's
    normalize_sku) silently diverging on an edge case."""
    if qr_claim is None:
        return None
    return qr_claim.product_name_raw or None
