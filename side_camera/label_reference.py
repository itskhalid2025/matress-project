"""
mattress/label_reference.py — SKU -> full label text lookup.

The live QR camera is positioned close enough to DECODE the QR, which means the
full printed label is only partially in frame. We don't OCR that partial view;
instead the QR decode gives us the SKU (and batch), and we look up the full,
verified label text here (transcribed once from clean phone photos) to DISPLAY.

This is REFERENCE data, not a live read — display it as such.
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = os.path.join(_HERE, "label_reference.json")

# label fields shown, in print order
_FIELD_ORDER = [
    ("commodity_name", "Name of the Commodity"),
    ("variety", "Variety"),
    ("dimension", "Dimension (L x W x T)"),
    ("product_code", "Product Code"),
    ("net_content", "Net Content"),
    ("manufacturing", "Month & Year of Mfg"),
    ("mrp", "MRP"),
    ("batch_no", "Batch No"),
    ("inventory_item_id", "Inventory Item ID"),
    ("trace_code", "Trace Code"),
]


def _load():
    with open(_JSON, "r") as f:
        return json.load(f)


_DATA = _load()


def get_label(sku):
    """Return the label dict for a SKU, or None if unknown."""
    return _DATA.get(sku)


def has_label(sku):
    return sku in _DATA and not sku.startswith("_")


def format_label(sku, batch_from_qr=None):
    """Human-readable multi-line label block for display. If the live QR
    carried a batch number, cross-check it against the stored one and flag a
    difference (different batch of same SKU => stored MRP/date may be stale)."""
    rec = _DATA.get(sku)
    if rec is None:
        return f"  (no reference label on file for '{sku}')"

    lines = [f"  ---- Label: {rec.get('commodity_name', sku)} ----"]
    for key, disp in _FIELD_ORDER:
        val = rec.get(key)
        if val:
            lines.append(f"  {disp:22s}: {val}")

    if batch_from_qr and rec.get("batch_no") and batch_from_qr != rec["batch_no"]:
        lines.append(f"  [!] live QR batch {batch_from_qr} != reference batch "
                     f"{rec['batch_no']} — displayed label data may be from a "
                     f"different batch of this SKU.")

    man = _DATA.get("_manufacturer", {})
    if man:
        lines.append(f"  Mfd/Mktd by           : {man.get('marketed_by', '')}")
        lines.append(f"  Contact               : {man.get('phone','')} | "
                     f"{man.get('email','')} | {man.get('website','')}")
    return "\n".join(lines)
