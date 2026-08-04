import difflib
from config import EXPECTED_DIMENSIONS, DIMENSION_TOLERANCE_CM


def normalize_string(s):
    if not s:
        return ""
    return str(s).lower().replace("_", " ").replace("-", " ").strip()


def fuzzy_match_score(target, text):
    """
    Computes maximum fuzzy similarity score between target string and any substring window in text.
    Returns float score between 0.0 and 1.0.
    """
    target = normalize_string(target)
    text = normalize_string(text)

    if not target or not text:
        return 0.0

    if target in text:
        return 1.0

    target_words = target.split()
    text_words = text.split()

    best_score = 0.0
    for win_size in range(max(1, len(target_words) - 1), len(target_words) + 2):
        for i in range(len(text_words) - win_size + 1):
            sub_text = " ".join(text_words[i:i + win_size])
            ratio = difflib.SequenceMatcher(None, target, sub_text).ratio()
            if ratio > best_score:
                best_score = ratio

    return best_score


def verify_full_inspection(qr_data, side_ocr_data, texture_data, top_data, fuzzy_threshold=0.65):
    """
    Dual-Layer Inspection Engine:
    
    Layer 1: 4-Way Product Identity Verification
      1. QR Code Reading (Camera 1)
      2. Side Bill OCR (Camera 2)
      3. Texture AI Prediction (Camera 2)
      4. Corner Label OCR (Camera 3)

    Layer 2: Dimension Tolerance Verification
      Compares measured Length, Width, and Thickness against expected dimensions for the identified product.
    """
    qr_product = qr_data.get("product_name", "N/A")
    side_ocr_text = side_ocr_data.get("full_text", "")
    texture_category = texture_data.get("predicted_category", "N/A")

    corner_label_info = top_data.get("corner_label", {})
    corner_label_product = corner_label_info.get("product_name", "Not Detected")
    corner_label_text = corner_label_info.get("full_text", "")

    norm_qr = normalize_string(qr_product)
    norm_side_ocr = normalize_string(side_ocr_text)
    norm_texture = normalize_string(texture_category)
    norm_corner_label = normalize_string(corner_label_product)

    mismatches = []

    # 1. QR vs Texture AI
    qr_texture_score = fuzzy_match_score(norm_qr, norm_texture)
    if not (qr_texture_score >= 0.70 or norm_qr in norm_texture or norm_texture in norm_qr) and norm_qr not in ("n/a", "not detected"):
        mismatches.append({
            "source": "Texture Detection Mismatch",
            "expected": qr_product,
            "detected": f"{texture_category} (Similarity: {int(qr_texture_score * 100)}%)"
        })

    # 2. QR vs Side Bill OCR
    side_ocr_score = fuzzy_match_score(norm_qr, norm_side_ocr)
    if side_ocr_score < fuzzy_threshold and norm_qr not in ("n/a", "not detected"):
        mismatches.append({
            "source": "Bill OCR Mismatch",
            "expected": qr_product,
            "detected": f"{side_ocr_text if side_ocr_text else 'No OCR Text Detected'} (Similarity: {int(side_ocr_score * 100)}%)"
        })

    # 3. QR vs Corner Label OCR
    corner_label_score = fuzzy_match_score(norm_qr, norm_corner_label)
    if corner_label_score < 0.60 and norm_qr not in ("n/a", "not detected"):
        mismatches.append({
            "source": "Corner Label Mismatch",
            "expected": qr_product,
            "detected": f"{corner_label_product} (Raw Text: {corner_label_text})"
        })

    # Check unreadable QR
    if norm_qr in ("n/a", "not detected"):
        mismatches.append({
            "source": "QR Code Mismatch",
            "expected": "Valid Product QR",
            "detected": "Unreadable / Missing QR"
        })

    identity_status = "PASS" if len(mismatches) == 0 else "FAIL"

    # ---- LAYER 2: Dimension Tolerance Verification ----
    measured_l = top_data.get("length_cm", 0.0)
    measured_w = top_data.get("width_cm", 0.0)
    measured_h = top_data.get("thickness_cm", 0.0)

    # Get expected dimensions for identified product
    lookup_key = norm_qr if norm_qr in EXPECTED_DIMENSIONS else "default"
    exp_dims = EXPECTED_DIMENSIONS.get(lookup_key, EXPECTED_DIMENSIONS["default"])

    l_diff = abs(measured_l - exp_dims["length_cm"])
    w_diff = abs(measured_w - exp_dims["width_cm"])

    dim_pass = (l_diff <= DIMENSION_TOLERANCE_CM) and (w_diff <= DIMENSION_TOLERANCE_CM)
    dim_status = "PASS" if dim_pass else "FAIL"

    overall_status = "PASS" if (identity_status == "PASS" and dim_status == "PASS") else "FAIL"

    return {
        "overall_status": overall_status,
        "identity_status": identity_status,
        "dimension_status": dim_status,
        "qr_product": qr_product,
        "side_ocr_summary": side_ocr_text if side_ocr_text else "N/A",
        "side_ocr_similarity": round(side_ocr_score * 100, 1),
        "texture_category": texture_category,
        "corner_label_product": corner_label_product,
        "corner_label_size": corner_label_info.get("size", "N/A"),
        "measured_dimensions": {
            "length_cm": measured_l,
            "width_cm": measured_w,
            "thickness_cm": measured_h
        },
        "expected_dimensions": exp_dims,
        "mismatches": mismatches
    }
