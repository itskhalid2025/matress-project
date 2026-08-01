"""
tile_pipeline.py — Sliding Window Tile Inspection Pipeline for sidecam_new.

Pipeline:
  1. Generates 640x640 overlapping tiles (25-30% overlap) across full-resolution image.
  2. Saves debug tiles to side_camera/sidecam_new/debug_tiles/ (tile_01.jpg, tile_02.jpg, ...).
  3. Scans each tile for QR code using qrtest.py.
     - If QR code is found: extracts Product Name, Batch Number, Inventory Item ID,
       draws a GREEN BORDER around that tile before saving, and stops QR search for remaining tiles.
  4. Runs Tesseract OCR on every tile after image preprocessing (Grayscale -> CLAHE -> Adaptive Threshold -> Noise Filter).
  5. Merges OCR text lines from all tiles, deduplicating identical lines while preserving reading order.
  6. Prints detailed per-tile terminal logs and final combined OCR block.
"""

import os
import sys
import shutil
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tiler import generate_tiles
from qrtest import inspect_qr_code
from ocr_reader import read_label_ocr
from label_detector import detect_and_crop_label, draw_label_bounding_box

DEBUG_TILES_DIR = os.path.join(BASE_DIR, "debug_tiles")


def run_tile_pipeline(full_resolution_image):
    """
    Executes tile-based sliding window inspection pipeline on full-resolution image.
    Uses YOLO11n to isolate the label first, falling back to full-frame sliding window if not detected.

    Returns dict:
      {
        "qr": {
          "product_name": str,
          "batch_no": str,
          "inventory_item_id": str,
          "found": bool,
          "tile_index": int or None
        },
        "ocr": {
          "text": str
        }
      }
    """
    # 1. Prepare / Clean debug_tiles directory
    os.makedirs(DEBUG_TILES_DIR, exist_ok=True)
    _clean_debug_tiles_dir()

    # Try detecting the label via YOLO/classic hybrid
    label_crop, box_pts, label_found, detected_qr_data = detect_and_crop_label(full_resolution_image)

    if label_found:
        print("\n[YOLO] Label detected! Running inspection on label crop.")
        draw_label_bounding_box(full_resolution_image, box_pts, "MATTRESS LABEL")
        # Generate tiles of the cropped label instead of the whole image
        tiles = generate_tiles(label_crop, tile_size=640, overlap_pct=0.25)
    else:
        print("\n[YOLO] Label NOT detected. Running full-frame sliding window fallback.")
        # 2. Generate 640x640 overlapping tiles with 25-30% overlap
        tiles = generate_tiles(full_resolution_image, tile_size=640, overlap_pct=0.25)
    
    if detected_qr_data is not None:
        print("[YOLO/Orientation] Using pre-detected QR code data.")
        qr_data = {
            "product_name": detected_qr_data.get("product_name", "Not Detected"),
            "batch_no": detected_qr_data.get("batch_no", "Not Detected"),
            "inventory_item_id": detected_qr_data.get("inventory_item_id", "Not Detected"),
            "found": True,
            "tile_index": 0
        }
        qr_search_stopped = True
    else:
        qr_data = {
            "product_name": "Not Detected",
            "batch_no": "Not Detected",
            "inventory_item_id": "Not Detected",
            "found": False,
            "tile_index": None
        }
        qr_search_stopped = False

    all_tile_ocr_lines = []

    print("\n" + "=" * 45)
    print("SLIDING WINDOW TILE INSPECTION STARTED")
    print(f"Total Generated Tiles: {len(tiles)}")
    print("=" * 45 + "\n")

    for tile_idx, tile_img, bbox in tiles:
        tile_filename = f"tile_{tile_idx:02d}.jpg"
        tile_save_path = os.path.join(DEBUG_TILES_DIR, tile_filename)

        # 3. QR Detection (stop further searching once QR is found in any tile)
        tile_qr_found = False
        tile_qr_info = None

        if not qr_search_stopped:
            qr_res = inspect_qr_code(tile_img)
            if qr_res and qr_res.get("qr_found"):
                qr_search_stopped = True
                tile_qr_found = True
                tile_qr_info = qr_res
                qr_data["product_name"] = qr_res["product_name"]
                qr_data["batch_no"] = qr_res["batch_no"]
                qr_data["inventory_item_id"] = qr_res["inventory_item_id"]
                qr_data["found"] = True
                qr_data["tile_index"] = tile_idx

        # Draw green border around tile image if QR was detected in this tile
        tile_save_img = tile_img.copy()
        if tile_qr_found:
            th, tw = tile_save_img.shape[:2]
            cv2.rectangle(tile_save_img, (0, 0), (tw - 1, th - 1), (0, 255, 0), 8)

        # Save tile image to debug_tiles/ folder
        cv2.imwrite(tile_save_path, tile_save_img)

        # 4. Tesseract OCR processing on tile
        raw_ocr_text = read_label_ocr(tile_img)
        tile_ocr_lines = _extract_valid_lines(raw_ocr_text)

        if tile_ocr_lines:
            all_tile_ocr_lines.extend(tile_ocr_lines)

        # 5. Print Terminal Output for this tile
        print("-----------------------------------------")
        print(f"Tile {tile_idx}")
        print()
        if tile_qr_found:
            print("QR : FOUND")
            print()
            print("Product Name:")
            print(tile_qr_info["product_name"])
            print()
            print("Batch Number:")
            print(tile_qr_info["batch_no"])
            print()
            print("Inventory Item ID:")
            print(tile_qr_info["inventory_item_id"])
        else:
            print("QR : Not Found")

        print()
        print(f"OCR : {len(tile_ocr_lines)} lines")
        print("-----------------------------------------\n")

    # 6. Merge & Deduplicate OCR Text across all tiles
    merged_ocr_text = _merge_and_deduplicate_lines(all_tile_ocr_lines)

    # 7. Print Final Combined OCR Block
    print("-----------------------------------------")
    print("Final OCR")
    print()
    print(merged_ocr_text if merged_ocr_text else "No OCR text extracted")
    print("-----------------------------------------\n")

    # 8. Run verification of the label contents
    verification_verdict, verification_detail = verify_label_contents(qr_data, merged_ocr_text)

    return {
        "qr": qr_data,
        "ocr": {
            "text": merged_ocr_text if merged_ocr_text else "No OCR text detected"
        },
        "verification": {
            "verdict": verification_verdict,
            "detail": verification_detail
        },
        "label_crop": label_crop if label_found else None
    }


def _extract_valid_lines(raw_text):
    """Parses raw OCR output into non-empty, clean text lines."""
    if not raw_text or "No OCR text" in raw_text or "Not Installed" in raw_text:
        return []

    lines = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        # Filter out line noise (less than 2 alphanumeric characters)
        alnum_chars = [c for c in cleaned if c.isalnum()]
        if len(alnum_chars) >= 2:
            lines.append(cleaned)
    return lines


def _merge_and_deduplicate_lines(lines):
    """
    Merges OCR lines from all tiles, removing duplicate/overlapping lines
    while preserving natural reading order.
    """
    if not lines:
        return ""

    seen = set()
    unique_lines = []

    for line in lines:
        # Standardize line string for similarity check
        norm = line.lower().strip()
        # Simple fuzzy normalization (removing spaces and punctuation for duplicate check)
        compact = "".join(c for c in norm if c.isalnum())
        
        if len(compact) < 2:
            continue

        if compact not in seen:
            seen.add(compact)
            unique_lines.append(line)

    return "\n".join(unique_lines)


def _clean_debug_tiles_dir():
    """Empties side_camera/sidecam_new/debug_tiles/ folder before running a new test."""
    if os.path.exists(DEBUG_TILES_DIR):
        for f in os.listdir(DEBUG_TILES_DIR):
            file_path = os.path.join(DEBUG_TILES_DIR, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception:
                pass


import json

def load_label_references():
    """Loads label_reference.json from the parent directory."""
    parent_dir = os.path.dirname(BASE_DIR)
    ref_path = os.path.join(parent_dir, "label_reference.json")
    if os.path.exists(ref_path):
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Verification Error] Failed to read {ref_path}: {e}")
    return {}


def verify_label_contents(qr_data, ocr_text):
    """
    Verifies that the extracted OCR text matches the reference fields for the SKU
    detected in the QR code.
    
    Returns a tuple: (verdict_str, details_str)
    """
    if not qr_data or not qr_data.get("found"):
        return "UNVERIFIED", "Verification skipped: QR code not detected."
        
    if not ocr_text or ocr_text.strip() == "" or "No OCR text" in ocr_text:
        return "UNVERIFIED", "Verification incomplete: Label OCR text not detected."

    refs = load_label_references()
    if not refs:
        return "UNVERIFIED", "Verification error: Reference database unavailable."

    prod_name_lower = qr_data.get("product_name", "").lower()
    
    sku_key = None
    for key in ["gravite", "ortholex", "maxi_pro", "maxi_plush"]:
        clean_key = key.replace("_", "")
        clean_prod = prod_name_lower.replace("_", "").replace(" ", "").replace("-", "")
        if clean_key in clean_prod:
            sku_key = key
            break
            
    if not sku_key:
        return "UNVERIFIED", f"Verification incomplete: Unrecognized QR product name '{qr_data.get('product_name')}'."

    sku_ref = refs.get(sku_key)
    if not sku_ref:
        return "UNVERIFIED", f"Verification incomplete: Reference not found for SKU '{sku_key}'."

    ocr_lower = ocr_text.lower()
    
    # Extract variety name and MRP digits for verification
    expected_variety = sku_ref.get("variety", "").lower()
    expected_mrp = sku_ref.get("mrp", "")
    
    mrp_digits = "".join([c for c in expected_mrp if c.isdigit()])
    if len(mrp_digits) > 4 and mrp_digits.endswith("00"):
        mrp_price_digits = mrp_digits[:-2]
    else:
        mrp_price_digits = mrp_digits

    # 1. Match Variety name in OCR
    variety_matched = False
    clean_variety = expected_variety.replace(" ", "")
    clean_ocr = ocr_lower.replace(" ", "")
    if clean_variety in clean_ocr:
        variety_matched = True
    else:
        parts = expected_variety.split()
        if all(part in ocr_lower for part in parts if len(part) > 2):
            variety_matched = True

    # 2. Match MRP price digits
    mrp_matched = False
    if mrp_price_digits and mrp_price_digits in clean_ocr:
        mrp_matched = True
        
    # 3. Match Product Code
    prod_code_matched = False
    expected_prod_code = sku_ref.get("product_code", "").lower()
    if expected_prod_code and expected_prod_code.replace(" ", "") in clean_ocr:
        prod_code_matched = True

    # Verification status decision logic
    if variety_matched:
        if mrp_matched or prod_code_matched:
            return "VERIFIED", f"Variety '{sku_ref.get('variety')}' matches QR. MRP/code matched reference."
        else:
            return "VERIFIED", f"Variety '{sku_ref.get('variety')}' matches QR. Other reference fields unverified."
    else:
        # Check if there is an explicit mismatch with another variety
        different_variety_found = None
        for other_key in ["gravite", "ortholex", "maxi_pro", "maxi_plush"]:
            if other_key == sku_key:
                continue
            other_ref = refs.get(other_key)
            other_variety = other_ref.get("variety", "").lower()
            if other_variety in ocr_lower:
                different_variety_found = other_ref.get("variety")
                break
                
        if different_variety_found:
            return "MISMATCH", f"CRITICAL: QR says '{sku_ref.get('variety')}' but label OCR shows '{different_variety_found}' variety!"
        else:
            return "UNVERIFIED", f"Verification incomplete: variety '{sku_ref.get('variety')}' not verified in OCR."
