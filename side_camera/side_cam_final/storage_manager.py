import os
import json
import datetime
import cv2
from config import RESULTS_DIR, DB_PATH


def save_full_inspection_record(raw_qr_frame, raw_bill_frame, raw_top_frame,
                                annotated_qr_frame, annotated_ocr_frame, top_res,
                                qr_data, ocr_data, texture_data, verification_res):
    """
    Saves raw & annotated images for 3 cameras to disk and logs 4-way inspection record to database.json.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    inspection_id = f"INS_MATTRESS_{timestamp}"

    folder_name = f"inspection_{timestamp}"
    run_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save images
    raw_qr_path = os.path.join(run_dir, "raw_qr_cam1.jpg")
    raw_bill_path = os.path.join(run_dir, "raw_bill_cam2.jpg")
    raw_top_path = os.path.join(run_dir, "raw_top_cam3.jpg")

    annotated_qr_path = os.path.join(run_dir, "annotated_qr.jpg")
    annotated_ocr_path = os.path.join(run_dir, "annotated_ocr.jpg")
    annotated_top_path = os.path.join(run_dir, "annotated_top_dims.jpg")
    corner_label_crop_path = os.path.join(run_dir, "corner_label_crop.jpg")

    cv2.imwrite(raw_qr_path, raw_qr_frame)
    cv2.imwrite(raw_bill_path, raw_bill_frame)
    cv2.imwrite(raw_top_path, raw_top_frame)

    cv2.imwrite(annotated_qr_path, annotated_qr_frame)
    cv2.imwrite(annotated_ocr_path, annotated_ocr_frame)
    cv2.imwrite(annotated_top_path, top_res["annotated_frame"])
    if top_res.get("corner_label_crop") is not None and top_res["corner_label_crop"].size > 0:
        cv2.imwrite(corner_label_crop_path, top_res["corner_label_crop"])

    # Construct JSON record
    record = {
        "inspection_id": inspection_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "overall_status": verification_res["overall_status"],
        "identity_status": verification_res["identity_status"],
        "product_variety": "Ortholex",
        "item_id": qr_data.get("inventory_item_id", "ORTHO-1001"),
        "batch_number": qr_data.get("batch_no", "BATCH-2026-A"),
        "qr_code_data": qr_data,
        "side_ocr_data": {
            "full_text": ocr_data.get("full_text", "Ortholex"),
            "average_confidence": ocr_data.get("avg_confidence", 100.0),
            "items": ocr_data.get("extracted_items", [])
        },
        "texture_data": texture_data,
        "top_camera_data": {
            "corner_label": top_res.get("corner_label", {})
        },
        "verification_result": verification_res,
        "image_paths": {
            "raw_qr_camera": os.path.relpath(raw_qr_path, RESULTS_DIR),
            "raw_bill_camera": os.path.relpath(raw_bill_path, RESULTS_DIR),
            "raw_top_camera": os.path.relpath(raw_top_path, RESULTS_DIR),
            "annotated_qr": os.path.relpath(annotated_qr_path, RESULTS_DIR),
            "annotated_ocr": os.path.relpath(annotated_ocr_path, RESULTS_DIR),
            "annotated_top": os.path.relpath(annotated_top_path, RESULTS_DIR),
            "corner_label_crop": os.path.relpath(corner_label_crop_path, RESULTS_DIR)
        }
    }

    # Append to database.json
    records = []
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    records.insert(0, record)  # Newest first

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return record


def get_all_records():
    """Retrieves all inspection records from database.json."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
