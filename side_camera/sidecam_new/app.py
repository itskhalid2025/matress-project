"""
app.py — Flask Backend Application for Side Camera Tile-Based Mattress Inspection System.

Pipeline:
  1. Capture full resolution camera frame
  2. Save original image permanently (captures/YYYYMMDD_HHMMSS.jpg) — never resized
  3. Divide frame into 640x640 overlapping tiles (25-30% overlap)
  4. Save debug tiles to debug_tiles/ folder (draw green border if QR found)
  5. Run QR detection (qrtest.py) on tiles — stop QR search once found
  6. Run Tesseract OCR on tiles, merge & deduplicate text lines
  7. Print terminal debug logs per tile and final combined OCR block
  8. Append structured record to inspection_records.json
  9. Return JSON response for asynchronous UI update
"""

import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2

# Add current directory to path for relative imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from camera_manager import SingleCameraManager
from tile_pipeline import run_tile_pipeline

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Define paths
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
RECORDS_FILE = os.path.join(BASE_DIR, "inspection_records.json")

# Ensure captures folder and JSON records file exist
os.makedirs(CAPTURES_DIR, exist_ok=True)
if not os.path.exists(RECORDS_FILE):
    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# Initialize global camera manager instance
camera_mgr = SingleCameraManager(camera_index=0, width=1920, height=1080, fps=30)


def load_records():
    """Reads inspection records from JSON file."""
    if not os.path.exists(RECORDS_FILE):
        return []
    try:
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[app] Error reading {RECORDS_FILE}: {e}")
        return []


def save_records(records):
    """Writes inspection records list to JSON file."""
    try:
        with open(RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
    except Exception as e:
        print(f"[app] Error writing {RECORDS_FILE}: {e}")


def generate_mjpeg_stream():
    """Generator function for MJPEG stream endpoint."""
    while True:
        frame_bytes = camera_mgr.get_mjpeg_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS output stream rate


@app.route("/")
def index():
    """Serves the main inspection dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Returns live camera stream without frame processing."""
    return Response(
        generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/process", methods=["POST"])
def process_image():
    """
    Triggered when operator clicks 'PROCESS IMAGE'.
    Runs full tile-based sliding window inspection workflow.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    filename_base = now.strftime("%Y%m%d_%H%M%S")
    
    orig_filename = f"{filename_base}.jpg"
    
    # Avoid duplicate file names if called multiple times in same second
    counter = 1
    orig_path = os.path.join(CAPTURES_DIR, orig_filename)
    while os.path.exists(orig_path):
        orig_filename = f"{filename_base}_{counter}.jpg"
        orig_path = os.path.join(CAPTURES_DIR, orig_filename)
        counter += 1

    # 1. Freeze camera feed on process to freeze frame for operator
    camera_mgr.pause()

    # 2. Capture current frame from camera manager at full resolution
    full_frame = camera_mgr.capture_current_frame()

    # 2. Execute sliding window tile inspection pipeline (will draw neon bounding box on full_frame in-place)
    pipeline_res = run_tile_pipeline(full_frame)

    # 3. Save full-resolution image permanently (with bounding box if detected)
    success_orig = cv2.imwrite(orig_path, full_frame)
    if not success_orig:
        return jsonify({
            "success": False,
            "error": "Failed to save captured original image."
        }), 500

    relative_orig_path = f"captures/{orig_filename}"

    # 4. Save cropped label image if detected in pipeline
    label_crop = pipeline_res.get("label_crop")
    label_detected = label_crop is not None
    relative_label_path = None

    if label_detected:
        label_filename = f"label_{orig_filename}"
        label_path = os.path.join(CAPTURES_DIR, label_filename)
        success_label = cv2.imwrite(label_path, label_crop)
        if success_label:
            relative_label_path = f"captures/{label_filename}"
            print(f"[app] Cropped label saved to {label_path}")

    # 5. Construct structured JSON record
    new_record = {
        "timestamp": timestamp_str,
        "image_path": relative_orig_path,
        "label_path": relative_label_path,
        "label_detected": label_detected,
        "qr": {
            "product_name": pipeline_res["qr"]["product_name"],
            "batch_no": pipeline_res["qr"]["batch_no"],
            "inventory_item_id": pipeline_res["qr"]["inventory_item_id"]
        },
        "ocr": {
            "text": pipeline_res["ocr"]["text"]
        },
        "verification": {
            "verdict": pipeline_res.get("verification", {}).get("verdict", "UNVERIFIED"),
            "detail": pipeline_res.get("verification", {}).get("detail", "")
        }
    }

    # 6. Append to JSON storage (never overwrite previous inspections)
    records = load_records()
    records.append(new_record)
    save_records(records)

    # Reverse order for history (newest first)
    records_newest_first = list(reversed(records))

    return jsonify({
        "success": True,
        "latest": new_record,
        "records": records_newest_first
    })


@app.route("/resume", methods=["POST"])
def resume_stream():
    """Triggered when operator resumes inspection. Unfreezes the camera feed."""
    camera_mgr.resume()
    return jsonify({"success": True})


@app.route("/records", methods=["GET"])
def get_records():
    """Returns all historical records (newest first)."""
    records = load_records()
    return jsonify({
        "success": True,
        "records": list(reversed(records))
    })


@app.route("/captures/<filename>")
def serve_capture(filename):
    """Serves saved inspection photos from the captures folder."""
    return send_from_directory(CAPTURES_DIR, filename)


if __name__ == "__main__":
    print("=====================================================")
    print("  Mattress Side Camera Tile-Based Inspection System")
    print("  Server starting at http://127.0.0.1:8005")
    print("=====================================================")
    app.run(host="0.0.0.0", port=8005, debug=False, threaded=True)
