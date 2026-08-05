import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory, request

from config import RESULTS_DIR, BASE_DIR
from camera_manager import init_cameras, qr_cam_stream, bill_cam_stream, top_cam_stream
from qr_module import scan_qr, parse_qr_payload, draw_qr_overlay, detect_qr_presence
from ocr_module import process_bill_ocr, detect_ocr_presence_fast
from texture_module import predict_texture
from top_camera_module import process_top_camera, detect_and_crop_corner_label
from verification_engine import verify_full_inspection
from storage_manager import save_full_inspection_record, get_all_records

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize 3 Cameras
init_cameras()


def generate_qr_feed():
    """MJPEG Generator for Camera 1 (QR Scanner) with live aiming box."""
    while True:
        frame = qr_cam_stream.read_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for pts in detect_qr_presence(gray):
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), 2)
            cv2.putText(frame, "QR Detected - Click Process", (pts[0][0], max(20, pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_bill_feed():
    """MJPEG Generator for Camera 2 (Side Bill OCR & Texture) with live text region detection."""
    while True:
        frame = bill_cam_stream.read_frame()
        boxes = detect_ocr_presence_fast(frame)
        if boxes:
            for (bx, by, bw, bh) in boxes[:5]:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(frame, f"Bill OCR Region ({len(boxes)} detected)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_top_feed():
    """MJPEG Generator for Camera 3 (Top Camera Dimensions & Corner Label) with live label region box."""
    while True:
        frame = top_cam_stream.read_frame()
        _, (xmin, ymin, xmax, ymax) = detect_and_crop_corner_label(frame)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(frame, "Corner Label Region", (xmin, max(25, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/qr')
def video_feed_qr():
    return Response(generate_qr_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/bill')
def video_feed_bill():
    return Response(generate_bill_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/top')
def video_feed_top():
    return Response(generate_top_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera_status')
def camera_status():
    return jsonify({
        "camera_qr": qr_cam_stream.is_connected(),
        "camera_bill": bill_cam_stream.is_connected(),
        "camera_top": top_cam_stream.is_connected()
    })


@app.route('/api/process', methods=['POST'])
def process_inspection():
    """
    1-Click Simultaneous Process Endpoint across 3 Cameras:
    - Camera 1: QR Scanning
    - Camera 2: Bill OCR & PyTorch Texture AI
    - Camera 3: Top Dimensions & Corner Label OCR
    - Evaluates 4-way identity verification + dimension tolerance checks
    - Saves DB record and image archives.
    """
    # 1. Simultaneous Capture
    raw_qr_frame = qr_cam_stream.read_frame()
    raw_bill_frame = bill_cam_stream.read_frame()
    raw_top_frame = top_cam_stream.read_frame()

    annotated_qr_frame = raw_qr_frame.copy()

    # 2. Camera 1: QR Code
    gray_qr = cv2.cvtColor(raw_qr_frame, cv2.COLOR_BGR2GRAY)
    qr_hits = scan_qr(gray_qr)
    qr_data = {"raw_text": "", "product_name": "Not Detected", "batch_no": "N/A", "inventory_item_id": "N/A"}

    if qr_hits:
        text, pts = qr_hits[0]
        qr_data = parse_qr_payload(text)
        draw_qr_overlay(annotated_qr_frame, pts, qr_data)

    # 3. Camera 2: Side Bill OCR
    ocr_res = process_bill_ocr(raw_bill_frame)
    annotated_bill_frame = ocr_res["annotated_frame"]

    # 4. Camera 2: PyTorch Texture AI
    texture_res = predict_texture(raw_bill_frame)

    # 5. Camera 3: Top Dimensions & Corner Label OCR
    top_res = process_top_camera(raw_top_frame)

    # 6. Dual-Layer 4-Way Verification
    verification_res = verify_full_inspection(qr_data, ocr_res, texture_res, top_res)

    # 7. Save to DB & Disk
    record = save_full_inspection_record(
        raw_qr_frame,
        raw_bill_frame,
        raw_top_frame,
        annotated_qr_frame,
        annotated_bill_frame,
        top_res,
        qr_data,
        ocr_res,
        texture_res,
        verification_res
    )

    return jsonify({
        "success": True,
        "record": record
    })


@app.route('/api/history')
def history():
    records = get_all_records()
    return jsonify(records)


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == '__main__':
    print("\n==================================================")
    print("     UNIFIED MATTRESS INSPECTION SYSTEM DASHBOARD  ")
    print("==================================================")
    print(" Server launching on http://127.0.0.1:5000")
    print("==================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
