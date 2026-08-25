"""
app.py — Flask Web Application for Calibrated Mattress Dimension System.

Features:
- Live MJPEG USB Webcam video streaming (`/video_feed`)
- Step 1: Interactive Black Border Detection & 4-Edge Overlay
- Step 2: Step-by-Step 4-Edge Length Input API (Top, Right, Bottom, Left)
- Step 3: Interactive 'Process' Button for Mattress Dimension Measurement
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

DIMENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIMENSION_DIR not in sys.path:
    sys.path.insert(0, DIMENSION_DIR)

from core import MattressDimensionEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

# System Application State
app_state = {
    "mode": "BORDER_SETUP",  # "BORDER_SETUP" | "READY" | "PROCESSED"
    "active_edge": "top",     # "top" | "right" | "bottom" | "left"
    "border_color": "black",
    "edge_lengths": {
        "top": 100.0,
        "right": 120.0,
        "bottom": 100.0,
        "left": 120.0
    },
    "calibrated": False,
    "last_result": None
}

# Initialize Engine
engine = MattressDimensionEngine(
    ref_width_cm=100.0,
    ref_height_cm=120.0,
    border_color_mode="black",
    edge_lengths=app_state["edge_lengths"]
)

# Initialize Camera (USB Webcam Index 0)
camera_cap = None


def get_camera():
    global camera_cap
    if camera_cap is None or not camera_cap.isOpened():
        camera_cap = cv2.VideoCapture(0)
        if not camera_cap.isOpened():
            camera_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera_cap.isOpened():
            camera_cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if camera_cap.isOpened():
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera_cap


def generate_video_frames():
    """Generator yielding JPEG frames for MJPEG video stream."""
    global app_state, engine
    cap = get_camera()

    while True:
        if cap is None or not cap.isOpened():
            # Generate blank error frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA NOT AVAILABLE", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            cap = get_camera()
            continue

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # Render visual frame based on current operational mode
        if app_state["mode"] == "BORDER_SETUP" or app_state["mode"] == "READY":
            annotated_frame, border_info = engine.detect_and_draw_border_overlay(
                frame,
                active_edge=app_state["active_edge"]
            )
            app_state["calibrated"] = border_info["detected"]
        elif app_state["mode"] == "PROCESSED":
            res, annotated_frame, _ = engine.process_frame(
                frame,
                ref_width_cm=(app_state["edge_lengths"]["top"] + app_state["edge_lengths"]["bottom"]) / 2.0,
                ref_height_cm=(app_state["edge_lengths"]["left"] + app_state["edge_lengths"]["right"]) / 2.0,
                border_color_mode=app_state["border_color"]
            )
            app_state["last_result"] = res

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main Web Dashboard Page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG Live Stream Endpoint."""
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns current system state and edge calibration data."""
    return jsonify({
        "success": True,
        "state": app_state
    })


@app.route('/api/set_active_edge', methods=['POST'])
def set_active_edge():
    """Sets active edge to highlight on stream ('top', 'right', 'bottom', 'left')."""
    data = request.json or {}
    edge = str(data.get("edge", "top")).lower()
    if edge in ["top", "right", "bottom", "left"]:
        app_state["active_edge"] = edge
        return jsonify({"success": True, "active_edge": edge})
    return jsonify({"success": False, "error": "Invalid edge name"}), 400


@app.route('/api/set_edge_lengths', methods=['POST'])
def set_edge_lengths():
    """
    Submits physical lengths (cm) for 4 edges (Top, Right, Bottom, Left)
    and updates system calibration.
    """
    data = request.json or {}
    top = float(data.get("top", app_state["edge_lengths"]["top"]))
    right = float(data.get("right", app_state["edge_lengths"]["right"]))
    bottom = float(data.get("bottom", app_state["edge_lengths"]["bottom"]))
    left = float(data.get("left", app_state["edge_lengths"]["left"]))

    app_state["edge_lengths"] = {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left
    }

    engine.calibrator.update_edge_lengths(top_cm=top, right_cm=right, bottom_cm=bottom, left_cm=left)
    engine.calculator.ref_width_cm = engine.calibrator.ref_width_cm
    engine.calculator.ref_height_cm = engine.calibrator.ref_height_cm

    app_state["mode"] = "READY"
    print(f"[Flask API] Updated 4-Edge Calibration: {app_state['edge_lengths']}")

    return jsonify({
        "success": True,
        "message": "4-Edge Calibration saved successfully!",
        "edge_lengths": app_state["edge_lengths"],
        "ref_width_cm": engine.calibrator.ref_width_cm,
        "ref_height_cm": engine.calibrator.ref_height_cm
    })


@app.route('/api/process_dimension', methods=['POST'])
def process_dimension():
    """Triggers mattress dimension calculation inside the border."""
    cap = get_camera()
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            res, annotated, _ = engine.process_frame(
                frame,
                ref_width_cm=engine.calibrator.ref_width_cm,
                ref_height_cm=engine.calibrator.ref_height_cm,
                border_color_mode=app_state["border_color"]
            )
            clean_res = {
                "success": res.get("success"),
                "error": res.get("error"),
                "width_cm": res.get("width_cm"),
                "length_cm": res.get("length_cm"),
                "width_in": res.get("width_in"),
                "length_in": res.get("length_in"),
                "area_sq_m": res.get("area_sq_m"),
                "pixel_gaps": res.get("pixel_gaps"),
                "metric_gaps_cm": res.get("metric_gaps_cm")
            }
            app_state["mode"] = "PROCESSED"
            app_state["last_result"] = clean_res
            return jsonify({
                "success": True,
                "result": clean_res
            })

    return jsonify({"success": False, "error": "Camera frame capture failed"}), 500


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Resets mode back to BORDER_SETUP."""
    app_state["mode"] = "BORDER_SETUP"
    app_state["last_result"] = None
    return jsonify({"success": True, "mode": app_state["mode"]})


if __name__ == '__main__':
    print("=" * 65)
    print(" STARTING MATTRESS DIMENSION FLASK WEB APPLICATION")
    print(" Dashboard URL: http://localhost:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
