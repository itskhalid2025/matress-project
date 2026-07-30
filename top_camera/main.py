"""
main.py — Self-Contained Web Dashboard for Top Camera Rig (Dimensions + Banner Checking).

Serves a premium Glassmorphic Dark-Mode UI on http://localhost:8002.
Provides:
  1. Live MJPEG stream from the webcam with real-time dimension bounding box annotations.
  2. Live calculations for Width & Height (in cm) using a calibrated PIXELS_PER_CM ratio.
  3. Real-time Banner/Sash checking (ORB visual matching and pytesseract OCR).
  4. Interactive dynamic calibration: Enter known size to update PIXELS_PER_CM on-the-fly.
  5. Hot-swapping camera indices directly from the web interface.
  6. Robust error handling (graceful camera reconnection, missing YOLO models, empty frames).
"""

import sys
import os
import argparse
import time
import json
import threading
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# Add parent directory to path to ensure correct relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from camera import ThreadedCamera
from dimensions import measure_dimensions
from banner import read_banner, read_banner_fast

# ==============================================================================
# Global State Managers
# ==============================================================================
camera_thread = None
pixels_per_cm = cfg.PIXELS_PER_CM  # load default from config.py

# Keep track of last verification outcomes
last_verdict = {
    "status": "Ready",
    "width_cm": None,
    "height_cm": None,
    "banner_sku": None,
    "banner_text": "",
    "method": "None",
    "timestamp": ""
}

# Lock to avoid concurrent analysis of camera frames
analysis_lock = threading.Lock()

# In-memory buffer for the last manual processed and annotated image frame
last_processed_image_bytes = None

# Placeholder image for camera disconnected state
def get_error_frame(message="CAMERA DISCONNECTED"):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Background pattern
    for i in range(0, 1080, 40):
        cv2.line(img, (0, i), (1920, i), (15, 20, 30), 1)
    for i in range(0, 1920, 40):
        cv2.line(img, (i, 0), (i, 1080), (15, 20, 30), 1)
        
    cv2.putText(img, "TOP CAMERA RIG", (760, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (18, 165, 165), 4)
    cv2.putText(img, message.upper(), (600, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
    cv2.putText(img, "Adjust camera index or verify USB connection", (640, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 130, 140), 2)
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

# ==============================================================================
# Web Server Request Handler
# ==============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    """
    HTTP server handler providing:
      - GET / or /index.html: The UI dashboard.
      - GET /stream: Live MJPEG stream with dimension markings.
      - GET /api/status: Returns current status, dimensions, and banner data in JSON.
      - POST /api/set_camera?index=N: Changes the webcam index.
      - POST /api/calibrate?known_width_cm=X: Calibrates PIXELS_PER_CM.
      - POST /api/verify: Triggers a high-accuracy validation pass (deep banner OCR sweep).
    """
    
    def log_message(self, format, *args):
        # Keep console clean from spamming preview stream requests
        pass

    def do_GET(self):
        global pixels_per_cm
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # 1. Main Dashboard HTML serving
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode('utf-8'))
            return

        # 2. Live MJPEG Webcam Stream
        elif path == "/stream":
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            # Simple frame ticker to skip intensive banner processing every frame
            frame_ticker = 0
            live_banner_sku = "None"
            live_banner_text = ""

            while True:
                if camera_thread is None:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame("Rig camera thread uninitialized") + b'\r\n')
                    time.sleep(0.5)
                    continue

                frame, err = camera_thread.read()
                if frame is None or err:
                    msg = err if err else "Webcam disconnected / no frame available"
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame(msg) + b'\r\n')
                    time.sleep(0.1)
                    continue

                # Draw only the static crop ROI boundary visual guide (no YOLO/contour/banner analysis on stream)
                annotated = frame.copy()
                y0, y1, x0, x1 = cfg.FIXED_CROP
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (180, 180, 180), 1)
                cv2.putText(annotated, "FIXED CROP ROI", (x0 + 10, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                # Encode frame to JPEG
                ret, encoded_jpeg = cv2.imencode('.jpg', annotated)
                if not ret:
                    continue

                try:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_jpeg.tobytes() + b'\r\n')
                except (ConnectionResetError, BrokenPipeError):
                    break
                time.sleep(0.03)  # limit stream to ~30 FPS
            return

        # 3. GET /api/status JSON Endpoint
        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Do not run background measurement checking to save CPU
            current_dim = {"width_cm": None, "height_cm": None, "raw_pixels_w": 0, "raw_pixels_h": 0}
                    
            status_data = {
                "pixels_per_cm": pixels_per_cm,
                "current_camera_index": camera_thread.index if camera_thread else cfg.CAM_INDEX,
                "current_live_dimensions": current_dim,
                "last_verification": last_verdict
            }
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            return

        # 4. GET /api/last_processed.jpg dynamic JPEG serving
        elif path == "/api/last_processed.jpg":
            global last_processed_image_bytes
            if last_processed_image_bytes is not None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.end_headers()
                self.wfile.write(last_processed_image_bytes)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No frame processed yet")
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")
            return

    def do_POST(self):
        global pixels_per_cm, last_verdict
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # 1. Calibrate PIXELS_PER_CM
        if path == "/api/calibrate":
            if "known_width_cm" not in query:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing query parameter: known_width_cm")
                return

            try:
                known_w = float(query["known_width_cm"][0])
                if known_w <= 0:
                    raise ValueError("Dimensions must be positive")
            except ValueError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid known_width_cm value")
                return

            if camera_thread is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Camera stream offline")
                return

            frame, _ = camera_thread.read()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Failed to capture frame for calibration")
                return

            # Measure contour width in PIXELS (passing None as pixels_per_cm returns raw pixels)
            raw_w_px, _, _ = measure_dimensions(
                img=frame,
                pixels_per_cm=None,
                edge_correction=cfg.EDGE_CORRECTION_FACTOR,
                min_contour_area=cfg.MIN_CONTOUR_AREA,
                fixed_crop=cfg.FIXED_CROP
            )

            if raw_w_px is None or raw_w_px <= 0:
                self.send_response(422)
                self.end_headers()
                self.wfile.write(b"No stable mattress contour detected in crop ROI. Place object and try again.")
                return

            # W_px / known_cm = pixels_per_cm (without applying correction factor in the denominator)
            # Since nW = (w_px / pixels_per_cm) * edge_correction, then:
            # known_w = (raw_w_px / pixels_per_cm) * edge_correction => pixels_per_cm = (raw_w_px * edge_correction) / known_w
            new_ratio = (raw_w_px * cfg.EDGE_CORRECTION_FACTOR) / known_w
            pixels_per_cm = round(new_ratio, 3)
            
            print(f"[calibration] Dynamic calibration updated: {pixels_per_cm} pixels/cm derived from {known_w}cm object.")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "pixels_per_cm": pixels_per_cm}).encode('utf-8'))
            return

        # 2. Hot-swap camera device index
        elif path == "/api/set_camera":
            if "index" not in query:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing index parameter")
                return

            try:
                new_idx = int(query["index"][0])
            except ValueError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid index value")
                return

            if camera_thread is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Camera stream offline")
                return

            success = camera_thread.change_camera(new_idx)
            if success:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "index": new_idx}).encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Failed to change camera to index {new_idx}".encode('utf-8'))
            return

        # 3. High-Accuracy Deep Verification (Runs full Stage A and B checks)
        elif path == "/api/verify":
            if camera_thread is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Camera stream offline")
                return

            frame, _ = camera_thread.read()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Failed to capture frame")
                return

            # Trigger analysis
            print("[verify] Initializing high-accuracy verification pass...")
            with analysis_lock:
                # 1. Run dimensions check (capturing the annotated image output)
                w_cm, h_cm, annotated = measure_dimensions(
                    img=frame,
                    pixels_per_cm=pixels_per_cm,
                    edge_correction=cfg.EDGE_CORRECTION_FACTOR,
                    min_contour_area=cfg.MIN_CONTOUR_AREA,
                    fixed_crop=cfg.FIXED_CROP
                )

                # 2. Run banner checking (including fallback sweep if necessary)
                y0, y1, x0, x1 = cfg.FIXED_CROP
                crop_bgr = frame[y0:y1, x0:x1]
                
                sku, angle, text, details = read_banner(crop_bgr)
                
                # Determine classification path/method used
                method = "None"
                if sku:
                    if angle is None:
                        # Stage A trigger
                        if "visual template" in text:
                            method = "Stage A (ORB Visual Template Match)"
                        else:
                            method = "Stage A (Local Band OCR)"
                    else:
                        method = f"Stage B (Rotation Sweep OCR at {angle} deg)"
                else:
                    method = "Not Found (OCR/Template mismatch)"

                # Draw the static crop ROI border on the annotated frame
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (180, 180, 180), 1)
                cv2.putText(annotated, "FIXED CROP ROI", (x0 + 10, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                # Cache the annotated frame to global in-memory buffer
                ret, encoded_jpeg = cv2.imencode('.jpg', annotated)
                if ret:
                    global last_processed_image_bytes
                    last_processed_image_bytes = encoded_jpeg.tobytes()

                # Save results to global state
                last_verdict = {
                    "status": "Success" if sku else "Mismatch/Absent",
                    "width_cm": w_cm,
                    "height_cm": h_cm,
                    "banner_sku": sku.upper() if sku else "UNKNOWN",
                    "banner_text": text,
                    "method": method,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(last_verdict).encode('utf-8'))
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")
            return

# ==============================================================================
# HTML Interface Design Template
# ==============================================================================
def get_dashboard_html():
    """Returns the Glassmorphic Dark-mode dashboard HTML & inline styling."""
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Mattress QC Rig — Top Camera Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: rgba(22, 28, 45, 0.65);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f1f5f9;
            --text-dim: #94a3b8;
            --teal: #12a5a5;
            --teal-hover: #17c2c2;
            --magenta: #e81b84;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #ef4444;
        }

        body {
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-image: radial-gradient(circle at 10% 20%, rgba(18, 165, 165, 0.05) 0%, transparent 60%);
        }

        .container {
            max-width: 1300px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 25px;
        }

        header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        header h1 span {
            color: var(--teal);
        }

        .cam-selector {
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            padding: 8px 15px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }

        .cam-selector select {
            background: #0f172a;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 5px 10px;
            border-radius: 8px;
            outline: none;
            font-weight: 600;
            cursor: pointer;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 24px;
        }

        .main-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .preview-card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            position: relative;
        }

        .preview-card h2 {
            margin: 0;
            padding: 16px 20px;
            font-size: 16px;
            font-weight: 700;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--green);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stream-container {
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .stream-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Glass Cards */
        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(15px);
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.25s ease;
        }

        .card:hover {
            border-color: rgba(18, 165, 165, 0.2);
        }

        .card h3 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 14px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Verification Action Bar */
        .control-bar {
            display: flex;
            gap: 15px;
            margin-top: 5px;
        }

        .btn {
            padding: 16px 24px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 16px;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
            outline: none;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--teal) 0%, #0d7f7f 100%);
            color: white;
            flex-grow: 1;
            box-shadow: 0 4px 15px rgba(18, 165, 165, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(18, 165, 165, 0.4);
        }

        .btn-primary:active {
            transform: translateY(1px);
        }

        /* Right Column Panels */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .data-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .data-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .data-row:last-child {
            border-bottom: none;
        }

        .data-label {
            color: var(--text-dim);
            font-size: 13.5px;
        }

        .data-value {
            font-weight: 700;
            font-size: 15px;
        }

        .highlight-value {
            font-size: 28px;
            font-weight: 850;
            color: var(--teal);
        }

        .sku-value {
            font-size: 28px;
            font-weight: 850;
            color: var(--magenta);
            letter-spacing: 0.5px;
        }

        /* Calibration Form */
        .calib-form {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .calib-form input {
            flex-grow: 1;
            background: #0f172a;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 10px 14px;
            border-radius: 8px;
            outline: none;
            font-size: 14px;
            font-weight: 600;
        }

        .calib-form button {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }

        .calib-form button:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* Verdict Notification Badge */
        .verdict-header {
            text-align: center;
            padding: 15px 0;
            border-radius: 12px;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }

        .verdict-success {
            background: rgba(16, 185, 129, 0.12);
            color: var(--green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .verdict-alert {
            background: rgba(239, 68, 68, 0.12);
            color: var(--red);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .verdict-pending {
            background: rgba(245, 158, 11, 0.12);
            color: var(--amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Mattress QC <span>Top Camera Rig</span></h1>
            <div class="cam-selector">
                <label for="cameraSelect">Webcam Index:</label>
                <select id="cameraSelect" onchange="swapCamera(this.value)">
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2" selected>2 (Default)</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                </select>
            </div>
        </header>

        <div class="dashboard-grid">
            <!-- Left Main Column (Livestream and Trigger) -->
            <div class="main-column">
                <div class="preview-card">
                    <h2>Live Conveyor Feed <span class="status-badge" id="cameraStatus">Online</span></h2>
                    <div class="stream-container">
                        <img id="webcamStream" src="/stream" alt="Live camera preview showing dimensions ROI">
                    </div>
                </div>

                <div class="control-bar" style="display: flex; gap: 10px;">
                    <button class="btn btn-primary" id="btnVerify" onclick="triggerVerification()" style="flex: 1;">PROCESS FRAME</button>
                    <button class="btn" id="btnResume" onclick="resumeLiveFeed()" style="display: none; background: rgba(255, 255, 255, 0.08); border: 1px solid var(--border-color); color: var(--text-color); padding: 16px 24px; border-radius: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s;">RESUME LIVE FEED</button>
                </div>
            </div>

            <!-- Right Sidebar (Data logs & calibration) -->
            <div class="sidebar">
                <!-- Processed Results Card -->
                <div class="card" id="resultsCard">
                    <h3>Processed Results</h3>
                    <div id="verdictBadge" class="verdict-header verdict-pending">READY TO SCAN</div>
                    
                    <div class="data-panel">
                        <div class="data-row">
                            <span class="data-label">Width (cm)</span>
                            <span class="data-value highlight-value" id="liveWidth">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Length (cm)</span>
                            <span class="data-value highlight-value" id="liveLength">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Detected Banner SKU</span>
                            <span class="data-value sku-value" id="outcomeSku">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Banner Match Method</span>
                            <span class="data-value" id="outcomeMethod">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Raw Banner OCR text</span>
                            <span class="data-value" id="outcomeOcr" style="font-family:monospace; font-size:12px; color:var(--amber);">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Calibration Factor</span>
                            <span class="data-value" id="liveRatio">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Timestamp</span>
                            <span class="data-value" id="outcomeTime" style="font-size:12px; color:var(--text-dim);">—</span>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 15px;">
                        <span class="data-label" style="display:block; font-size:12px; margin-bottom:5px;">Calibrate with Known Object in Frame:</span>
                        <div class="calib-form">
                            <input type="number" id="calibWidth" placeholder="Known width (cm)" min="1" step="0.1">
                            <button onclick="submitCalibration()">Calibrate</button>
                        </div>
                        <span id="calibFeedback" style="display:block; font-size:11px; margin-top:6px; color:var(--text-dim);"></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Fetch dashboard status regularly
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update camera selection dropdown if changed outside
                    document.getElementById('cameraSelect').value = data.current_camera_index;
                    document.getElementById('liveRatio').innerText = data.pixels_per_cm ? data.pixels_per_cm.toFixed(2) + " px/cm" : "Uncalibrated";
                    
                    // Update verification / processed results
                    const verify = data.last_verification;
                    if (verify.timestamp) {
                        document.getElementById('outcomeSku').innerText = verify.banner_sku;
                        document.getElementById('outcomeMethod').innerText = verify.method;
                        document.getElementById('outcomeOcr').innerText = verify.banner_text ? verify.banner_text : "No text matches";
                        document.getElementById('outcomeTime').innerText = verify.timestamp;

                        if (verify.width_cm) {
                            document.getElementById('liveWidth').innerText = verify.width_cm.toFixed(1) + " cm";
                        } else {
                            document.getElementById('liveWidth').innerText = "—";
                        }

                        if (verify.height_cm) {
                            document.getElementById('liveLength').innerText = verify.height_cm.toFixed(1) + " cm";
                        } else {
                            document.getElementById('liveLength').innerText = "—";
                        }

                        const badge = document.getElementById('verdictBadge');
                        if (verify.status === "Success") {
                            badge.innerText = "VERIFIED PASS";
                            badge.className = "verdict-header verdict-success";
                        } else {
                            badge.innerText = "QC ALERT: MISSING/MISMATCH";
                            badge.className = "verdict-header verdict-alert";
                        }
                    } else {
                        document.getElementById('liveWidth').innerText = "—";
                        document.getElementById('liveLength').innerText = "—";
                        document.getElementById('outcomeSku').innerText = "—";
                        document.getElementById('outcomeMethod').innerText = "—";
                        document.getElementById('outcomeOcr').innerText = "—";
                        document.getElementById('outcomeTime').innerText = "—";
                    }
                })
                .catch(err => {
                    console.error("Status polling failed:", err);
                    document.getElementById('cameraStatus').innerText = "Offline";
                    document.getElementById('cameraStatus').className = "status-badge";
                });
        }

        // Calibrate ratio dynamically
        function submitCalibration() {
            const val = document.getElementById('calibWidth').value;
            const feedback = document.getElementById('calibFeedback');
            
            if (!val || parseFloat(val) <= 0) {
                feedback.innerText = "Please enter a valid size in cm.";
                feedback.style.color = "var(--red)";
                return;
            }

            feedback.innerText = "Calibrating ratio based on current frame...";
            feedback.style.color = "var(--amber)";

            fetch(`/api/calibrate?known_width_cm=${val}`, { method: 'POST' })
                .then(res => {
                    if (res.ok) return res.json();
                    return res.text().then(text => { throw new Error(text) });
                })
                .then(data => {
                    feedback.innerText = `Calibrated successfully: ${data.pixels_per_cm.toFixed(2)} px/cm`;
                    feedback.style.color = "var(--green)";
                    updateStatus();
                })
                .catch(err => {
                    feedback.innerText = err.message;
                    feedback.style.color = "var(--red)";
                });
        }

        // Swapping camera indices dynamically
        function swapCamera(idx) {
            fetch(`/api/set_camera?index=${idx}`, { method: 'POST' })
                .then(res => {
                    if (res.ok) {
                        console.log(`Swapped to camera ${idx}`);
                        // Reload stream to force reconnecting image source
                        const img = document.getElementById('webcamStream');
                        img.src = '/stream?t=' + new Date().getTime();
                        // Hide resume button if live stream is restarted
                        document.getElementById('btnResume').style.display = 'none';
                    } else {
                        alert("Failed to connect to camera at index " + idx);
                    }
                });
        }

        // Trigger comprehensive QC check
        function triggerVerification() {
            const btn = document.getElementById('btnVerify');
            const badge = document.getElementById('verdictBadge');
            
            btn.disabled = true;
            btn.innerText = "PROCESSING FRAME...";
            badge.innerText = "PROCESSING...";
            badge.className = "verdict-header verdict-pending";

            fetch('/api/verify', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    btn.disabled = false;
                    btn.innerText = "PROCESS FRAME";
                    
                    // Switch camera display to the captured static annotated image
                    const img = document.getElementById('webcamStream');
                    img.src = '/api/last_processed.jpg?t=' + new Date().getTime();
                    
                    // Show the Resume button
                    document.getElementById('btnResume').style.display = 'inline-block';
                    
                    updateStatus();
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.innerText = "PROCESS FRAME";
                    alert("Verification check failed: " + err);
                });
        }

        // Restore live video stream
        function resumeLiveFeed() {
            const img = document.getElementById('webcamStream');
            img.src = '/stream';
            
            // Hide the Resume button
            document.getElementById('btnResume').style.display = 'none';
        }

        // Start status updater polling
        updateStatus();
        setInterval(updateStatus, 1200);
    </script>
</body>
</html>
"""

# ==============================================================================
# Executable Entry Point
# ==============================================================================
def main():
    global camera_thread
    
    parser = argparse.ArgumentParser(description="Mattress Top Camera QC Rig Server")
    parser.add_argument("--index", type=int, default=cfg.CAM_INDEX, help="Webcam device index (default: from config)")
    parser.add_argument("--port", type=int, default=8002, help="HTTP dashboard port (default: 8002)")
    parser.add_argument("--width", type=int, default=cfg.CAPTURE_W, help="Capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=cfg.CAPTURE_H, help="Capture height (default: 1080)")
    args = parser.parse_args()

    print("======================================================================")
    print("MATTRESS TOP CAMERA QUALITY CONTROL SYSTEM")
    print("======================================================================")
    print(f"Initializing camera thread on index {args.index} at {args.width}x{args.height}...")

    # Start background capture thread
    try:
        camera_thread = ThreadedCamera(index=args.index, width=args.width, height=args.height)
    except Exception as e:
        print(f"FATAL: Could not initialize camera thread: {str(e)}")
        sys.exit(1)

    # Launch local HTTP server
    server_address = ('', args.port)
    try:
        httpd = ThreadingHTTPServer(server_address, DashboardHandler)
        print(f"Dashboard successfully launched at http://localhost:{args.port}")
        print("Press Ctrl+C to terminate the rig server.")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"FATAL: Server crashed: {str(e)}")
    finally:
        if camera_thread:
            camera_thread.release()

if __name__ == "__main__":
    main()
