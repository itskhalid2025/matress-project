#!/usr/bin/env python3
"""
============================================================
FILE: live_dimensions_stream.py
PATH: top_camera/global shutter/live_dimensions_stream.py
============================================================
DESCRIPTION:
  Live web streaming server combining Picamera2 Global Shutter
  IMX296 video capture with real-time YOLOv8 + OpenCV contour
  dimension measurement (rotated bounding box & area).
  Features:
  - High-FPS Picamera2 video capture (IMX296)
  - YOLO AI target isolation (skips human class 0)
  - Real-time FPS overlay & annotated dimension tags
  - Interactive live calibration endpoint & web dashboard

CREATED: 2026-08-01 | 12:53 PM

EDIT LOG:
------------------------------------------------------------
[2026-08-01 | 12:53 PM] - Initial creation of live YOLO dimension stream.
============================================================
"""

import sys
import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# Ensure top_camera directory is in Python path to import dimensions.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if TOP_CAM_DIR not in sys.path:
    sys.path.insert(0, TOP_CAM_DIR)

from dimensions import measure_dimensions

app = Flask(__name__)

# Global Configuration & Calibration State
pixels_per_cm = 10.0  # Default calibration (pixels per cm)
edge_correction = 1.1

import pytesseract
import platform

if platform.system() == "Windows":
    TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_WIN_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_WIN_PATH

def _extract_general_ocr(img):
    """Runs general pytesseract OCR on the target image to extract ANY detected text."""
    if img is None or img.size == 0:
        return ""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply contrast scaling + Gaussian blur + Otsu thresholding
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try both direct and inverted text thresholding
        raw_text_a = pytesseract.image_to_string(thresh, config='--psm 11').strip()
        raw_text_b = pytesseract.image_to_string(255 - thresh, config='--psm 11').strip()
        
        combined = (raw_text_a + " " + raw_text_b).strip()
        clean = " ".join(combined.split())
        return clean[:80]
    except Exception:
        return ""

class LiveDimensionStreamer:
    """Manages high-speed camera capture and real-time YOLO dimension processing."""
    def __init__(self, width=1456, height=1088, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.picam2 = None
        self.cap = None

        self.latest_jpeg = None
        self.latest_dims = {"width_cm": None, "width_in": None, "height_cm": None, "height_in": None, "ocr_text": "", "status": "Initializing"}
        self.latest_ocr_text = ""
        self.lock = threading.Lock()
        self.running = False

        self.fps_counter = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()
        self.frame_index = 0

        self._init_camera()

        # Start dedicated background worker thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_and_process_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # 1. Picamera2 Native High-FPS Video Mode
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            try:
                config = picam.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                picam.configure(config)
                picam.start()
                self.picam2 = picam
                print(f"[live_stream] Connected via Picamera2 ({self.fps} FPS target)")
                return
            except Exception as inner_e:
                picam.close()
                raise inner_e
        except Exception as e:
            print(f"[live_stream] Picamera2 init note: {e}")

        # 2. OpenCV V4L2 Fallback
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                print("[live_stream] Connected via OpenCV V4L2 fallback")
                return
        except Exception as e:
            print(f"[live_stream] OpenCV V4L2 failed: {e}")

    def _capture_and_process_loop(self):
        global pixels_per_cm, edge_correction
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        while self.running:
            raw_bgr = None

            if self.picam2:
                try:
                    rgb = self.picam2.capture_array()
                    raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception:
                    time.sleep(0.005)
                    continue
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    raw_bgr = frame
                else:
                    time.sleep(0.005)
                    continue

            if raw_bgr is None:
                time.sleep(0.01)
                continue

            self.frame_index += 1

            # Run YOLO + OpenCV rotated bounding box measurement
            w_meas, h_meas, annotated_frame = measure_dimensions(
                img=raw_bgr,
                pixels_per_cm=pixels_per_cm,
                edge_correction=edge_correction
            )

            # Run periodic general OCR text extraction every 8 frames (~0.25s) to preserve 30 FPS speed
            if self.frame_index % 8 == 0:
                extracted = _extract_general_ocr(raw_bgr)
                if extracted:
                    self.latest_ocr_text = extracted

            # Draw OCR text directly on live frame if detected
            if self.latest_ocr_text:
                cv2.putText(annotated_frame, f"OCR: {self.latest_ocr_text}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"OCR: {self.latest_ocr_text}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

            # Calculate FPS
            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            # Draw FPS & Status Banner directly on video feed
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(annotated_frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(annotated_frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Compress annotated frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, jpeg_params)
            if ret:
                w_in = round(w_meas / 2.54, 1) if w_meas is not None else None
                h_in = round(h_meas / 2.54, 1) if h_meas is not None else None
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_dims = {
                        "width_cm": w_meas,
                        "width_in": w_in,
                        "height_cm": h_meas,
                        "height_in": h_in,
                        "ocr_text": self.latest_ocr_text,
                        "status": "Target Detected" if w_meas is not None else "Searching..."
                    }

            time.sleep(0.001)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_dims(self):
        with self.lock:
            return self.latest_dims

    def release(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap:
            self.cap.release()

streamer = LiveDimensionStreamer()

def generate_feed():
    last_sent = None
    while True:
        jpeg = streamer.get_jpeg()
        if jpeg is not None and jpeg != last_sent:
            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)

# ==============================================================================
# Web Dashboard Interface
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live YOLO Dimension & OCR Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: rgba(22, 28, 45, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --primary: #06b6d4;
            --magenta: #e81b84;
            --amber: #f59e0b;
            --green: #10b981;
            --text: #f8fafc;
            --dim: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); padding: 20px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }

        .header { width: 100%; max-width: 1200px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 16px 24px; background: var(--panel); border: 1px solid var(--border); border-radius: 16px; backdrop-filter: blur(12px); }
        .header h1 { font-size: 22px; font-weight: 800; }
        .header h1 span { color: var(--primary); }

        .layout { width: 100%; max-width: 1200px; display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }
        
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; backdrop-filter: blur(12px); }
        .video-box { width: 100%; aspect-ratio: 16/10; background: #000; border-radius: 14px; overflow: hidden; display: flex; justify-content: center; align-items: center; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .video-box img { width: 100%; height: 100%; object-fit: contain; }

        .stat-group { display: flex; flex-direction: column; gap: 14px; }
        .stat-card { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); padding: 14px; border-radius: 14px; }
        .stat-label { font-size: 11px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
        .stat-val { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--primary); }
        .stat-val.magenta { color: var(--magenta); }
        .stat-val.amber { color: var(--amber); font-size: 16px; font-weight: 700; word-break: break-word; }
        
        .calib-box { margin-top: 10px; padding-top: 15px; border-top: 1px solid var(--border); }
        .calib-input { display: flex; gap: 8px; margin-top: 8px; }
        .calib-input input { flex: 1; background: #0f172a; border: 1px solid var(--border); color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 14px; }
        .calib-input button { background: var(--primary); color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; }
        .calib-input button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📐 Real-Time YOLO <span>Dimension & OCR Stream</span></h1>
        <div style="color: var(--green); font-weight: 700; font-size: 14px;">● IMX296 ACTIVE</div>
    </div>

    <div class="layout">
        <div class="card">
            <div class="video-box">
                <img src="/video_feed" alt="Live Camera Stream">
            </div>
        </div>

        <div class="card">
            <div class="stat-group">
                <div class="stat-card">
                    <div class="stat-label">Live Width</div>
                    <div class="stat-val" id="valWidth">—</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Live Height / Length</div>
                    <div class="stat-val magenta" id="valHeight">—</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Detected Text / OCR</div>
                    <div class="stat-val amber" id="valOcr">Searching text...</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Detection Status</div>
                    <div class="stat-val" id="valStatus" style="font-size: 16px; color: var(--green);">Searching...</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Calibration Factor</div>
                    <div class="stat-val" id="valCalib" style="font-size: 18px;">{{ pixels_per_cm }} px/cm</div>
                </div>

                <div class="calib-box">
                    <div class="stat-label">Dual-Dimension Live Calibration</div>
                    <div style="font-size: 12px; color: var(--dim); margin-bottom: 6px;">Enter real physical size of target in frame:</div>
                    <div class="calib-input" style="flex-direction: column; gap: 8px;">
                        <input type="number" id="knownW" placeholder="Known Width cm (e.g. 48)">
                        <input type="number" id="knownH" placeholder="Known Length cm (e.g. 47)">
                        <button onclick="calibrateBoth()" style="margin-top: 4px; padding: 12px;">Calibrate Both Axes</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateMetrics() {
            fetch('/api/dims')
                .then(res => res.json())
                .then(data => {
                    if (data.width_cm !== null) {
                        document.getElementById('valWidth').innerText = data.width_cm + ' cm (' + data.width_in + ' in)';
                    } else {
                        document.getElementById('valWidth').innerText = '—';
                    }

                    if (data.height_cm !== null) {
                        document.getElementById('valHeight').innerText = data.height_cm + ' cm (' + data.height_in + ' in)';
                    } else {
                        document.getElementById('valHeight').innerText = '—';
                    }

                    document.getElementById('valOcr').innerText = data.ocr_text ? data.ocr_text : 'No text detected';
                    document.getElementById('valStatus').innerText = data.status;
                })
                .catch(err => console.error("Metrics error:", err));
        }

        function calibrateBoth() {
            const knownW = document.getElementById('knownW').value;
            const knownH = document.getElementById('knownH').value;
            if (!knownW && !knownH) {
                alert('Please enter at least Known Width or Known Length (e.g. 48 or 47)');
                return;
            }
            fetch(`/api/calibrate?known_width_cm=${knownW || ''}&known_length_cm=${knownH || ''}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('valCalib').innerText = data.new_pixels_per_cm + ' px/cm';
                        alert('✅ Calibration updated: ' + data.new_pixels_per_cm + ' px/cm');
                    } else {
                        alert('❌ Calibration failed: ' + data.error);
                    }
                });
        }

        setInterval(updateMetrics, 500);
        updateMetrics();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, pixels_per_cm=round(pixels_per_cm, 2))

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/dims')
def api_dims():
    return jsonify(streamer.get_dims())

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    global pixels_per_cm
    try:
        known_w_cm = float(request.args.get('known_width_cm', 0) or 0)
        known_h_cm = float(request.args.get('known_length_cm', 0) or 0)

        if known_w_cm <= 0 and known_h_cm <= 0:
            return jsonify({"success": False, "error": "Please enter a valid known width or length (> 0)"}), 400

        dims = streamer.get_dims()
        curr_width_cm = dims.get("width_cm")
        curr_height_cm = dims.get("height_cm")

        if curr_width_cm is None or curr_height_cm is None or curr_width_cm <= 0:
            return jsonify({"success": False, "error": "No valid object detected in frame to calibrate against"}), 422

        ratios = []
        if known_w_cm > 0:
            pixel_w = (curr_width_cm / edge_correction) * pixels_per_cm
            ratio_w = (pixel_w * edge_correction) / known_w_cm
            ratios.append(ratio_w)

        if known_h_cm > 0:
            pixel_h = (curr_height_cm / edge_correction) * pixels_per_cm
            ratio_h = (pixel_h * edge_correction) / known_h_cm
            ratios.append(ratio_h)

        new_ratio = sum(ratios) / len(ratios)
        pixels_per_cm = round(new_ratio, 3)
        print(f"[calibrate] Updated pixels_per_cm ratio to: {pixels_per_cm} px/cm using targets W={known_w_cm}, H={known_h_cm}")
        return jsonify({"success": True, "new_pixels_per_cm": pixels_per_cm})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" 🚀 LIVE YOLO DIMENSION STREAM ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()
