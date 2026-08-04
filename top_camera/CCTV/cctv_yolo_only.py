#!/usr/bin/env python3
"""
============================================================
FILE: cctv_yolo_only.py
PATH: top_camera/global shutter/cctv_yolo_only.py
============================================================
DESCRIPTION:
  Simple, isolated test rig to run ONLY the YOLOv8 model (bestdimension.pt)
  on the CCTV camera stream. Does not perform dimension checking,
  contour geometry, OCR, or calibration. Renders native YOLO bounding boxes.

CREATED: 2026-08-02
============================================================
"""

import sys
import os
import time
import threading
import cv2
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# Resolve paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

# Load YOLO Model
MODEL_PATH = os.path.join(TOP_CAM_DIR, "bestdimension.pt")
if not os.path.exists(MODEL_PATH):
    print(f"[YOLO Only] WARNING: Custom model {MODEL_PATH} not found. Falling back to yolov8n.pt.")
    MODEL_PATH = os.path.join(TOP_CAM_DIR, "yolov8n.pt")

print(f"[YOLO Only] Loading model from {MODEL_PATH}...")
yolo_model = YOLO(MODEL_PATH)

app = Flask(__name__)

class SimpleYOLOStreamer:
    def __init__(self, webcam_index=0):
        self.webcam_index = webcam_index
        self.cap = None
        self.picam2 = None
        self.running = False
        self.latest_jpeg = None
        self.lock = threading.Lock()
        
        self.camera_type = "Offline"
        self.fps_counter = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()
        
        self._init_camera()
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # 1. Try RTSP / Network Stream
        if isinstance(self.webcam_index, str) and any(self.webcam_index.startswith(proto) for proto in ["rtsp://", "rtmp://", "http://", "https://"]):
            try:
                print(f"[YOLO Only] Connecting to RTSP: {self.webcam_index} using FFMPEG...")
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    self.cap = cap
                    self.camera_type = "RTSP CCTV"
                    print("[YOLO Only] RTSP Connected successfully.")
                    return
            except Exception as e:
                print(f"[YOLO Only] RTSP failed: {e}")

        # 2. Try USB Webcam
        try:
            idx = self.webcam_index if isinstance(self.webcam_index, int) else 0
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
            print(f"[YOLO Only] Trying USB cam index {idx}...")
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap = cap
                self.camera_type = f"USB Cam (Index {idx})"
                print("[YOLO Only] USB Cam Connected.")
                return
        except Exception as e:
            print(f"[YOLO Only] USB Cam failed: {e}")

        # 3. Fallback to Picamera2
        try:
            print("[YOLO Only] Trying Picamera2...")
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
            picam.configure(config)
            picam.start()
            self.picam2 = picam
            self.camera_type = "Picamera2 (IMX296)"
            print("[YOLO Only] Picamera2 Connected.")
        except Exception as e:
            print(f"[YOLO Only] Picamera2 failed: {e}")
            self.camera_type = "Offline"

    def _worker_loop(self):
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

            # Run YOLO Model Predict
            results = yolo_model.predict(source=raw_bgr, conf=0.10, save=False, verbose=False)
            
            # Plot native bounding boxes
            if len(results) > 0:
                annotated = results[0].plot()
            else:
                annotated = raw_bgr

            # Calculate FPS
            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            fps_txt = f"FPS: {self.current_fps} | {self.camera_type}"
            cv2.putText(annotated, fps_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated, fps_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', annotated, jpeg_params)
            if ret:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()

            time.sleep(0.001)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

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

DEFAULT_RTSP = "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
streamer = SimpleYOLOStreamer(webcam_index=DEFAULT_RTSP)

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CCTV YOLOv8 Raw Predictions</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --panel: rgba(17, 24, 39, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --primary: #06b6d4;
            --green: #10b981;
            --text: #f3f4f6;
            --dim: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .header {
            width: 100%;
            max-width: 1100px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 16px 24px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            backdrop-filter: blur(12px);
        }
        .header h1 { font-size: 20px; font-weight: 800; }
        .header h1 span { color: var(--primary); }
        .badge {
            color: var(--green);
            font-weight: 700;
            font-size: 13px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 5px 12px;
            border-radius: 20px;
            background: rgba(16, 185, 129, 0.08);
        }
        .container {
            width: 100%;
            max-width: 1100px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-box {
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .video-box img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .info {
            margin-top: 15px;
            font-size: 14px;
            color: var(--dim);
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 YOLOv8 Live <span>Raw Detections</span></h1>
        <div class="badge" id="sourceBadge">● {{ camera_type }} ACTIVE</div>
    </div>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" alt="YOLO Stream">
        </div>
        <div class="info">
            Model: {{ model_path }} | Config: confidence threshold >= 0.10
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, camera_type=streamer.camera_type, model_path=os.path.basename(MODEL_PATH))

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({"camera": streamer.camera_type, "model": os.path.basename(MODEL_PATH)})

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" 🚀 YOLO-ONLY STREAM ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()
