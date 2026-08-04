#!/usr/bin/env python3
"""
============================================================
FILE: cctv_dimensions.py
PATH: top_camera/global shutter/cctv_dimensions.py
============================================================
DESCRIPTION:
  Upgraded Live Web Streaming & Measurement Rig powered by 
  CCTV RTSP camera stream and YOLO target tracking.
  Features:
  - RTSP CCTV stream integration (low latency + FFMPEG backend)
  - Auto-fallback to local USB camera or Picamera2 (IMX296)
  - Tight YOLO target isolation (prevents background table/wall bleed)
  - Rotated minimum area bounding box + convex hull geometry
  - Dual Centimeters & Inches real-time display
  - Clean Mattress Sash & Label OCR (extracts MAXI PLUSH cleanly)
  - Dual-Axis Live Web Calibration (Known Width & Known Length)
  - Beautiful, Premium Dynamic Web UI Dashboard showing active camera source

CREATED: 2026-08-01 | 11:20 PM
============================================================
"""

import sys
import os
import time
import threading
import re
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO

# Import the proven banner OCR engine from the top_camera module
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if TOP_CAM_DIR not in sys.path:
    sys.path.insert(0, TOP_CAM_DIR)

try:
    from banner import read_banner_fast
    import config as _cfg
    # Lower saturation threshold so darker/camera-compressed sashes are detected
    _cfg.SASH_S_THRESH = 60
    _cfg.SASH_V_THRESH = 60
    BANNER_OCR_AVAILABLE = True
    print("[cctv_stream] banner.py OCR engine loaded (SASH thresholds relaxed to 60)")
except Exception as _be:
    BANNER_OCR_AVAILABLE = False
    print(f"[cctv_stream] banner.py not available: {_be}")

app = Flask(__name__)

# Global Calibration Parameters
pixels_per_cm = 10.0  # Default pixels per cm
edge_correction = 1.0

# Load YOLO Model
MODEL_PATH = os.path.join(TOP_CAM_DIR, "yolov8n.pt")

print(f"[cctv_stream] Loading YOLOv8 model from {MODEL_PATH}...")
yolo_model = YOLO(MODEL_PATH)

def process_frame_tight_geometry(img, current_px_per_cm):
    """
    Isolates the target item using conveyor region crop [50:1030, 200:1300] & YOLO,
    restricts contour search strictly inside the item to eliminate background bleed,
    and returns (width_cm, height_cm, annotated_image, crop_box).
    """
    if img is None or img.size == 0:
        return None, None, img, None

    annotated = img.copy()
    h_img, w_img = img.shape[:2]

    # Crop to conveyor belt region to exclude top ceiling/wall (y<50) and side borders (x>1300)
    x0_roi, y0_roi, x1_roi, y1_roi = 200, 50, min(1300, w_img), min(1030, h_img)
    roi_img = img[y0_roi:y1_roi, x0_roi:x1_roi]

    if roi_img.size == 0:
        return None, None, annotated, None

    # STAGE 1: Try YOLO Target Isolation
    results = yolo_model.predict(source=roi_img, save=False, conf=0.10, verbose=False)
    valid_box = None
    max_area = 0

    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:  # Skip human operator
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                valid_box = box

    if valid_box is not None:
        x1_b, y1_b, x2_b, y2_b = valid_box.xyxy[0].cpu().numpy()
        pad = 15
        x1_c = max(0, int(x1_b) - pad) + x0_roi
        y1_c = max(0, int(y1_b) - pad) + y0_roi
        x2_c = min(w_img, int(x2_b) + pad) + x0_roi
        y2_c = min(h_img, int(y2_b) + pad) + y0_roi
    else:
        # Fallback to conveyor ROI region
        x1_c, y1_c, x2_c, y2_c = x0_roi, y0_roi, x1_roi, y1_roi

    crop_item = img[y1_c:y2_c, x1_c:x2_c]
    if crop_item.size == 0:
        return None, None, annotated, None

    # STAGE 2: Perform High-Precision Contour Geometry
    crop_gray = cv2.cvtColor(crop_item, cv2.COLOR_BGR2GRAY)
    crop_blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)

    edges = cv2.Canny(crop_blur, 30, 120)
    _, thresh = cv2.threshold(crop_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(edges, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, annotated, (x1_c, y1_c, x2_c, y2_c)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_c = None
    item_crop_area = crop_item.shape[0] * crop_item.shape[1]

    for c in contours:
        c_area = cv2.contourArea(c)
        if 2000 < c_area < (item_crop_area * 0.90):
            best_c = c
            break

    if best_c is None:
        best_c = contours[0]

    hull = cv2.convexHull(best_c)
    rect = cv2.minAreaRect(hull)
    (cx, cy), (w, h), angle = rect

    if w < 5 or h < 5:
        return None, None, annotated, (x1_c, y1_c, x2_c, y2_c)

    # Offset coordinates back to full image system
    cx_orig = cx + x1_c
    cy_orig = cy + y1_c

    hull_offset = hull + np.array([x1_c, y1_c])
    cv2.drawContours(annotated, [hull_offset], -1, (0, 0, 255), 2)

    box_pts = cv2.boxPoints(rect)
    box_offset = np.int32(box_pts + np.array([x1_c, y1_c]))
    cv2.drawContours(annotated, [box_offset], 0, (0, 255, 0), 3)
    cv2.circle(annotated, (int(cx_orig), int(cy_orig)), 6, (0, 255, 255), -1)

    # Convert pixels to cm & inches
    raw_w = w / current_px_per_cm
    raw_h = h / current_px_per_cm

    nW_cm = round(raw_w * edge_correction, 1)
    nH_cm = round(raw_h * edge_correction, 1)

    nW_in = round(nW_cm / 2.54, 1)
    nH_in = round(nH_cm / 2.54, 1)

    # Draw Dimension Text Tags on image
    tag_w = f"W: {nW_cm} cm ({nW_in} in)"
    tag_l = f"L: {nH_cm} cm ({nH_in} in)"

    cv2.putText(annotated, tag_w, (int(cx_orig) - 120, int(cy_orig) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, tag_l, (int(cx_orig) - 120, int(cy_orig) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)

    return nW_cm, nH_cm, annotated, (x1_c, y1_c, x2_c, y2_c)


# ---------------------------------------------------------------------------
# Background OCR Thread — runs read_banner_fast() completely decoupled from
# the video capture loop so FPS is never impacted by OCR latency.
# ---------------------------------------------------------------------------
_ocr_latest_frame = None        # latest BGR frame shared to OCR thread
_ocr_frame_lock   = threading.Lock()
_ocr_result       = ""          # most recent OCR text result
_ocr_result_lock  = threading.Lock()

def _ocr_worker():
    """Dedicated background thread: reads frames and calls read_banner_fast."""
    global _ocr_result, _ocr_latest_frame
    while True:
        frame_to_process = None
        with _ocr_frame_lock:
            if _ocr_latest_frame is not None:
                frame_to_process = _ocr_latest_frame.copy()
                _ocr_latest_frame = None   # Clear so stale frame isn't reprocessed
        if frame_to_process is None:
            time.sleep(0.1)
            continue
        try:
            if BANNER_OCR_AVAILABLE:
                sku, raw_text, _ = read_banner_fast(frame_to_process)
                if sku:
                    label = sku.replace("_", " ").upper()
                elif raw_text and len(raw_text.strip()) > 2:
                    words = re.findall(r"[A-Za-z0-9]{3,}", raw_text)
                    label = " ".join(words).upper()[:40]
                else:
                    label = ""
                if label:
                    with _ocr_result_lock:
                        _ocr_result = label
                    print(f"[ocr_worker] Detected: {label}")
        except Exception as exc:
            print(f"[ocr_worker] {exc}")
        # Short pause then check for next frame
        time.sleep(0.5)

# Start OCR thread once at module load
_ocr_thread = threading.Thread(target=_ocr_worker, daemon=True)
_ocr_thread.start()


class StreamerServer:
    def __init__(self, width=1920, height=1080, fps=30, webcam_index=8):
        self.width = width
        self.height = height
        self.fps = fps
        self.webcam_index = webcam_index

        self.picam2 = None
        self.cap = None
        self.camera_type = "Offline"

        self.latest_jpeg = None
        self.latest_dims = {
            "width_cm": None, "width_in": None,
            "height_cm": None, "height_in": None,
            "ocr_text": "", "status": "Initializing",
            "camera_type": "Offline"
        }
        self.latest_ocr = ""
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0

        self.fps_counter = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()

        self._init_camera()

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # ── 1. Try RTSP / Network Stream first if input is a string ──────────
        if isinstance(self.webcam_index, str) and any(self.webcam_index.startswith(proto) for proto in ["rtsp://", "rtmp://", "http://", "https://"]):
            try:
                print(f"[cctv_stream] Connecting to RTSP Stream: {self.webcam_index} using FFMPEG backend...")
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    self.cap = cap
                    self.camera_type = "RTSP CCTV"
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cctv_stream] Connected to RTSP camera successfully at {actual_w}x{actual_h}")
                    return
                else:
                    print("[cctv_stream] Failed to open RTSP stream source.")
            except Exception as e:
                print(f"[cctv_stream] RTSP stream connection failed: {e}")

        # ── 2. Try USB Webcam (Platform aware backend selection) ─────────────
        try:
            # Determine indices
            idx = self.webcam_index if isinstance(self.webcam_index, int) else 0
            
            # Select backend based on OS
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
            print(f"[cctv_stream] Attempting USB webcam index {idx} with backend {backend}...")
            cap = cv2.VideoCapture(idx, backend)
            
            if not cap.isOpened():
                # Fallback to default auto-selected backend
                print(f"[cctv_stream] Backend {backend} failed, trying default cv2 VideoCapture backend...")
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap = cap
                self.camera_type = f"USB Cam (Index {idx})"
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[cctv_stream] Connected to USB webcam successfully at {actual_w}x{actual_h}")
                return
        except Exception as e:
            print(f"[cctv_stream] USB webcam initialization failed: {e}")

        # ── 3. Fallback: Picamera2 (IMX296 global shutter) ──────────────────
        try:
            print("[cctv_stream] Attempting Picamera2 fallback...")
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            self.picam2 = picam
            self.camera_type = "Picamera2 (IMX296)"
            print("[cctv_stream] Fallback: connected via Picamera2 (IMX296)")
        except Exception as e:
            print(f"[cctv_stream] Picamera2 fallback failed: {e}")
            self.camera_type = "Offline"

    def _worker_loop(self):
        global pixels_per_cm
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

            self.frame_count += 1

            # Run tight geometry process
            w_cm, h_cm, annotated, target_box = process_frame_tight_geometry(raw_bgr, pixels_per_cm)

            # Feed latest frame to background OCR thread every 5 frames
            if self.frame_count % 5 == 0:
                with _ocr_frame_lock:
                    global _ocr_latest_frame
                    _ocr_latest_frame = raw_bgr.copy()

            # Pick up whatever the background OCR thread last produced
            with _ocr_result_lock:
                self.latest_ocr = _ocr_result

            # Draw OCR text tag on video feed
            if self.latest_ocr:
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

            # Update FPS
            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            fps_txt = f"FPS: {self.current_fps}"
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', annotated, jpeg_params)
            if ret:
                w_in = round(w_cm / 2.54, 1) if w_cm is not None else None
                h_in = round(h_cm / 2.54, 1) if h_cm is not None else None
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_dims = {
                        "width_cm": w_cm,
                        "width_in": w_in,
                        "height_cm": h_cm,
                        "height_in": h_in,
                        "ocr_text": self.latest_ocr,
                        "status": "Target Detected" if w_cm is not None else "Searching Target...",
                        "camera_type": self.camera_type
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


# Default to CCTV RTSP camera stream configuration
DEFAULT_RTSP = "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
streamer = StreamerServer(width=1920, height=1080, fps=30, webcam_index=DEFAULT_RTSP)


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
# Web UI Dashboard
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CCTV Mattress Dimension Stream</title>
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
        .camera-badge { color: var(--green); font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 12px; border-radius: 20px; background: rgba(16, 185, 129, 0.08); display: flex; align-items: center; gap: 6px; }
        .camera-badge.offline { color: var(--magenta); border-color: rgba(232, 27, 132, 0.3); background: rgba(232, 27, 132, 0.08); }

        .layout { width: 100%; max-width: 1200px; display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }
        
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; backdrop-filter: blur(12px); }
        .video-box { width: 100%; aspect-ratio: 16/10; background: #000; border-radius: 14px; overflow: hidden; display: flex; justify-content: center; align-items: center; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .video-box img { width: 100%; height: 100%; object-fit: contain; }

        .stat-group { display: flex; flex-direction: column; gap: 14px; }
        .stat-card { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); padding: 14px; border-radius: 14px; }
        .stat-label { font-size: 11px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
        .stat-val { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--primary); }
        .stat-val.magenta { color: var(--magenta); }
        .stat-val.amber { color: var(--amber); font-size: 18px; font-weight: 700; word-break: break-word; }
        
        .calib-box { margin-top: 10px; padding-top: 15px; border-top: 1px solid var(--border); }
        .calib-input { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
        .calib-input input { background: #0f172a; border: 1px solid var(--border); color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 14px; }
        .calib-input button { background: var(--primary); color: #fff; border: none; padding: 12px; border-radius: 8px; font-weight: 700; cursor: pointer; }
        .calib-input button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📐 Precision CCTV <span>Dimension Stream</span></h1>
        <div class="camera-badge" id="cameraBadge">● {{ camera_type }} ACTIVE</div>
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
                    <div class="stat-label">Detected Brand / OCR</div>
                    <div class="stat-val amber" id="valOcr">Reading...</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Status</div>
                    <div class="stat-val" id="valStatus" style="font-size: 16px; color: var(--green);">Searching Target...</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Calibration Factor</div>
                    <div class="stat-val" id="valCalib" style="font-size: 18px;">{{ pixels_per_cm }} px/cm</div>
                </div>

                <div class="calib-box">
                    <div class="stat-label">Dual-Axis Calibration</div>
                    <div style="font-size: 12px; color: var(--dim); margin-bottom: 6px;">Enter real physical size of target item:</div>
                    <div class="calib-input">
                        <input type="number" id="knownW" placeholder="Known Width cm (e.g. 48)">
                        <input type="number" id="knownH" placeholder="Known Length cm (e.g. 47)">
                        <button onclick="calibrateBoth()">Calibrate Both Axes</button>
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
                    
                    const badge = document.getElementById('cameraBadge');
                    const camType = data.camera_type ? data.camera_type : 'Offline';
                    badge.innerText = '● ' + camType + ' ACTIVE';
                    if (camType.toLowerCase() === 'offline') {
                        badge.classList.add('offline');
                    } else {
                        badge.classList.remove('offline');
                    }
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
    return render_template_string(HTML_TEMPLATE, pixels_per_cm=round(pixels_per_cm, 2), camera_type=streamer.camera_type)

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
            return jsonify({"success": False, "error": "No valid target detected in frame to calibrate against"}), 422

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
        print(f"[calibrate] Updated pixels_per_cm ratio to: {pixels_per_cm} px/cm for W={known_w_cm}, H={known_h_cm}")
        return jsonify({"success": True, "new_pixels_per_cm": pixels_per_cm})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" 🚀 PRECISION CCTV DIMENSION ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()
