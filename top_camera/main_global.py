"""
main.py — Self-Contained Web Dashboard for Top Camera Rig.
Updated for Raspberry Pi Global Shutter Camera (IMX296) using V4L2.
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

import config_global as cfg
from dimensions import measure_dimensions
from banner import read_banner, read_banner_fast

# ==============================================================================
# Global Shutter Camera Manager
# ==============================================================================
class GlobalShutterCamera:
    """Replaces old ThreadedCamera to explicitly support Global Shutter hardware."""
    def __init__(self, index, width, height):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self._connect()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        
        # V4L2 backend required for libcamerify bridge
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        time.sleep(0.5) # Allow sensor AE/AWB to stabilize

    def read(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None, "Camera not opened"
            ret, frame = self.cap.read()
            if not ret:
                return None, "Failed to grab frame"
            return frame, None

    def change_camera(self, new_idx):
        with self.lock:
            self.index = new_idx
            self._connect()
            return self.cap.isOpened()

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None

# ==============================================================================
# Global State Managers
# ==============================================================================
camera_thread = None
pixels_per_cm = cfg.PIXELS_PER_CM  

last_verdict = {
    "status": "Ready",
    "width_cm": None,
    "height_cm": None,
    "banner_sku": None,
    "banner_text": "",
    "method": "None",
    "timestamp": ""
}

analysis_lock = threading.Lock()
last_processed_image_bytes = None

def get_error_frame(message="CAMERA DISCONNECTED"):
    img = np.zeros((cfg.CAPTURE_H, cfg.CAPTURE_W, 3), dtype=np.uint8)
    for i in range(0, cfg.CAPTURE_H, 40):
        cv2.line(img, (0, i), (cfg.CAPTURE_W, i), (15, 20, 30), 1)
    for i in range(0, cfg.CAPTURE_W, 40):
        cv2.line(img, (i, 0), (i, cfg.CAPTURE_H), (15, 20, 30), 1)
        
    cv2.putText(img, "TOP CAMERA RIG", (int(cfg.CAPTURE_W/2)-150, int(cfg.CAPTURE_H/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (18, 165, 165), 4)
    cv2.putText(img, message.upper(), (int(cfg.CAPTURE_W/2)-300, int(cfg.CAPTURE_H/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

# ==============================================================================
# Web Server Request Handler
# ==============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global pixels_per_cm
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode('utf-8'))
            return

        elif path == "/stream":
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                if camera_thread is None:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame("Rig camera uninitialized") + b'\r\n')
                    time.sleep(0.5)
                    continue

                frame, err = camera_thread.read()
                if frame is None or err:
                    msg = err if err else "Webcam disconnected"
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame(msg) + b'\r\n')
                    time.sleep(0.1)
                    continue

                annotated = frame.copy()
                y0, y1, x0, x1 = cfg.FIXED_CROP
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (180, 180, 180), 1)
                cv2.putText(annotated, "FIXED CROP ROI", (x0 + 10, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                ret, encoded_jpeg = cv2.imencode('.jpg', annotated)
                if not ret: continue

                try:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_jpeg.tobytes() + b'\r\n')
                except (ConnectionResetError, BrokenPipeError):
                    break
                time.sleep(0.03) 
            return

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            current_dim = {"width_cm": None, "height_cm": None, "raw_pixels_w": 0, "raw_pixels_h": 0}
                    
            status_data = {
                "pixels_per_cm": pixels_per_cm,
                "current_camera_index": camera_thread.index if camera_thread else cfg.CAM_INDEX,
                "current_live_dimensions": current_dim,
                "last_verification": last_verdict
            }
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            return

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

        if path == "/api/calibrate":
            if "known_width_cm" not in query:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing query parameter")
                return
            try:
                known_w = float(query["known_width_cm"][0])
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return

            frame, _ = camera_thread.read()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                return

            raw_w_px, _, _ = measure_dimensions(
                img=frame, pixels_per_cm=None,
                edge_correction=cfg.EDGE_CORRECTION_FACTOR,
                min_contour_area=cfg.MIN_CONTOUR_AREA,
                fixed_crop=cfg.FIXED_CROP
            )

            if raw_w_px is None or raw_w_px <= 0:
                self.send_response(422)
                self.end_headers()
                return

            new_ratio = (raw_w_px * cfg.EDGE_CORRECTION_FACTOR) / known_w
            pixels_per_cm = round(new_ratio, 3)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "pixels_per_cm": pixels_per_cm}).encode('utf-8'))
            return

        elif path == "/api/set_camera":
            new_idx = int(query["index"][0])
            success = camera_thread.change_camera(new_idx)
            if success:
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(500)
                self.end_headers()
            return

        elif path == "/api/verify":
            frame, _ = camera_thread.read()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                return

            with analysis_lock:
                w_cm, h_cm, annotated = measure_dimensions(
                    img=frame, pixels_per_cm=pixels_per_cm,
                    edge_correction=cfg.EDGE_CORRECTION_FACTOR,
                    min_contour_area=cfg.MIN_CONTOUR_AREA,
                    fixed_crop=cfg.FIXED_CROP
                )

                y0, y1, x0, x1 = cfg.FIXED_CROP
                crop_bgr = frame[y0:y1, x0:x1]
                sku, angle, text, details = read_banner(crop_bgr)
                
                method = "None"
                if sku:
                    method = "Stage B" if angle is not None else "Stage A"
                else:
                    method = "Not Found"

                cv2.rectangle(annotated, (x0, y0), (x1, y1), (180, 180, 180), 1)
                
                ret, encoded_jpeg = cv2.imencode('.jpg', annotated)
                if ret:
                    global last_processed_image_bytes
                    last_processed_image_bytes = encoded_jpeg.tobytes()

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

# ==============================================================================
# HTML Interface
# ==============================================================================
def get_dashboard_html():
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
            --magenta: #e81b84;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #ef4444;
        }

        body {
            margin: 0; padding: 0;
            background-color: var(--bg-color); color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-image: radial-gradient(circle at 10% 20%, rgba(18, 165, 165, 0.05) 0%, transparent 60%);
        }

        .container { max-width: 1300px; margin: 0 auto; padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 20px; margin-bottom: 25px; }
        header h1 { margin: 0; font-size: 24px; font-weight: 800; }
        header h1 span { color: var(--teal); }

        .cam-selector { display: flex; gap: 10px; background: var(--panel-bg); padding: 8px 15px; border-radius: 12px; }
        .cam-selector select { background: #0f172a; color: var(--text-color); border: 1px solid var(--border-color); padding: 5px 10px; border-radius: 8px; }

        .dashboard-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 24px; }
        .main-column { display: flex; flex-direction: column; gap: 20px; }
        
        .preview-card { background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 16px; overflow: hidden; }
        .preview-card h2 { margin: 0; padding: 16px 20px; font-size: 16px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; }
        .status-badge { background: rgba(16, 185, 129, 0.15); color: var(--green); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
        .stream-container { width: 100%; aspect-ratio: 16/9; background: #000; }
        .stream-container img { width: 100%; height: 100%; object-fit: contain; }

        .card { background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 20px; }
        .card h3 { margin-top: 0; margin-bottom: 15px; font-size: 14px; color: var(--text-dim); }

        .control-bar { display: flex; gap: 15px; }
        .btn { padding: 16px 24px; border-radius: 12px; font-weight: 700; cursor: pointer; border: none; }
        .btn-primary { background: linear-gradient(135deg, var(--teal) 0%, #0d7f7f 100%); color: white; flex-grow: 1; }
        
        .data-panel { display: flex; flex-direction: column; gap: 12px; }
        .data-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
        .data-label { color: var(--text-dim); font-size: 13.5px; }
        .data-value { font-weight: 700; font-size: 15px; }
        .highlight-value { font-size: 28px; color: var(--teal); }
        .sku-value { font-size: 28px; color: var(--magenta); }

        .calib-form { display: flex; gap: 10px; margin-top: 10px; }
        .calib-form input { flex-grow: 1; background: #0f172a; color: white; border: 1px solid var(--border-color); padding: 10px; border-radius: 8px; }
        .calib-form button { background: rgba(255,255,255,0.08); color: white; border: 1px solid var(--border-color); padding: 8px 16px; border-radius: 8px; cursor: pointer; }
        
        .verdict-header { text-align: center; padding: 15px 0; border-radius: 12px; font-size: 20px; font-weight: 800; margin-bottom: 15px; }
        .verdict-success { background: rgba(16, 185, 129, 0.12); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .verdict-alert { background: rgba(239, 68, 68, 0.12); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }
        .verdict-pending { background: rgba(245, 158, 11, 0.12); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Mattress QC <span>Top Camera Rig</span></h1>
            <div class="cam-selector">
                <label>Webcam Index:</label>
                <select id="cameraSelect" onchange="swapCamera(this.value)">
                    <option value="0" selected>0 (Global Shutter)</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                </select>
            </div>
        </header>

        <div class="dashboard-grid">
            <div class="main-column">
                <div class="preview-card">
                    <h2>Live Conveyor Feed <span class="status-badge" id="cameraStatus">Online</span></h2>
                    <div class="stream-container">
                        <img id="webcamStream" src="/stream" alt="Live camera preview">
                    </div>
                </div>
                <div class="control-bar">
                    <button class="btn btn-primary" id="btnVerify" onclick="triggerVerification()">PROCESS FRAME</button>
                    <button class="btn" id="btnResume" onclick="resumeLiveFeed()" style="display: none; background: rgba(255, 255, 255, 0.08); color: white;">RESUME LIVE FEED</button>
                </div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <h3>Processed Results</h3>
                    <div id="verdictBadge" class="verdict-header verdict-pending">READY TO SCAN</div>
                    
                    <div class="data-panel">
                        <div class="data-row"><span class="data-label">Width (cm)</span><span class="data-value highlight-value" id="liveWidth">—</span></div>
                        <div class="data-row"><span class="data-label">Length (cm)</span><span class="data-value highlight-value" id="liveLength">—</span></div>
                        <div class="data-row"><span class="data-label">Banner SKU</span><span class="data-value sku-value" id="outcomeSku">—</span></div>
                        <div class="data-row"><span class="data-label">Method</span><span class="data-value" id="outcomeMethod">—</span></div>
                        <div class="data-row"><span class="data-label">OCR Data</span><span class="data-value" id="outcomeOcr" style="font-family:monospace; color:var(--amber);">—</span></div>
                        <div class="data-row"><span class="data-label">Calibration Factor</span><span class="data-value" id="liveRatio">—</span></div>
                        <div class="data-row"><span class="data-label">Timestamp</span><span class="data-value" id="outcomeTime" style="font-size:12px;">—</span></div>
                    </div>
                    
                    <div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 15px;">
                        <span style="font-size:12px;">Calibrate with Known Object in Frame:</span>
                        <div class="calib-form">
                            <input type="number" id="calibWidth" placeholder="Known width (cm)">
                            <button onclick="submitCalibration()">Calibrate</button>
                        </div>
                        <span id="calibFeedback" style="font-size:11px; margin-top:6px;"></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status').then(res => res.json()).then(data => {
                document.getElementById('cameraSelect').value = data.current_camera_index;
                document.getElementById('liveRatio').innerText = data.pixels_per_cm ? data.pixels_per_cm.toFixed(2) + " px/cm" : "Uncalibrated";
                
                const verify = data.last_verification;
                if (verify.timestamp) {
                    document.getElementById('outcomeSku').innerText = verify.banner_sku;
                    document.getElementById('outcomeMethod').innerText = verify.method;
                    document.getElementById('outcomeOcr').innerText = verify.banner_text || "No text matches";
                    document.getElementById('outcomeTime').innerText = verify.timestamp;
                    document.getElementById('liveWidth').innerText = verify.width_cm ? verify.width_cm.toFixed(1) + " cm" : "—";
                    document.getElementById('liveLength').innerText = verify.height_cm ? verify.height_cm.toFixed(1) + " cm" : "—";

                    const badge = document.getElementById('verdictBadge');
                    if (verify.status === "Success") {
                        badge.innerText = "VERIFIED PASS";
                        badge.className = "verdict-header verdict-success";
                    } else {
                        badge.innerText = "QC ALERT: MISSING/MISMATCH";
                        badge.className = "verdict-header verdict-alert";
                    }
                }
            });
        }

        function submitCalibration() {
            const val = document.getElementById('calibWidth').value;
            fetch(`/api/calibrate?known_width_cm=${val}`, { method: 'POST' }).then(() => updateStatus());
        }

        function swapCamera(idx) {
            fetch(`/api/set_camera?index=${idx}`, { method: 'POST' }).then(() => {
                document.getElementById('webcamStream').src = '/stream?t=' + Date.now();
                document.getElementById('btnResume').style.display = 'none';
            });
        }

        function triggerVerification() {
            document.getElementById('btnVerify').disabled = true;
            document.getElementById('verdictBadge').innerText = "PROCESSING...";
            fetch('/api/verify', { method: 'POST' }).then(() => {
                document.getElementById('btnVerify').disabled = false;
                document.getElementById('webcamStream').src = '/api/last_processed.jpg?t=' + Date.now();
                document.getElementById('btnResume').style.display = 'inline-block';
                updateStatus();
            });
        }

        function resumeLiveFeed() {
            document.getElementById('webcamStream').src = '/stream';
            document.getElementById('btnResume').style.display = 'none';
        }

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
    parser.add_argument("--index", type=int, default=cfg.CAM_INDEX)
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    print("======================================================================")
    print("MATTRESS TOP CAMERA QUALITY CONTROL SYSTEM (Global Shutter V4L2)")
    print("======================================================================")
    
    try:
        # Initialize directly with GlobalShutterCamera
        camera_thread = GlobalShutterCamera(index=args.index, width=cfg.CAPTURE_W, height=cfg.CAPTURE_H)
    except Exception as e:
        print(f"FATAL: Could not initialize camera: {str(e)}")
        sys.exit(1)

    try:
        httpd = ThreadingHTTPServer(('', args.port), DashboardHandler)
        print(f"Dashboard launched at http://localhost:{args.port}")
        print("Important: Ensure you started this via 'libcamerify python main.py'")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if camera_thread:
            camera_thread.release()

if __name__ == "__main__":
    main()