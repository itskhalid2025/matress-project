"""
main.py — Integrated Web Dashboard for Side Camera Rig.

Serves a premium Glassmorphic Dark-Mode UI on http://localhost:8003.
Integrated Subsystems:
  1. Live MJPEG Video Stream (webcam / Basler side camera).
  2. Texture Identification (Classical LBP/GLCM/FFT signatures + TFLite MobileNet classifier).
  3. QR Code Decoding (pyzbar with multi-stage retry ladder).
  4. Side Label OCR Reading (pytesseract + label_reference.json lookup).
  5. Multi-Way Reconciliation (Policy check across Texture, QR, and Label).
  6. On-Demand Computation: Process button triggers analysis on current frame.
  7. Results Panel: Renders all outputs and verdict on the right panel.
"""

import os
import sys
import time
import json
import argparse
import threading
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# Ensure side_camera directory is on Python path for relative imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import config as cfg
from camera import ThreadedCamera
from claim import QRReader, LabelOCRReader
from label_reference import get_label, format_label

# Import Classical Texture Pipeline
from reference import ReferenceStore
from pipeline import IdentificationPipeline, PipelineStatus

# Import Reconcile policy
from reconcile import reconcile

# Import TFLite interpreter for deep learning texture model
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        tflite = None

# ==============================================================================
# Global State & Subsystem Initialization
# ==============================================================================
camera_thread = None
texture_pipeline = None
tflite_interpreter = None
tflite_input_details = None
tflite_output_details = None

TFLITE_CLASSES = [
    "Dual harmony", "Gravite", "Maxi plush", "Maxi pro",
    "Memorise", "Ortholex", "Purity plus", "Velvet"
]

analysis_lock = threading.Lock()
last_processed_image_bytes = None

last_verdict_data = {
    "verdict": "READY",
    "verdict_reason": "Press Process Frame to analyze",
    "texture_sku": "—",
    "texture_status": "—",
    "tflite_sku": "—",
    "tflite_confidence": "—",
    "qr_sku": "—",
    "qr_payload": "—",
    "label_sku": "—",
    "label_text": "—",
    "timestamp": "—"
}

qr_reader = QRReader()
label_ocr_reader = LabelOCRReader()

def init_subsystems(db_path="references_db.pkl", tflite_path="Mattress_Texture_Model.tflite"):
    global texture_pipeline, tflite_interpreter, tflite_input_details, tflite_output_details
    
    # 1. Initialize Classical CV Texture Pipeline
    full_db_path = os.path.join(THIS_DIR, db_path)
    if os.path.exists(full_db_path):
        try:
            store = ReferenceStore()
            store.load(full_db_path)
            texture_pipeline = IdentificationPipeline(store)
            print(f"[init] Classical Texture Pipeline loaded from {db_path}")
        except Exception as e:
            print(f"[init WARNING] Could not load ReferenceStore ({e})")
    else:
        print(f"[init WARNING] Signature DB missing at {full_db_path}")

    # 2. Initialize TFLite Deep Learning Texture Model
    full_tflite_path = os.path.join(THIS_DIR, tflite_path)
    if tflite is not None and os.path.exists(full_tflite_path):
        try:
            tflite_interpreter = tflite.Interpreter(model_path=full_tflite_path)
            tflite_interpreter.allocate_tensors()
            tflite_input_details = tflite_interpreter.get_input_details()
            tflite_output_details = tflite_interpreter.get_output_details()
            print(f"[init] TFLite Texture Model loaded from {tflite_path}")
        except Exception as e:
            print(f"[init WARNING] Could not initialize TFLite model ({e})")
            tflite_interpreter = None
    else:
        if tflite is None:
            print("[init WARNING] tflite_runtime/tensorflow not available")
        else:
            print(f"[init WARNING] TFLite model file missing at {full_tflite_path}")


def run_tflite_inference(frame_bgr):
    """Runs TFLite MobileNet texture classification on a frame."""
    if tflite_interpreter is None or tflite_input_details is None:
        return "N/A", 0.0
    
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224)).astype(np.float32)
        normalized = (resized / 127.5) - 1.0
        input_data = np.expand_dims(normalized, axis=0)

        tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
        tflite_interpreter.invoke()
        output = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])[0]

        pred_idx = np.argmax(output)
        confidence = float(output[pred_idx]) * 100.0
        sku_name = TFLITE_CLASSES[pred_idx] if pred_idx < len(TFLITE_CLASSES) else "Unknown"
        return sku_name, round(confidence, 1)
    except Exception as e:
        print(f"[tflite error] {e}")
        return "Error", 0.0


def get_error_frame(message="CAMERA DISCONNECTED"):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for i in range(0, 1080, 40):
        cv2.line(img, (0, i), (1920, i), (15, 20, 30), 1)
    for i in range(0, 1920, 40):
        cv2.line(img, (i, 0), (i, 1080), (15, 20, 30), 1)
        
    cv2.putText(img, "SIDE CAMERA RIG", (760, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (18, 165, 165), 4)
    cv2.putText(img, message.upper(), (600, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
    cv2.putText(img, "Verify camera connection or index parameter", (620, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 130, 140), 2)
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

# ==============================================================================
# HTTP Request Handler & Server
# ==============================================================================
class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode('utf-8'))
            return

        elif path == "/stream":
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                if camera_thread is None:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame("Camera thread uninitialized") + b'\r\n')
                    time.sleep(0.5)
                    continue

                frame, err = camera_thread.read()
                if frame is None or err:
                    msg = err if err else "Camera disconnected"
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_frame(msg) + b'\r\n')
                    time.sleep(0.1)
                    continue

                annotated = frame.copy()
                cv2.putText(annotated, "SIDE CAMERA LIVE FEED", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)

                ret, encoded = cv2.imencode('.jpg', annotated)
                if not ret:
                    continue

                try:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
                except (ConnectionResetError, BrokenPipeError):
                    break
                time.sleep(0.03)
            return

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = {
                "current_camera_index": camera_thread.index if camera_thread else cfg.CAM_INDEX,
                "last_verification": last_verdict_data
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
        global last_verdict_data, last_processed_image_bytes
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/set_camera":
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
                self.wfile.write(b"Invalid index")
                return

            if camera_thread is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Camera offline")
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

        elif path == "/api/process":
            if camera_thread is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Camera thread uninitialized")
                return

            frame, err = camera_thread.read()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Failed to capture frame")
                return

            print("[process] Running full inspection on side camera frame...")
            with analysis_lock:
                annotated = frame.copy()
                
                # 1. Classical Texture Identification
                texture_sku = "MISSING"
                texture_status_str = "NOT_EVALUATED"
                if texture_pipeline is not None:
                    try:
                        p_status, p_id = texture_pipeline.process_frame(frame)
                        texture_status_str = p_status.value if hasattr(p_status, 'value') else str(p_status)
                        if p_status == PipelineStatus.SUCCESS and p_id:
                            texture_sku = p_id.upper()
                        else:
                            texture_sku = f"REJECTED ({texture_status_str})"
                    except Exception as e:
                        texture_sku = f"ERROR ({str(e)})"
                        texture_status_str = "ERROR"

                # 2. TFLite Deep Learning Texture Inference
                tflite_sku, tflite_conf = run_tflite_inference(frame)

                # 3. QR Code Reader (pyzbar)
                qr_result = qr_reader.read(frame)
                qr_sku = "MISSING"
                qr_payload = "No QR detected"
                if qr_result:
                    qr_sku = qr_result.sku.upper() if qr_result.sku else "DECODED_NO_SKU"
                    qr_payload = qr_result.url

                # 4. Side Label OCR Reader (pytesseract)
                label_result = label_ocr_reader.read(frame)
                label_sku = "MISSING"
                label_text = "No label text found"
                if label_result:
                    label_sku = label_result.sku.upper() if label_result.sku else "OCR_READ"
                    label_text = label_result.raw_text

                # 5. Multi-Way Reconciliation Policy
                verdict_obj = reconcile(
                    texture_raw=texture_sku if not texture_sku.startswith("REJECTED") and not texture_sku.startswith("ERROR") else None,
                    banner_raw=None,  # Side camera has no top banner
                    qr_payload=qr_result
                )
                
                verdict_name = verdict_obj.verdict.name if hasattr(verdict_obj.verdict, 'name') else str(verdict_obj.verdict)

                # Annotate processed frame overlay
                cv2.putText(annotated, f"VERDICT: {verdict_name}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if verdict_name == "PASS" else (0, 0, 255), 3)
                cv2.putText(annotated, f"Texture: {texture_sku} | TFLite: {tflite_sku} ({tflite_conf}%)", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"QR SKU: {qr_sku} | Label SKU: {label_sku}", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                ret, encoded = cv2.imencode('.jpg', annotated)
                if ret:
                    last_processed_image_bytes = encoded.tobytes()

                last_verdict_data = {
                    "verdict": verdict_name,
                    "verdict_reason": f"Texture: {texture_sku}, QR: {qr_sku}, Label: {label_sku}",
                    "texture_sku": texture_sku,
                    "texture_status": texture_status_str,
                    "tflite_sku": tflite_sku,
                    "tflite_confidence": f"{tflite_conf}%" if tflite_conf > 0 else "N/A",
                    "qr_sku": qr_sku,
                    "qr_payload": qr_payload,
                    "label_sku": label_sku,
                    "label_text": label_text,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(last_verdict_data).encode('utf-8'))
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")
            return


# ==============================================================================
# Glassmorphic UI Dashboard HTML
# ==============================================================================
def get_dashboard_html():
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Mattress QC Rig — Side Camera Dashboard</title>
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
        }

        .cam-selector select {
            background: #0f172a;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 5px 10px;
            border-radius: 8px;
            outline: none;
            font-weight: 600;
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

        .btn {
            padding: 16px 24px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 16px;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
            outline: none;
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

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(15px);
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        }

        .card h3 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 14px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .data-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .data-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
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
            font-size: 14.5px;
        }

        .sku-value {
            font-size: 24px;
            font-weight: 850;
            color: var(--magenta);
        }

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
            <h1>Mattress QC <span>Side Camera Rig</span></h1>
            <div class="cam-selector">
                <label for="cameraSelect">Camera Index:</label>
                <select id="cameraSelect" onchange="swapCamera(this.value)">
                    <option value="0" selected>0 (Default)</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                </select>
            </div>
        </header>

        <div class="dashboard-grid">
            <div class="main-column">
                <div class="preview-card">
                    <h2>Live Side View Feed <span class="status-badge" id="cameraStatus">Online</span></h2>
                    <div class="stream-container">
                        <img id="webcamStream" src="/stream" alt="Live side camera preview">
                    </div>
                </div>

                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-primary" id="btnVerify" onclick="triggerProcess()">PROCESS FRAME</button>
                    <button class="btn" id="btnResume" onclick="resumeLiveFeed()" style="display: none; background: rgba(255, 255, 255, 0.08); border: 1px solid var(--border-color); color: var(--text-color);">RESUME LIVE FEED</button>
                </div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <h3>Inspection Results</h3>
                    <div id="verdictBadge" class="verdict-header verdict-pending">READY TO SCAN</div>

                    <div class="data-panel">
                        <div class="data-row">
                            <span class="data-label">Texture SKU (Classical)</span>
                            <span class="data-value sku-value" id="textureSku">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">TFLite Deep Learning SKU</span>
                            <span class="data-value" id="tfliteSku" style="color:var(--teal);">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">QR Code Decoded SKU</span>
                            <span class="data-value" id="qrSku">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">QR Payload Text</span>
                            <span class="data-value" id="qrPayload" style="font-family:monospace; font-size:12px; color:var(--amber);">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Side Label OCR SKU</span>
                            <span class="data-value" id="labelSku">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Label OCR Raw Text</span>
                            <span class="data-value" id="labelText" style="font-family:monospace; font-size:12px; color:var(--text-dim);">—</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Timestamp</span>
                            <span class="data-value" id="outcomeTime" style="font-size:12px; color:var(--text-dim);">—</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    const verify = data.last_verification;
                    if (verify.timestamp && verify.timestamp !== "—") {
                        document.getElementById('textureSku').innerText = verify.texture_sku;
                        document.getElementById('tfliteSku').innerText = verify.tflite_sku + " (" + verify.tflite_confidence + ")";
                        document.getElementById('qrSku').innerText = verify.qr_sku;
                        document.getElementById('qrPayload').innerText = verify.qr_payload;
                        document.getElementById('labelSku').innerText = verify.label_sku;
                        document.getElementById('labelText').innerText = verify.label_text;
                        document.getElementById('outcomeTime').innerText = verify.timestamp;

                        const badge = document.getElementById('verdictBadge');
                        if (verify.verdict === "PASS") {
                            badge.innerText = "VERIFIED PASS";
                            badge.className = "verdict-header verdict-success";
                        } else if (verify.verdict === "MISMATCH" || verify.verdict === "DOUBLE_MISMATCH" || verify.verdict === "CONFLICT") {
                            badge.innerText = "QC ALERT: MISMATCH";
                            badge.className = "verdict-header verdict-alert";
                        } else {
                            badge.innerText = verify.verdict;
                            badge.className = "verdict-header verdict-pending";
                        }
                    }
                });
        }

        function triggerProcess() {
            const btn = document.getElementById('btnVerify');
            const badge = document.getElementById('verdictBadge');
            
            btn.disabled = true;
            btn.innerText = "PROCESSING FRAME...";
            badge.innerText = "PROCESSING...";
            badge.className = "verdict-header verdict-pending";

            fetch('/api/process', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    btn.disabled = false;
                    btn.innerText = "PROCESS FRAME";
                    
                    const img = document.getElementById('webcamStream');
                    img.src = '/api/last_processed.jpg?t=' + new Date().getTime();
                    document.getElementById('btnResume').style.display = 'inline-block';
                    
                    updateStatus();
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.innerText = "PROCESS FRAME";
                    alert("Processing failed: " + err);
                });
        }

        function resumeLiveFeed() {
            const img = document.getElementById('webcamStream');
            img.src = '/stream';
            document.getElementById('btnResume').style.display = 'none';
        }

        function swapCamera(idx) {
            fetch(`/api/set_camera?index=${idx}`, { method: 'POST' })
                .then(res => {
                    if (res.ok) {
                        const img = document.getElementById('webcamStream');
                        img.src = '/stream?t=' + new Date().getTime();
                        document.getElementById('btnResume').style.display = 'none';
                    }
                });
        }

        updateStatus();
        setInterval(updateStatus, 1500);
    </script>
</body>
</html>
"""

# ==============================================================================
# Entry Point
# ==============================================================================
def main():
    global camera_thread

    parser = argparse.ArgumentParser(description="Mattress Side Camera QC Rig Server")
    parser.add_argument("--index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--port", type=int, default=8003, help="HTTP server port (default: 8003)")
    parser.add_argument("--width", type=int, default=1920, help="Capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Capture height (default: 1080)")
    args = parser.parse_args()

    print("======================================================================")
    print("MATTRESS SIDE CAMERA QUALITY CONTROL RIG")
    print("======================================================================")
    
    init_subsystems()

    print(f"Initializing camera thread on index {args.index} at {args.width}x{args.height}...")
    try:
        camera_thread = ThreadedCamera(index=args.index, width=args.width, height=args.height)
    except Exception as e:
        print(f"FATAL: Could not initialize camera thread: {str(e)}")
        sys.exit(1)

    server_address = ('', args.port)
    try:
        httpd = ThreadingHTTPServer(server_address, DashboardHandler)
        print(f"Side Camera Dashboard running at http://localhost:{args.port}")
        print("Press Ctrl+C to terminate.")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down side camera rig...")
    except Exception as e:
        print(f"FATAL: Server error: {str(e)}")
    finally:
        if camera_thread:
            camera_thread.release()

if __name__ == "__main__":
    main()
