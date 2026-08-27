"""
flask_dashboard.py — Premium Flask Dashboard for Side Camera QR/OCR Rig.

Runs at http://localhost:8003
Features:
  - Live MJPEG webcam stream (index 0)
  - One-click QR code decoding + URL scrape
  - Tesseract Label OCR reading
  - SKU reconciliation verdict
  - Premium dark-mode glassmorphic UI
"""

import os
import sys
import time
import json
import threading
import base64

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import config as cfg
from claim2 import QRReader, LabelOCRReader
from texture_classifier import TextureClassifier
from reconcile import reconcile

# ==============================================================================
# Camera Setup — Index 0 direct
# ==============================================================================
_cap = None
_cap_lock = threading.Lock()


def get_camera():
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
        args = (0,) if backend is None else (0, backend)
        cap = cv2.VideoCapture(*args)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, test = cap.read()
            if ret and test is not None:
                _cap = cap
                return _cap
            cap.release()
    return None


def read_frame():
    with _cap_lock:
        cap = get_camera()
        if cap is None:
            return None
        cap.grab()
        ret, frame = cap.retrieve()
        if ret and frame is not None:
            return frame
        return None


def make_error_frame(msg="NO CAMERA SIGNAL"):
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (854, 480), (10, 12, 20), -1)
    for i in range(0, 480, 30):
        cv2.line(img, (0, i), (854, i), (18, 22, 35), 1)
    cv2.putText(img, "SIDE CAM RIG", (280, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 180, 180), 2)
    cv2.putText(img, msg, (200, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)
    cv2.putText(img, "Check USB connection / camera index", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 90, 110), 1)
    _, enc = cv2.imencode('.jpg', img)
    return enc.tobytes()


# ==============================================================================
# App & Subsystems
# ==============================================================================
app = Flask(__name__)
qr_reader = QRReader()
ocr_reader = LabelOCRReader()
texture_classifier = TextureClassifier()

state_lock = threading.Lock()
state = {
    "status": "READY",
    "verdict": "—",
    "verdict_class": "ready",
    "qr_sku": "—",
    "qr_url": "—",
    "qr_batch": "—",
    "ocr_sku": "—",
    "ocr_text": "—",
    "processing": False,
    "timestamp": "—",
    "snapshot_b64": None,
}

# ==============================================================================
# Routes
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def gen_stream():
    while True:
        frame = read_frame()
        if frame is None:
            jpg = make_error_frame()
        else:
            # Overlay live badge
            cv2.rectangle(frame, (0, 0), (220, 36), (10, 10, 20), -1)
            cv2.putText(frame, "LIVE  |  SIDE CAM  |  IDX 0", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 160), 1)
            _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg = enc.tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        time.sleep(0.033)


@app.route("/stream")
def stream():
    return Response(gen_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/process", methods=["POST"])
def process():
    with state_lock:
        if state["processing"]:
            return jsonify({"error": "Already processing"}), 429
        state["processing"] = True

    try:
        frame = read_frame()
        if frame is None:
            with state_lock:
                state.update({"status": "ERROR", "verdict": "No camera frame", "verdict_class": "fail", "processing": False})
            return jsonify(state)

        # Snapshot for panel
        _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        snap_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

        # QR
        qr_claim = None
        try:
            qr_claim, _ = qr_reader.read(frame, return_stage=True)
        except Exception:
            pass

        # OCR
        ocr_claim = None
        try:
            ocr_claim = ocr_reader.read(frame)
        except Exception:
            pass

        # Texture Classification
        texture_res = None
        try:
            texture_res = texture_classifier.predict(frame)
        except Exception:
            pass

        # Extract SKUs from all 3 claims
        texture_sku = texture_res["sku"] if texture_res else None
        qr_sku = qr_claim.sku if qr_claim else None
        ocr_sku = ocr_claim.sku if ocr_claim else None
        qr_raw_url = getattr(qr_claim, "url", None) if qr_claim else None

        # Full 3-way Reconcile: reconcile(texture_raw, banner_raw, qr_payload)
        result = reconcile(texture_sku, ocr_sku, qr_raw_url)

        verdict_class = "pass" if result.verdict.name == "PASS" else ("warn" if result.verdict.name == "SKIP" else "fail")

        ts = time.strftime("%H:%M:%S")

        with state_lock:
            state.update({
                "status": "DONE",
                "verdict": result.verdict.name,
                "verdict_class": verdict_class,
                "verdict_reason": getattr(result, "detail", "—"),
                "texture_sku": texture_sku or "—",
                "texture_conf": f"{texture_res['confidence']}%" if texture_res else "—",
                "qr_sku": qr_sku or "—",
                "qr_url": qr_raw_url or "—",
                "qr_batch": getattr(qr_claim, "batch_no", "—") or "—",
                "ocr_sku": ocr_sku or "—",
                "ocr_text": (ocr_claim.matched_text[:80] if ocr_claim else "—"),
                "processing": False,
                "timestamp": ts,
                "snapshot_b64": snap_b64,
            })

        with state_lock:
            return jsonify({k: v for k, v in state.items() if k != "snapshot_b64"} | {"snapshot_b64": snap_b64})

    except Exception as e:
        with state_lock:
            state.update({"status": "ERROR", "verdict": str(e), "verdict_class": "fail", "processing": False})
        return jsonify(state)


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({k: v for k, v in state.items() if k != "snapshot_b64"})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        state.update({
            "status": "READY", "verdict": "—", "verdict_class": "ready",
            "qr_sku": "—", "qr_url": "—", "qr_batch": "—",
            "ocr_sku": "—", "ocr_text": "—",
            "processing": False, "timestamp": "—", "snapshot_b64": None
        })
    return jsonify({"ok": True})


# ==============================================================================
# Dashboard HTML
# ==============================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MATTRESSQC — Side Camera Rig</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100vh;width:100vw;overflow:hidden;font-family:'Inter',sans-serif;background:#070b14;color:#e2e8f0}

/* ── Navbar ── */
.nav{height:52px;background:#0d1321;border-bottom:1px solid #1e293b;display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;flex-shrink:0}
.nav-brand{display:flex;align-items:center;gap:.6rem}
.nav-brand h1{font-size:1.1rem;font-weight:900;letter-spacing:.5px;color:#fff}
.badge{background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;font-size:.6rem;font-weight:800;padding:2px 8px;border-radius:4px;letter-spacing:.5px}
.nav-right{display:flex;align-items:center;gap:1rem}
.pill{background:#0f172a;border:1px solid #1e293b;border-radius:20px;padding:4px 14px;font-size:.75rem;font-weight:600;display:flex;align-items:center;gap:6px;color:#94a3b8}
.dot{width:8px;height:8px;border-radius:50%}
.dot-green{background:#22c55e;box-shadow:0 0 8px #22c55e}
.dot-amber{background:#f59e0b;box-shadow:0 0 8px #f59e0b;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Main Grid ── */
.main{display:grid;grid-template-columns:1fr 380px;gap:1rem;padding:1rem;height:calc(100vh - 52px);overflow:hidden}

/* ── Camera Card ── */
.cam-card{background:#0d1321;border:1px solid #1e293b;border-radius:14px;padding:1rem;display:flex;flex-direction:column;gap:.75rem;overflow:hidden}
.card-header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b;padding-bottom:.6rem}
.card-header h2{font-size:1rem;font-weight:700}
.tag{background:#1e293b;color:#64748b;font-size:.65rem;font-weight:700;padding:3px 8px;border-radius:6px;letter-spacing:.4px}

.stream-box{flex:1;min-height:0;background:#000;border-radius:10px;overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative;border:1px solid #1e293b}
.stream-box img{width:100%;height:100%;object-fit:contain}

.action-row{display:grid;grid-template-columns:2fr 1fr;gap:.75rem}
.btn{width:100%;padding:11px 16px;border:none;border-radius:9px;font-weight:700;font-size:.9rem;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:8px;letter-spacing:.3px}
.btn-process{background:linear-gradient(135deg,#10b981,#059669);color:#fff;box-shadow:0 4px 18px rgba(16,185,129,.4)}
.btn-process:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 24px rgba(16,185,129,.55)}
.btn-process:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-reset{background:#1e293b;color:#94a3b8}
.btn-reset:hover{background:#334155}

/* ── Right Panel ── */
.panel{display:flex;flex-direction:column;gap:.75rem;height:100%;overflow-y:auto}

.result-card{background:#0d1321;border:1px solid #1e293b;border-radius:14px;padding:1rem;display:flex;flex-direction:column;gap:.75rem}

/* Verdict Banner */
.verdict-banner{border-radius:10px;padding:.85rem 1rem;text-align:center;transition:all .4s}
.verdict-ready{background:#1e293b;border:1px solid #334155}
.verdict-pass{background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(5,150,105,.08));border:1px solid rgba(16,185,129,.4)}
.verdict-fail{background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(185,28,28,.08));border:1px solid rgba(239,68,68,.4)}
.verdict-warn{background:linear-gradient(135deg,rgba(245,158,11,.18),rgba(180,83,9,.08));border:1px solid rgba(245,158,11,.4)}
.verdict-label{font-size:.65rem;font-weight:700;color:#64748b;letter-spacing:1px;margin-bottom:4px}
.verdict-value{font-size:1.8rem;font-weight:900;letter-spacing:2px}
.verdict-ready .verdict-value{color:#475569}
.verdict-pass .verdict-value{color:#10b981}
.verdict-fail .verdict-value{color:#ef4444}
.verdict-warn .verdict-value{color:#f59e0b}
.verdict-reason{font-size:.75rem;color:#64748b;margin-top:4px}

/* Snapshot */
.snapshot-box{border-radius:10px;overflow:hidden;border:1px solid #1e293b;background:#000;min-height:140px;display:flex;align-items:center;justify-content:center}
.snapshot-box img{width:100%;display:block}
.snap-placeholder{color:#334155;font-size:.8rem;font-weight:600}

/* Data Fields */
.fields{display:flex;flex-direction:column;gap:.45rem}
.field{background:#0a0f1a;border:1px solid #1e293b;border-radius:8px;padding:.55rem .8rem;display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem}
.field-label{font-size:.65rem;font-weight:700;color:#475569;letter-spacing:.5px;white-space:nowrap;padding-top:2px}
.field-val{font-size:.8rem;font-weight:600;color:#e2e8f0;font-family:'JetBrains Mono',monospace;text-align:right;word-break:break-all;max-width:200px}
.field-val.highlight{color:#38bdf8}

.section-title{font-size:.7rem;font-weight:700;color:#475569;letter-spacing:1px;text-transform:uppercase;padding-bottom:4px;border-bottom:1px solid #1e293b}

/* Processing overlay */
.processing-ring{display:none;width:20px;height:20px;border:2px solid rgba(255,255,255,.2);border-top:2px solid #fff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.btn-process.loading .btn-text{display:none}
.btn-process.loading .processing-ring{display:block}
</style>
</head>
<body>

<!-- Navbar -->
<nav class="nav">
  <div class="nav-brand">
    <h1>MATTRESSQC</h1>
    <span class="badge">SIDE CAM</span>
  </div>
  <div class="nav-right">
    <div class="pill" id="cam-pill">
      <span class="dot dot-green" id="cam-dot"></span>
      <span id="cam-label">WEBCAM IDX 0</span>
    </div>
    <div class="pill" id="ts-pill">
      <span>—</span>
    </div>
  </div>
</nav>

<!-- Main -->
<div class="main">

  <!-- Left: Live Camera -->
  <div class="cam-card">
    <div class="card-header">
      <h2>📷 Live Side Camera Feed</h2>
      <span class="tag">USB WEBCAM | INDEX 0</span>
    </div>
    <div class="stream-box">
      <img id="stream" src="/stream" alt="live feed">
    </div>
    <div class="action-row">
      <button class="btn btn-process" id="btn-process" onclick="processFrame()">
        <span class="processing-ring" id="spin"></span>
        <span class="btn-text">⚡ PROCESS FRAME</span>
      </button>
      <button class="btn btn-reset" onclick="resetState()">↺ RESET</button>
    </div>
  </div>

  <!-- Right: Results Panel -->
  <div class="panel">

    <!-- Verdict Card -->
    <div class="result-card">
      <div class="card-header">
        <h2>Verdict</h2>
        <span class="tag" id="ts-tag">—</span>
      </div>
      <div class="verdict-banner verdict-ready" id="verdict-banner">
        <div class="verdict-label">RECONCILIATION RESULT</div>
        <div class="verdict-value" id="verdict-val">—</div>
        <div class="verdict-reason" id="verdict-reason">Press Process Frame to analyze the current camera feed</div>
      </div>

      <!-- Snapshot -->
      <div class="snapshot-box" id="snapshot-box">
        <span class="snap-placeholder">Snapshot appears after processing</span>
      </div>
    </div>

    <!-- Texture Classifier Card -->
    <div class="result-card">
      <div class="card-header">
        <h2>Fabric Texture Data</h2>
        <span class="tag">EFFICIENTNET-B0</span>
      </div>
      <div class="fields">
        <div class="section-title">PyTorch Quilting Classifier</div>
        <div class="field">
          <span class="field-label">SKU</span>
          <span class="field-val highlight" id="tex-sku">—</span>
        </div>
        <div class="field">
          <span class="field-label">CONFIDENCE</span>
          <span class="field-val" id="tex-conf">—</span>
        </div>
      </div>
    </div>

    <!-- QR Data Card -->
    <div class="result-card">
      <div class="card-header">
        <h2>QR Code Data</h2>
        <span class="tag">STEP 1</span>
      </div>
      <div class="fields">
        <div class="section-title">QR Decoded</div>
        <div class="field">
          <span class="field-label">SKU</span>
          <span class="field-val highlight" id="qr-sku">—</span>
        </div>
        <div class="field">
          <span class="field-label">BATCH NO</span>
          <span class="field-val" id="qr-batch">—</span>
        </div>
        <div class="field">
          <span class="field-label">URL</span>
          <span class="field-val" id="qr-url" style="font-size:.68rem">—</span>
        </div>
      </div>
    </div>

    <!-- OCR Data Card -->
    <div class="result-card">
      <div class="card-header">
        <h2>Label OCR Data</h2>
        <span class="tag">STEP 2</span>
      </div>
      <div class="fields">
        <div class="section-title">Tesseract OCR</div>
        <div class="field">
          <span class="field-label">SKU</span>
          <span class="field-val highlight" id="ocr-sku">—</span>
        </div>
        <div class="field" style="align-items:flex-start">
          <span class="field-label">TEXT</span>
          <span class="field-val" id="ocr-text" style="font-size:.7rem;max-width:230px">—</span>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
function setProcessing(on) {
  const btn = document.getElementById('btn-process');
  if (on) {
    btn.disabled = true;
    btn.classList.add('loading');
    document.getElementById('spin').style.display = 'block';
  } else {
    btn.disabled = false;
    btn.classList.remove('loading');
    document.getElementById('spin').style.display = 'none';
  }
}

function applyVerdict(d) {
  const banner = document.getElementById('verdict-banner');
  const val = document.getElementById('verdict-val');
  const reason = document.getElementById('verdict-reason');

  banner.className = 'verdict-banner';
  const cls = d.verdict_class || 'ready';
  banner.classList.add('verdict-' + cls);
  val.textContent = d.verdict || '—';
  reason.textContent = d.verdict_reason || '';
}

function processFrame() {
  setProcessing(true);
  document.getElementById('verdict-val').textContent = 'ANALYZING...';
  document.getElementById('verdict-reason').textContent = 'Classifying texture, decoding QR, running OCR...';

  fetch('/api/process', {method:'POST'})
    .then(r => r.json())
    .then(d => {
      setProcessing(false);
      applyVerdict(d);

      document.getElementById('tex-sku').textContent   = d.texture_sku  || '—';
      document.getElementById('tex-conf').textContent  = d.texture_conf || '—';
      document.getElementById('qr-sku').textContent    = d.qr_sku   || '—';
      document.getElementById('qr-batch').textContent  = d.qr_batch || '—';
      document.getElementById('qr-url').textContent    = d.qr_url   || '—';
      document.getElementById('ocr-sku').textContent   = d.ocr_sku  || '—';
      document.getElementById('ocr-text').textContent  = d.ocr_text || '—';
      document.getElementById('ts-tag').textContent    = d.timestamp || '—';
      document.getElementById('ts-pill').innerHTML     = '<span>' + (d.timestamp || '—') + '</span>';

      if (d.snapshot_b64) {
        document.getElementById('snapshot-box').innerHTML =
          '<img src="' + d.snapshot_b64 + '" alt="snapshot">';
      }
    })
    .catch(e => {
      setProcessing(false);
      document.getElementById('verdict-val').textContent = 'ERROR';
      document.getElementById('verdict-reason').textContent = e.toString();
      document.getElementById('verdict-banner').className = 'verdict-banner verdict-fail';
    });
}

function resetState() {
  fetch('/api/reset', {method:'POST'}).then(() => {
    document.getElementById('verdict-banner').className = 'verdict-banner verdict-ready';
    document.getElementById('verdict-val').textContent = '—';
    document.getElementById('verdict-reason').textContent = 'Press Process Frame to analyze the current camera feed';
    document.getElementById('ts-tag').textContent = '—';
    ['tex-sku','tex-conf','qr-sku','qr-batch','qr-url','ocr-sku','ocr-text'].forEach(id => {
      document.getElementById(id).textContent = '—';
    });
    document.getElementById('snapshot-box').innerHTML =
      '<span class="snap-placeholder">Snapshot appears after processing</span>';
  });
}
</script>
</body>
</html>"""

# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MATTRESSQC — SIDE CAMERA RIG DASHBOARD")
    print("  Dashboard: http://localhost:8003")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8003, threaded=True)
