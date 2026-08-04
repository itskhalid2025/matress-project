#!/usr/bin/env python3
"""
============================================================
FILE: cctv_bestdimension_color.py
PATH: top_camera/global_shutter/cctv_bestdimension_color.py
============================================================
DESCRIPTION:
  Live Web Streaming & Measurement Rig powered by a CCTV RTSP
  camera stream, using PURE COLOR / CONTOUR based detection
  (no neural network) to locate and measure a mattress against
  a contrasting background.

WHY THIS VERSION (vs the YOLO rig):
  cctv_bestdimension_fixed.py runs a neural net first to get a
  bounding box, then refines it with contour analysis inside
  that box. This version skips the neural net entirely and runs
  color/contrast segmentation (HSV + CLAHE + Otsu, with a Canny
  edge fallback) directly on the FULL camera frame.

  Trade-off: lighter weight (no GPU/torch/ultralytics needed,
  runs fine on CPU), but it depends much more heavily on:
    - consistent, even lighting (avoid harsh shadows/glare)
    - a genuinely CONTRASTING background (the mattress must
      stand out clearly in brightness or saturation from
      whatever it's resting on)
    - nothing else large/high-contrast entering the frame
      (a box, a person, a cart) since there's no "mattress
      class" to distinguish it from other objects -- only
      geometry (rectangularity, angle, size) is used as a filter

ARCHITECTURE (kept identical to the YOLO rig otherwise):
  - RTSP capture with auto-reconnect
  - Full-frame color/contour segmentation -> rotated bounding box
  - Rectangularity + angle + area-ratio quality gate (rejects
    garbage contours -- e.g. a background box, or a sash/label
    that splits the mattress into two blobs)
  - Rolling-median smoothing + outlier rejection across frames
  - Pinhole-distance-corrected pixel -> cm conversion
  - Flask MJPEG stream + dashboard + live calibration endpoint
  - Background OCR thread for brand/banner text (unchanged)

TUNING:
  If detection is flaky, the first things to adjust are (in
  CONFIG below): INVERT_MASK (flip if mattress is darker than
  background), MIN_MATTRESS_AREA_RATIO / MAX_MATTRESS_AREA_RATIO
  (how much of the frame the mattress should occupy at your
  camera's fixed distance), and MIN_RECTANGULARITY (lower it if
  valid mattress contours are being rejected).
============================================================
"""

import sys
import os
import time
import threading
import re
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# Force TCP transport for RTSP
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Import the banner OCR engine from the top_camera module if available
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if TOP_CAM_DIR not in sys.path:
    sys.path.insert(0, TOP_CAM_DIR)

try:
    from banner import read_banner_fast
    import config as _cfg
    _cfg.SASH_S_THRESH = 60
    _cfg.SASH_V_THRESH = 60
    BANNER_OCR_AVAILABLE = True
    print("[cctv_stream] banner.py OCR engine loaded")
except Exception as _be:
    BANNER_OCR_AVAILABLE = False
    print(f"[cctv_stream] banner.py not available: {_be}")

app = Flask(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
CAMERA_DISTANCE_CM = 200.0     # Fixed camera distance to mattress plane
KNOWN_REF_WIDTH_CM = 47.0      # Physical reference horizontal width
KNOWN_REF_HEIGHT_CM = 46.0     # Physical reference vertical length

INVERT_MASK = False              # True if mattress is DARKER than the background
MIN_RECTANGULARITY = 0.75        # hull_area / rotated-rect area guard
MIN_MATTRESS_AREA_RATIO = 0.06   # rotated-rect area must be >= 6% of the full frame
MAX_MATTRESS_AREA_RATIO = 0.95   # and not basically the whole frame (lens edge/glare)
MAX_ANGLE_DEG = 30.0              # max rotation angle allowed
MIN_CONTOUR_AREA_PX = 1500        # raw pixel-area floor before any ratio checks

DETECT_EVERY_N_FRAMES = 1        # color detection is cheap -- safe to run every frame

SMOOTH_HISTORY = 7                # frames of rolling-median history
SMOOTH_MAX_DEVIATION = 0.35       # reject outlier readings (>35% deviation)
SMOOTH_WARMUP = 3                 # frames before measurement reported stable

RTSP_RECONNECT_AFTER_FAILURES = 60
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)

pixels_per_cm_x = 10.0      # Horizontal pixels per cm (calibrated via UI)
pixels_per_cm_y = 10.0      # Vertical pixels per cm   (calibrated via UI)
edge_correction = 1.0
_calib_lock = threading.Lock()

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ==============================================================================
# Color / contour segmentation (this replaces the YOLO stage entirely)
# ==============================================================================
def _normalize_angle(angle):
    a = angle % 90
    if a > 45:
        a -= 90
    return a


def _rect_quality_ok(hull, rect, frame_area):
    (_, _), (w_cv, h_cv), angle = rect
    rect_area = max(w_cv * h_cv, 1e-6)

    if abs(_normalize_angle(angle)) > MAX_ANGLE_DEG:
        return False

    area_ratio = rect_area / frame_area
    if area_ratio < MIN_MATTRESS_AREA_RATIO or area_ratio > MAX_MATTRESS_AREA_RATIO:
        return False

    rectangularity = cv2.contourArea(hull) / rect_area
    if rectangularity < MIN_RECTANGULARITY:
        return False

    return True


def _largest_valid_contour(mask, frame_area):
    """Scan the top-N largest contours in the mask and return the first
    one that passes the rectangularity/angle/area-ratio quality gate."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA_PX:
            continue
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        if _rect_quality_ok(hull, rect, frame_area):
            return hull, rect
    return None


def _threshold_mask(bgr):
    """Primary mask: HSV value (CLAHE + Otsu) OR'd with a saturation
    threshold. This is the core 'contrast background' segmentation --
    it works because the mattress and background differ strongly in
    either brightness or colorfulness."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_eq = _clahe.apply(hsv[:, :, 2])
    thresh_type = cv2.THRESH_BINARY if INVERT_MASK else cv2.THRESH_BINARY_INV
    _, mask_v = cv2.threshold(v_eq, 0, 255, thresh_type + cv2.THRESH_OTSU)
    _, mask_s = cv2.threshold(hsv[:, :, 1], 35, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_v, mask_s)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _edge_mask(bgr):
    """Fallback mask for scenes where the Otsu split is weak (low
    contrast lighting) -- uses Canny edges instead of a flat threshold."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    return closed


class MeasurementSmoother:
    def __init__(self, history=SMOOTH_HISTORY, max_deviation=SMOOTH_MAX_DEVIATION, warmup=SMOOTH_WARMUP):
        self.history = history
        self.max_deviation = max_deviation
        self.warmup = warmup
        self.w_buf = deque(maxlen=history)
        self.h_buf = deque(maxlen=history)

    def reset(self):
        self.w_buf.clear()
        self.h_buf.clear()

    def update(self, w_cm, h_cm):
        if w_cm is None or h_cm is None or w_cm <= 0 or h_cm <= 0:
            return None, None, False

        if len(self.w_buf) >= self.warmup:
            med_w = float(np.median(self.w_buf))
            med_h = float(np.median(self.h_buf))
            if med_w > 0 and med_h > 0:
                dev_w = abs(w_cm - med_w) / med_w
                dev_h = abs(h_cm - med_h) / med_h
                if dev_w > self.max_deviation or dev_h > self.max_deviation:
                    return round(med_w, 1), round(med_h, 1), True

        self.w_buf.append(w_cm)
        self.h_buf.append(h_cm)
        stable = len(self.w_buf) >= self.warmup
        return round(float(np.median(self.w_buf)), 1), round(float(np.median(self.h_buf)), 1), stable


_smoother = MeasurementSmoother()


def process_frame_tight_geometry(img, px_cm_x, px_cm_y, distance_cm=200.0, frame_index=0):
    """
    Computes mattress dimensions using PURE color/contour segmentation
    on the full camera frame -- no neural network stage.
    """
    if img is None or img.size == 0:
        return None, None, img, None, False

    annotated = img.copy()
    h_orig, w_orig = img.shape[:2]
    frame_area = float(w_orig * h_orig)

    try:
        # Color detection is cheap, so we just run it every frame regardless
        # of DETECT_EVERY_N_FRAMES; the knob is kept for parity/tuning.
        found = _largest_valid_contour(_threshold_mask(img), frame_area)
        if found is None:
            found = _largest_valid_contour(_edge_mask(img), frame_area)

        if found is None:
            # No qualifying contour this frame -- don't guess, just report searching.
            _smoother.reset()
            cv2.putText(annotated, "Searching Target...", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return None, None, annotated, None, False

        hull_final, rect_final = found
        (cx, cy), (w_px, h_px), angle = rect_final
        if angle < -45 or angle > 45:
            w_px, h_px = h_px, w_px

        cv2.drawContours(annotated, [hull_final], -1, (0, 0, 255), 2)
        box_pts = np.int32(cv2.boxPoints(rect_final))
        cv2.drawContours(annotated, [box_pts], 0, (0, 255, 0), 3)
        cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 255, 255), -1)

        # ==================================================================
        # Pinhole Distance Math + Smoothing (unchanged from the YOLO rig)
        # ==================================================================
        if px_cm_x <= 0 or px_cm_y <= 0:
            _smoother.reset()
            return None, None, annotated, None, False

        nW_cm_raw = (w_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_x * edge_correction
        nH_cm_raw = (h_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_y * edge_correction

        nW_cm, nH_cm, stable = _smoother.update(round(nW_cm_raw, 1), round(nH_cm_raw, 1))
        if nW_cm is None:
            return None, None, annotated, None, False

        nW_in = round(nW_cm / 2.54, 1)
        nH_in = round(nH_cm / 2.54, 1)

        suffix = "" if stable else " (stabilizing...)"
        cv2.putText(annotated, f'W: {nW_cm} cm ({nW_in} in){suffix}', (int(cx) - 120, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f'L: {nH_cm} cm ({nH_in} in)', (int(cx) - 120, int(cy) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)

        x1_g = int(max(0, cx - w_px / 2))
        y1_g = int(max(0, cy - h_px / 2))
        x2_g = int(min(w_orig, cx + w_px / 2))
        y2_g = int(min(h_orig, cy + h_px / 2))

        return nW_cm, nH_cm, annotated, (x1_g, y1_g, x2_g, y2_g), stable

    except Exception as e:
        print(f"[dimensions] Error during processing: {str(e)}")

    return None, None, annotated, None, False


# ---------------------------------------------------------------------------
# Background OCR Thread — runs read_banner_fast() decoupled from capture loop
# (unchanged from the YOLO rig -- detection method doesn't affect OCR)
# ---------------------------------------------------------------------------
_ocr_latest_frame = None
_ocr_frame_lock   = threading.Lock()
_ocr_result       = ""
_ocr_result_lock  = threading.Lock()

def _ocr_worker():
    global _ocr_result, _ocr_latest_frame
    while True:
        frame_to_process = None
        with _ocr_frame_lock:
            if _ocr_latest_frame is not None:
                frame_to_process = _ocr_latest_frame.copy()
                _ocr_latest_frame = None
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
        time.sleep(0.5)

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
            "camera_type": "Offline", "distance_cm": CAMERA_DISTANCE_CM
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
        # 1. Try RTSP Stream
        if isinstance(self.webcam_index, str) and any(self.webcam_index.startswith(proto) for proto in ["rtsp://", "rtmp://", "http://", "https://"]):
            try:
                print(f"[cctv_stream] Connecting to RTSP Stream: {self.webcam_index} using FFMPEG backend...")
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.camera_type = "RTSP CCTV"
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cctv_stream] Connected to RTSP camera successfully at {actual_w}x{actual_h}")
                    return
            except Exception as e:
                print(f"[cctv_stream] RTSP stream connection failed: {e}")

        # 2. Try USB Webcam
        try:
            idx = self.webcam_index if isinstance(self.webcam_index, int) else 0
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
            cap = cv2.VideoCapture(idx, backend)

            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap = cap
                self.camera_type = f"USB Cam (Index {idx})"
                return
        except Exception as e:
            print(f"[cctv_stream] USB webcam initialization failed: {e}")

        # 3. Fallback: Picamera2
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            self.picam2 = picam
            self.camera_type = "Picamera2 (IMX296)"
        except Exception as e:
            print(f"[cctv_stream] Picamera2 fallback failed: {e}")
            self.camera_type = "Offline"

    def _reconnect_rtsp(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        _smoother.reset()
        self._init_camera()

    def _worker_loop(self):
        global pixels_per_cm_x, pixels_per_cm_y
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        consecutive_failures = 0

        while self.running:
            raw_bgr = None
            if self.picam2:
                try:
                    rgb = self.picam2.capture_array()
                    raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    consecutive_failures = 0
                except Exception:
                    time.sleep(0.005)
                    continue
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    raw_bgr = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= RTSP_RECONNECT_AFTER_FAILURES:
                        print("[cctv_stream] RTSP stream unresponsive — attempting reconnect...")
                        self._reconnect_rtsp()
                        consecutive_failures = 0
                    time.sleep(0.005)
                    continue
            else:
                time.sleep(0.01)
                continue

            if raw_bgr is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            w_cm, h_cm, annotated, target_box, stable = process_frame_tight_geometry(
                raw_bgr, pixels_per_cm_x, pixels_per_cm_y,
                distance_cm=CAMERA_DISTANCE_CM, frame_index=self.frame_count,
            )

            if self.frame_count % 5 == 0:
                with _ocr_frame_lock:
                    global _ocr_latest_frame
                    _ocr_latest_frame = raw_bgr.copy()

            with _ocr_result_lock:
                self.latest_ocr = _ocr_result

            if self.latest_ocr:
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(annotated, f"BRAND/OCR: {self.latest_ocr}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            fps_txt = f"FPS: {self.current_fps} | Distance: {CAMERA_DISTANCE_CM} cm"
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', annotated, jpeg_params)
            if ret:
                w_in = round(w_cm / 2.54, 1) if w_cm is not None else None
                h_in = round(h_cm / 2.54, 1) if h_cm is not None else None
                if w_cm is not None:
                    status = "Target Detected" if stable else "Stabilizing..."
                else:
                    status = "Searching Target..."
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_dims = {
                        "width_cm": w_cm,
                        "width_in": w_in,
                        "height_cm": h_cm,
                        "height_in": h_in,
                        "ocr_text": self.latest_ocr,
                        "status": status,
                        "camera_type": self.camera_type,
                        "distance_cm": CAMERA_DISTANCE_CM
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
    <title>CCTV Mattress Color-Contour Stream</title>
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
        <h1>📐 CCTV <span>Color-Contour Dimension Stream</span></h1>
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
                    <div class="stat-label">Camera Distance</div>
                    <div class="stat-val" style="color: #38bdf8;">{{ distance_cm }} cm</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Status</div>
                    <div class="stat-val" id="valStatus" style="font-size: 16px; color: var(--green);">Searching Target...</div>
                </div>

                <div class="calib-box">
                    <div class="stat-label">Dual-Axis Distance Calibration</div>
                    <div style="font-size: 12px; color: var(--dim); margin-bottom: 6px;">Preset: Known 47cm (W) x 46cm (H) @ 200cm</div>
                    <div class="calib-input">
                        <input type="number" id="knownW" value="47" placeholder="Known Width cm (47)">
                        <input type="number" id="knownH" value="46" placeholder="Known Length cm (46)">
                        <button onclick="calibrateBoth()">Calibrate Dual Axes</button>
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
            fetch(`/api/calibrate?known_width_cm=${knownW || '47'}&known_length_cm=${knownH || '46'}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Calibration Updated:\\nWidth Scale: ' + data.px_cm_x + ' px/cm\\nLength Scale: ' + data.px_cm_y + ' px/cm');
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
    return render_template_string(
        HTML_TEMPLATE,
        pixels_per_cm=round(pixels_per_cm_x, 2),
        camera_type=streamer.camera_type,
        distance_cm=CAMERA_DISTANCE_CM
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/dims')
def api_dims():
    return jsonify(streamer.get_dims())

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    global pixels_per_cm_x, pixels_per_cm_y
    try:
        known_w_cm = float(request.args.get('known_width_cm', 47) or 47)
        known_h_cm = float(request.args.get('known_length_cm', 46) or 46)

        dims = streamer.get_dims()
        curr_width_cm = dims.get("width_cm")
        curr_height_cm = dims.get("height_cm")

        if curr_width_cm is None or curr_height_cm is None or curr_width_cm <= 0:
            return jsonify({"success": False, "error": "No valid mattress target detected to calibrate against"}), 422

        with _calib_lock:
            pixel_w = curr_width_cm * pixels_per_cm_x
            pixel_h = curr_height_cm * pixels_per_cm_y

            pixels_per_cm_x = round(pixel_w / known_w_cm, 3)
            pixels_per_cm_y = round(pixel_h / known_h_cm, 3)

        _smoother.reset()

        print(f"[calibrate] Updated Scale Ratios at Distance {CAMERA_DISTANCE_CM}cm: X={pixels_per_cm_x} px/cm, Y={pixels_per_cm_y} px/cm")
        return jsonify({"success": True, "px_cm_x": pixels_per_cm_x, "px_cm_y": pixels_per_cm_y})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" 🚀 PINHOLE DISTANCE-AWARE CCTV COLOR-CONTOUR STREAM ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()