#!/usr/bin/env python3
"""
============================================================
FILE: cctv_contour_only.py
PATH: top_camera/global_shutter/cctv_contour_only.py
============================================================
DESCRIPTION:
  Pure Color & Contour Mattress Dimension Measurement Rig.
  Operates without any YOLO model dependency — relies purely on
  HSV thresholding, Canny edge detection, convex hulls, and rotated
  minAreaRect geometry.

FEATURES:
1. Pure Color & Contour segmentation (HSV CLAHE Otsu + Canny fallback).
2. Live Web Dashboard UI with real-time video feed & measurement stats.
3. Interactive Dual-Axis Distance Calibration via Web UI (/api/calibrate).
4. RTSP CCTV stream support with TCP transport and auto-reconnect.
5. Rolling-median measurement smoothing with outlier rejection.
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

# Force TCP transport for RTSP stream stability
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Import OCR engine if available
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
    print("[contour_only] banner.py OCR engine loaded")
except Exception as _be:
    BANNER_OCR_AVAILABLE = False
    print(f"[contour_only] banner.py not available: {_be}")

app = Flask(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
CAMERA_DISTANCE_CM = 200.0     # Fixed camera distance to mattress plane
KNOWN_REF_WIDTH_CM = 47.0      # Physical reference horizontal width
KNOWN_REF_HEIGHT_CM = 46.0     # Physical reference vertical length

ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX = 50, 30, 1750, 1050  # Default Conveyor Crop Box
_roi_lock = threading.Lock()

MIN_RECTANGULARITY = 0.72        # hull_area / rotated-rect area (rect geometry guard)
MAX_ANGLE_DEG = 35.0             # max rotation angle allowed
MIN_CONTOUR_AREA = 2500          # minimum contour area px^2 to avoid background noise

SMOOTH_HISTORY = 7               # frames of rolling-median history
SMOOTH_MAX_DEVIATION = 0.35      # reject outlier readings (>35% deviation)
SMOOTH_WARMUP = 3                # frames before measurement reported stable

RTSP_RECONNECT_AFTER_FAILURES = 60
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)

pixels_per_cm_x = 10.0      # Horizontal pixels per cm (calibrated via UI)
pixels_per_cm_y = 10.0      # Vertical pixels per cm   (calibrated via UI)
edge_correction = 1.0
_calib_lock = threading.Lock()

# Reusable CLAHE instance
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ==============================================================================
# Pure Color & Contour Segmentation Helpers
# ==============================================================================
def _normalize_angle(angle):
    a = angle % 90
    if a > 45:
        a -= 90
    return a


def _rect_quality_ok(hull, rect):
    (_, _), (w_cv, h_cv), angle = rect
    rect_area = max(w_cv * h_cv, 1e-6)

    if abs(_normalize_angle(angle)) > MAX_ANGLE_DEG:
        return False

    rectangularity = cv2.contourArea(hull) / rect_area
    if rectangularity < MIN_RECTANGULARITY:
        return False

    return True


def _threshold_mask(bgr_crop):
    """Multi-Color & Contrast Fusion Mask: Handles blue, white, yellow, and patterned mattresses."""
    # Bilateral filter smoothes internal stitching/patterns while keeping outer edges sharp
    smoothed = cv2.bilateralFilter(bgr_crop, 7, 75, 75)
    
    # 1. HSV Color Space (V-channel + Saturation)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    v_eq = _clahe.apply(hsv[:, :, 2])
    _, mask_v = cv2.threshold(v_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_s = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    
    # 2. Lab Color Space (b* channel captures yellow/blue patterns against background)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    b_eq = _clahe.apply(lab[:, :, 2])
    _, mask_b = cv2.threshold(b_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combined Fusion
    mask = cv2.bitwise_or(mask_v, mask_s)
    mask = cv2.bitwise_or(mask, mask_b)

    # Large 25x25 morphological closing bridges thick sashes (like MAXI PLUSH purple sash)
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_lg, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_sm, iterations=1)
    return mask


def _edge_mask(bgr_crop):
    """Multi-Scale Edge Mask: Morphological Gradient + Canny Edges."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Morphological gradient captures subtle color boundaries
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel_grad)
    _, mask_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(blur, 20, 100)
    combined = cv2.bitwise_or(mask_grad, edges)

    kernel_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_lg, iterations=3)
    closed = cv2.dilate(closed, kernel_lg, iterations=2)
    return closed


def _find_best_contour(crop_img):
    """Finds and unifies all mattress contour sections across sashes and color splits."""
    roi_area = crop_img.shape[0] * crop_img.shape[1]

    for mask_fn in [_threshold_mask, _edge_mask]:
        mask = mask_fn(crop_img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # 1. Try unified multi-contour merge first (bridges across sashes, logos, and color splits)
        valid_cs = [c for c in contours if MIN_CONTOUR_AREA < cv2.contourArea(c) < (roi_area * 0.95)]
        if valid_cs:
            all_pts = np.vstack(valid_cs)
            hull = cv2.convexHull(all_pts)
            rect = cv2.minAreaRect(hull)
            if _rect_quality_ok(hull, rect):
                return hull, rect

        # 2. Fall back to individual top contours
        sorted_cs = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in sorted_cs:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA or cv2.contourArea(c) > (roi_area * 0.95):
                continue
            hull = cv2.convexHull(c)
            rect = cv2.minAreaRect(hull)
            if _rect_quality_ok(hull, rect):
                return hull, rect

    return None


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


def process_frame_pure_contour(img, px_cm_x, px_cm_y, distance_cm=200.0):
    """
    Computes mattress dimensions using pure Color & Contour algorithms.
    """
    if img is None or img.size == 0:
        return None, None, img, None, False

    annotated = img.copy()
    h_orig, w_orig = img.shape[:2]

    with _roi_lock:
        curr_x0, curr_y0, curr_x1, curr_y1 = ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX

    x0, y0, x1, y1 = max(0, curr_x0), max(0, curr_y0), min(curr_x1, w_orig), min(curr_y1, h_orig)
    crop_img = img[y0:y1, x0:x1]

    # Draw Conveyor ROI Box (Cyan)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)

    if crop_img.size == 0:
        _smoother.reset()
        return None, None, annotated, None, False

    try:
        found = _find_best_contour(crop_img)

        if found is None:
            _smoother.reset()
            return None, None, annotated, (x0, y0, x1, y1), False

        hull_final, rect_final = found
        (cx, cy), (w_px, h_px), angle = rect_final
        if angle < -45 or angle > 45:
            w_px, h_px = h_px, w_px

        cx_orig = cx + x0
        cy_orig = cy + y0

        # Draw Red Convex Hull & Green Rotated Rect
        hull_offset = hull_final + np.array([x0, y0])
        cv2.drawContours(annotated, [hull_offset], -1, (0, 0, 255), 2)
        box_pts = cv2.boxPoints(rect_final)
        box_offset = np.int32(box_pts + np.array([x0, y0]))
        cv2.drawContours(annotated, [box_offset], 0, (0, 255, 0), 3)
        cv2.circle(annotated, (int(cx_orig), int(cy_orig)), 6, (0, 255, 255), -1)

        # Distance Math & Calibration Scaling
        nW_cm_raw = (w_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_x * edge_correction
        nH_cm_raw = (h_px * (distance_cm / CAMERA_DISTANCE_CM)) / px_cm_y * edge_correction

        nW_cm, nH_cm, stable = _smoother.update(round(nW_cm_raw, 1), round(nH_cm_raw, 1))
        if nW_cm is None:
            return None, None, annotated, (x0, y0, x1, y1), False

        nW_in = round(nW_cm / 2.54, 1)
        nH_in = round(nH_cm / 2.54, 1)

        suffix = "" if stable else " (stabilizing...)"
        cv2.putText(annotated, f'W: {nW_cm} cm ({nW_in} in){suffix}', (int(cx_orig) - 120, int(cy_orig) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f'L: {nH_cm} cm ({nH_in} in)', (int(cx_orig) - 120, int(cy_orig) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)

        return nW_cm, nH_cm, annotated, (x0, y0, x1, y1), stable

    except Exception as e:
        print(f"[contour_only] Error during processing: {str(e)}")

    _smoother.reset()
    return None, None, annotated, (x0, y0, x1, y1), False


# ---------------------------------------------------------------------------
# Background OCR Thread
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
                print(f"[contour_only] Connecting to RTSP: {self.webcam_index} using FFMPEG...")
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.camera_type = "RTSP CCTV"
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[contour_only] Connected to RTSP camera at {actual_w}x{actual_h}")
                    return
            except Exception as e:
                print(f"[contour_only] RTSP stream connection failed: {e}")

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
            print(f"[contour_only] USB webcam initialization failed: {e}")

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
            print(f"[contour_only] Picamera2 fallback failed: {e}")
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
                        print("[contour_only] RTSP stream unresponsive — attempting reconnect...")
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

            w_cm, h_cm, annotated, target_box, stable = process_frame_pure_contour(
                raw_bgr, pixels_per_cm_x, pixels_per_cm_y,
                distance_cm=CAMERA_DISTANCE_CM,
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

            fps_txt = f"FPS: {self.current_fps} | Distance: {CAMERA_DISTANCE_CM} cm (Pure Contour)"
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, fps_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

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
    <title>Pure Color & Contour Mattress Measurement Dashboard</title>
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
        <h1>📐 Color & Contour <span>Mattress Measurement</span></h1>
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

                <div class="calib-box" style="margin-top: 15px;">
                    <div class="stat-label">Live Adjustable Conveyor ROI Crop</div>
                    <div style="font-size: 12px; color: var(--dim); margin-bottom: 6px;">Set ROI Boundaries (X0, Y0, X1, Y1)</div>
                    <div class="calib-input">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                            <input type="number" id="roiX0" value="50" placeholder="X-Min (Left)">
                            <input type="number" id="roiY0" value="30" placeholder="Y-Min (Top)">
                            <input type="number" id="roiX1" value="1750" placeholder="X-Max (Right)">
                            <input type="number" id="roiY1" value="1050" placeholder="Y-Max (Bottom)">
                        </div>
                        <button onclick="updateRoiControls()">Apply Live ROI Crop</button>
                        <div style="display: flex; gap: 6px; margin-top: 6px;">
                            <button onclick="presetRoi(0, 0, 1920, 1080)" style="background: #334155; font-size: 11px; padding: 6px; border: none; border-radius: 6px; color: #fff; cursor: pointer;">Full Frame</button>
                            <button onclick="presetRoi(50, 30, 1750, 1050)" style="background: #334155; font-size: 11px; padding: 6px; border: none; border-radius: 6px; color: #fff; cursor: pointer;">Wide Conveyor</button>
                            <button onclick="presetRoi(200, 50, 1300, 1030)" style="background: #334155; font-size: 11px; padding: 6px; border: none; border-radius: 6px; color: #fff; cursor: pointer;">Center ROI</button>
                        </div>
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

        function updateRoiControls() {
            const x0 = document.getElementById('roiX0').value;
            const y0 = document.getElementById('roiY0').value;
            const x1 = document.getElementById('roiX1').value;
            const y1 = document.getElementById('roiY1').value;
            fetch(`/api/roi?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Live ROI Crop Updated to [' + data.roi.join(', ') + ']');
                    }
                });
        }

        function presetRoi(x0, y0, x1, y1) {
            document.getElementById('roiX0').value = x0;
            document.getElementById('roiY0').value = y0;
            document.getElementById('roiX1').value = x1;
            document.getElementById('roiY1').value = y1;
            updateRoiControls();
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

@app.route('/api/roi', methods=['GET', 'POST'])
def api_roi():
    global ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX
    if request.method == 'POST':
        try:
            x0 = int(request.args.get('x0', ROI_X0))
            y0 = int(request.args.get('y0', ROI_Y0))
            x1 = int(request.args.get('x1', ROI_X1_MAX))
            y1 = int(request.args.get('y1', ROI_Y1_MAX))
            with _roi_lock:
                ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX = x0, y0, x1, y1
            _smoother.reset()
            print(f"[roi] Updated Live ROI Bounds to: [{ROI_X0}, {ROI_Y0}, {ROI_X1_MAX}, {ROI_Y1_MAX}]")
            return jsonify({"success": True, "roi": [ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX]})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        with _roi_lock:
            return jsonify({"roi": [ROI_X0, ROI_Y0, ROI_X1_MAX, ROI_Y1_MAX]})

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
    print(" 🚀 PURE COLOR & CONTOUR MATTRESS MEASUREMENT ACTIVE: http://localhost:5000/")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        streamer.release()
