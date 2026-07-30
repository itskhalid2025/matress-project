#!/usr/bin/env python3
"""
global_shutter_test.py — High-FPS IMX296 Global Shutter Streamer for Raspberry Pi 5.
"""

import time
import socket
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

app = Flask(__name__)

class HighFPSGlobalShutter:
    def __init__(self, width=1456, height=1088, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        self.picam2 = None
        self.cap = None
        self.backend = "Initializing"
        
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        self._init_camera()
        
        # Start background pre-encoder thread
        self.running = True
        self.thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # 1. Picamera2 High-FPS Video Mode
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameDurationLimits": (int(1000000 / self.fps), int(1000000 / self.fps))}
            )
            picam.configure(config)
            picam.start()
            self.picam2 = picam
            self.backend = "Picamera2 (IMX296 60FPS Video Mode)"
            print(f"[Camera] Connected using {self.backend}")
            return
        except Exception as e:
            print(f"[Camera] Picamera2 init note: {e}")

        # 2. OpenCV V4L2 fallback
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                self.backend = "OpenCV V4L2"
                return
        except Exception as e:
            print(f"[Camera] OpenCV V4L2 init failed: {e}")

        self.backend = "Disconnected"

    def _grab_worker(self):
        jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        while self.running:
            frame = None
            if self.picam2:
                try:
                    rgb = self.picam2.capture_array()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception:
                    time.sleep(0.01)
                    continue
            elif self.cap and self.cap.isOpened():
                ret, bgr = self.cap.read()
                if ret:
                    frame = bgr
                else:
                    time.sleep(0.01)
                    continue

            if frame is not None:
                # Add timestamp watermark
                ts = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"IMX296 60FPS | {ts}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame, jpeg_quality)
                if ret:
                    with self.lock:
                        self.latest_jpeg = buffer.tobytes()
            
            time.sleep(0.002)

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

camera = HighFPSGlobalShutter(width=1456, height=1088, fps=60)

def generate_frames():
    last_sent = None
    while True:
        jpeg = camera.get_jpeg()
        if jpeg is not None and jpeg != last_sent:
            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMX296 High-FPS Global Shutter Stream</title>
        <style>
            body { background: #0f172a; color: #f8fafc; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #06b6d4; margin-bottom: 5px; }
            p { color: #94a3b8; font-size: 14px; margin-bottom: 20px; }
            .container { max-width: 1000px; margin: 0 auto; background: #1e293b; padding: 15px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            img { width: 100%; height: auto; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ IMX296 High-FPS (60 FPS) Stream</h1>
            <p>Non-blocking background pre-encoding stream</p>
            <img src="/video_feed" alt="Camera Stream">
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    ip_addr = get_ip()
    port = 5000
    print("\n" + "="*60)
    print(" 🚀 HIGH-FPS IMX296 FLASK STREAM ACTIVE (UP TO 60 FPS)!")
    print(f" 👉 Local URL (VS Code SSH): http://localhost:{port}/")
    print(f" 👉 Network URL:            http://{ip_addr}:{port}/")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=port, threaded=True)
    finally:
        camera.release()