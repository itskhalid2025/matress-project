#!/usr/bin/env python3
"""
stream.py — Minimal high-FPS camera stream with FPS overlay directly on video feed.
"""

import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

class MinimalCameraStream:
    def __init__(self, width=1456, height=1088, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self.picam2 = None
        self.cap = None
        
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = False
        
        self.fps_counter = 0
        self.current_fps = 0.0
        self.last_fps_time = time.time()
        
        self._init_camera()
        
        # Start fast background capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        # 1. Picamera2 native video mode
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            try:
                config = picam.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                picam.configure(config)
                picam.start()
                self.picam2 = picam
                print(f"[stream] Connected via Picamera2 ({self.fps} FPS target)")
                return
            except Exception as inner_e:
                picam.close()
                raise inner_e
        except Exception as e:
            print(f"[stream] Picamera2 init note: {e}")

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
                print("[stream] Connected via OpenCV V4L2")
                return
        except Exception as e:
            print(f"[stream] OpenCV V4L2 failed: {e}")

    def _capture_loop(self):
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        
        while self.running:
            frame = None
            if self.picam2:
                try:
                    frame = self.picam2.capture_array()
                except Exception:
                    time.sleep(0.005)
                    continue
            elif self.cap and self.cap.isOpened():
                ret, bgr = self.cap.read()
                if ret:
                    frame = bgr
                else:
                    time.sleep(0.005)
                    continue

            if frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Update FPS calculation
            self.fps_counter += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.last_fps_time = now

            # Draw FPS directly on video feed
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, jpeg_params)
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

camera = MinimalCameraStream()

def generate_feed():
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
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Camera Stream</title>
    <style>
        body { margin: 0; padding: 0; background: #000; display: flex; justify-content: center; align-items: center; min-height: 100vh; overflow: hidden; }
        img { max-width: 100vw; max-height: 100vh; object-fit: contain; }
    </style>
</head>
<body>
    <img src="/video_feed">
</body>
</html>'''

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" 🎥 MINIMAL STREAM ACTIVE: http://localhost:5000/")
    print("="*50 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera.release()
