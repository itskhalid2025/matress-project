#!/usr/bin/env python3
"""
stream.py — Minimal high-FPS camera stream with natural colors.
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

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):

        # ---------- Picamera2 ----------
        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "RGB888"
                },
                buffer_count=3
            )

            self.picam2.configure(config)
            self.picam2.start()

            # Allow AWB/AE to settle
            time.sleep(2)

            self.picam2.set_controls({
                "AwbEnable": True,
                "AeEnable": True
            })

            print("[stream] Picamera2 started successfully.")
            print(self.picam2.camera_configuration())

            return

        except Exception as e:
            print("Picamera2 failed:", e)

        # ---------- OpenCV fallback ----------
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def _capture_loop(self):

        jpeg_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            90
        ]

        while self.running:

            frame = None

            if self.picam2 is not None:

                try:
                    frame = self.picam2.capture_array()

                    # RGB -> BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                except Exception:
                    continue

            elif self.cap is not None:

                ret, frame = self.cap.read()

                if not ret:
                    continue

            if frame is None:
                continue

            # ---------------- FPS ----------------

            self.fps_counter += 1

            now = time.time()

            if now - self.last_fps_time >= 1:

                self.current_fps = self.fps_counter / (
                    now - self.last_fps_time
                )

                self.fps_counter = 0
                self.last_fps_time = now

            fps_text = f"FPS: {self.current_fps:.1f}"

            cv2.putText(
                frame,
                fps_text,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                fps_text,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                jpeg_params
            )

            if ok:
                with self.lock:
                    self.latest_jpeg = jpg.tobytes()

    def get_jpeg(self):

        with self.lock:
            return self.latest_jpeg

    def release(self):

        self.running = False

        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()

        if self.cap is not None:
            self.cap.release()


camera = MinimalCameraStream()


def generate():

    while True:

        frame = camera.get_jpeg()

        if frame is None:
            time.sleep(0.005)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


@app.route("/")
def index():

    return """
<!DOCTYPE html>
<html>
<head>
<title>Camera</title>

<style>

body{
margin:0;
background:black;
display:flex;
justify-content:center;
align-items:center;
height:100vh;
}

img{
max-width:100%;
max-height:100%;
}

</style>

</head>

<body>

<img src="/video_feed">

</body>

</html>

"""


@app.route("/video_feed")
def video_feed():

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    print("Camera running on http://0.0.0.0:5000")

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            debug=False,
        )

    finally:

        camera.release()