"""
camera_manager.py — Thread-Safe Single Camera Instance Manager for sidecam_new.

Guarantees only ONE VideoCapture instance exists.
Runs a background grab thread to maintain high FPS live streaming without doing frame processing.
"""

import threading
import time
import cv2
import numpy as np


class SingleCameraManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingleCameraManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        if self._initialized:
            return
        
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.latest_frame = None
        self.paused = False
        self.paused_frame = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.thread = None
        self.is_connected = False

        self._start_camera()
        self._initialized = True

    def _start_camera(self):
        """Initializes the webcam device and starts the background thread."""
        try:
            # Attempt DSHOW first for Windows compatibility, then standard fallback
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.is_connected = True
                print(f"[CameraManager] Connected to camera index {self.camera_index}")
            else:
                print(f"[CameraManager] WARNING: Could not open camera index {self.camera_index}. Using fallback frame mode.")
                self.is_connected = False
        except Exception as e:
            print(f"[CameraManager] Camera initialization error: {e}")
            self.is_connected = False

        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True, name="SideCamGrabLoop")
        self.thread.start()

    def _grab_loop(self):
        """Background thread continuously grabbing frames for zero latency."""
        delay = 1.0 / self.fps
        while self.running:
            start_t = time.time()

            if self.is_connected and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                else:
                    with self.frame_lock:
                        self.latest_frame = self._generate_error_frame("CAM FRAME READ ERROR")
            else:
                with self.frame_lock:
                    self.latest_frame = self._generate_error_frame("CAMERA DISCONNECTED")

            elapsed = time.time() - start_t
            sleep_t = max(0.005, delay - elapsed)
            time.sleep(sleep_t)

    def _generate_error_frame(self, message="CAMERA OFFLINE"):
        """Generates a placeholder grid frame when camera is unavailable."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(0, 720, 30):
            cv2.line(img, (0, i), (1280, i), (25, 25, 35), 1)
        for i in range(0, 1280, 30):
            cv2.line(img, (i, 0), (i, 720), (25, 25, 35), 1)
            
        cv2.putText(img, "MATTRESS INSPECTION SYSTEM", (380, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        cv2.putText(img, message, (460, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        cv2.putText(img, "Connect webcam to capture live feed", (430, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 160), 1)
        return img

    def pause(self):
        """Freezes the camera stream on the current frame."""
        with self.frame_lock:
            self.paused = True
            if self.latest_frame is not None:
                self.paused_frame = self.latest_frame.copy()
            else:
                self.paused_frame = self._generate_error_frame("PAUSED ON EMPTY FEED")
        print("[CameraManager] Stream paused.")

    def resume(self):
        """Unfreezes the camera stream."""
        with self.frame_lock:
            self.paused = False
            self.paused_frame = None
        print("[CameraManager] Stream resumed.")

    def get_mjpeg_bytes(self):
        """Returns encoded JPEG bytes of the current live frame (or paused frame) for stream response."""
        with self.frame_lock:
            if self.paused and self.paused_frame is not None:
                frame = self.paused_frame
            elif self.latest_frame is None:
                frame = self._generate_error_frame("INITIALIZING STREAM")
            else:
                frame = self.latest_frame

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            return b''
        return buffer.tobytes()

    def capture_current_frame(self):
        """Returns a BGR copy of the latest grabbed frame (or paused frame) for inspection processing."""
        with self.frame_lock:
            if self.paused and self.paused_frame is not None:
                return self.paused_frame.copy()
            elif self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return self._generate_error_frame("NO FRAME CAPTURED")

    def release(self):
        """Safely stops thread and closes VideoCapture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("[CameraManager] Camera released.")
