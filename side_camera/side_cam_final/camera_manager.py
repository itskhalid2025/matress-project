import cv2
import threading
import time
import platform
import numpy as np
from config import CAMERA_QR_INDEX, CAMERA_BILL_TEXTURE_INDEX, CAMERA_TOP_INDEX, DEFAULT_RTSP


class CameraStream:
    def __init__(self, source, label="Camera", fallback_source=None):
        self.source = source
        self.fallback_source = fallback_source
        self.label = label
        self.cap = None
        self.running = False
        self.consecutive_failures = 0
        self.lock = threading.Lock()
        self.current_frame = None
        self.last_reconnect_time = 0

    def _try_open(self, src):
        if src is None:
            return None
        backend = cv2.CAP_DSHOW if (platform.system() == "Windows" and isinstance(src, int)) else cv2.CAP_V4L2
        if isinstance(src, str) and src.startswith("rtsp"):
            backend = cv2.CAP_FFMPEG

        try:
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                if isinstance(src, int):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                time.sleep(0.1)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    return cap
                cap.release()
        except Exception as e:
            print(f"[ERROR] Exception opening {self.label} on source {src}: {e}")
        return None

    def start(self):
        self._connect()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.running = False

        cap = self._try_open(self.source)
        active_src = self.source
        if not cap and self.fallback_source is not None:
            print(f"[INFO] Primary source '{self.source}' failed for {self.label}, trying fallback: '{self.fallback_source}'...")
            cap = self._try_open(self.fallback_source)
            active_src = self.fallback_source

        with self.lock:
            if cap and cap.isOpened():
                self.cap = cap
                self.running = True
                self.consecutive_failures = 0
                print(f"[INFO] Successfully opened {self.label} on Source: {active_src}")
            else:
                self.cap = None
                self.running = False
                print(f"[WARN] Could not open {self.label} on Source {self.source}. Simulation mode active.")

    def _update_loop(self):
        while True:
            frame_captured = False
            if self.running and self.cap:
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        with self.lock:
                            self.current_frame = frame
                            self.consecutive_failures = 0
                        frame_captured = True
                    else:
                        with self.lock:
                            self.consecutive_failures += 1
                except Exception as e:
                    with self.lock:
                        self.consecutive_failures += 1

            # Auto Reconnect logic if read fails repeatedly (e.g. USB glitch or temporary disconnect)
            if not frame_captured:
                if self.consecutive_failures > 10 or not self.running:
                    now = time.time()
                    if now - self.last_reconnect_time > 3.0:
                        self.last_reconnect_time = now
                        print(f"[RECONNECT] Attempting auto-reconnect for {self.label} (Source: {self.source})...")
                        self._connect()

                # Generate synthetic informational frame (never pitch black!)
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Dark slate background gradient
                frame[:, :] = (35, 25, 20)
                status_text = "RECONNECTING..." if self.consecutive_failures > 0 else "OFFLINE / SIMULATION"
                cv2.putText(frame, f"[ {self.label.upper()} ]", (40, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)
                cv2.putText(frame, f"Source: {self.source} | Status: {status_text}", (40, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
                t_str = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Timestamp: {t_str} | Check physical camera connection", (40, 410),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

                with self.lock:
                    self.current_frame = frame

                time.sleep(0.05)
            else:
                time.sleep(0.01)

    def read_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            else:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:, :] = (35, 25, 20)
                cv2.putText(frame, f"{self.label} - Initializing...", (40, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                return frame

    def is_connected(self):
        with self.lock:
            return self.running and self.cap is not None and self.cap.isOpened()

    def reconnect(self):
        self._connect()
        return self.is_connected()

    def stop(self):
        self.running = False
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None


# 3-Camera Manager Instances
qr_cam_stream = CameraStream(CAMERA_QR_INDEX, "Camera 1 (QR Scanner)")
bill_cam_stream = CameraStream(CAMERA_BILL_TEXTURE_INDEX, "Camera 2 (Side Bill OCR & Texture)")
top_cam_stream = CameraStream(CAMERA_TOP_INDEX, "Camera 3 (Top Camera Dimensions & Corner Label)", fallback_source=DEFAULT_RTSP)


def init_cameras():
    qr_cam_stream.start()
    time.sleep(0.4)
    bill_cam_stream.start()
    time.sleep(0.4)
    top_cam_stream.start()
    time.sleep(0.4)
