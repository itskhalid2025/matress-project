import cv2
import threading
import time
import platform
import numpy as np
from config import CAMERA_QR_INDEX, CAMERA_BILL_TEXTURE_INDEX, CAMERA_TOP_INDEX, DEFAULT_RTSP


class CameraStream:
    def __init__(self, source, label="Camera"):
        self.source = source
        self.label = label
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame = None

    def start(self):
        backend = cv2.CAP_DSHOW if (platform.system() == "Windows" and isinstance(self.source, int)) else cv2.CAP_V4L2
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            backend = cv2.CAP_FFMPEG

        self.cap = cv2.VideoCapture(self.source, backend)

        if self.cap.isOpened():
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.running = True
            print(f"[INFO] Started {self.label} on Source: {self.source}")
        else:
            print(f"[WARN] Could not open {self.label} on Source {self.source}. Using synthetic placeholder.")
            self.running = False

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while True:
            if self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.current_frame = frame
                else:
                    time.sleep(0.01)
            else:
                # Generate synthetic test pattern if camera is offline/disconnected
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, f"{self.label} ({self.source}) - Disconnected / Simulation",
                            (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                t_str = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Time: {t_str}", (40, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                with self.lock:
                    self.current_frame = frame
                time.sleep(0.03)

    def read_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            else:
                return np.zeros((720, 1280, 3), dtype=np.uint8)

    def is_connected(self):
        return self.running and self.cap and self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# 3-Camera Manager Instances
qr_cam_stream = CameraStream(CAMERA_QR_INDEX, "Camera 1 (QR Scanner)")
bill_cam_stream = CameraStream(CAMERA_BILL_TEXTURE_INDEX, "Camera 2 (Side Bill OCR & Texture)")
top_cam_stream = CameraStream(DEFAULT_RTSP, "Camera 3 (Top Camera Dimensions & Corner Label)")


def init_cameras():
    qr_cam_stream.start()
    bill_cam_stream.start()
    top_cam_stream.start()
