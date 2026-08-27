import cv2
import threading
import time
import platform
import numpy as np
from config import CAMERA_QR_INDEX, CAMERA_BILL_TEXTURE_INDEX, CAMERA_TOP_INDEX, DEFAULT_RTSP


class CameraStream:
    def __init__(self, source, label="Camera"):
        if isinstance(source, (list, tuple)):
            seen = set()
            self.sources = [x for x in source if not (x in seen or seen.add(x))]
        else:
            self.sources = [source]
        self.label = label
        self.cap = None
        self.active_source = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.thread = None

    def connect(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.active_source = None

        for src in self.sources:
            if src is None:
                continue

            # Determine appropriate backends. Use CAP_DSHOW exclusively on Windows for device indices to prevent MSMF hanging.
            if platform.system() == "Windows" and isinstance(src, int):
                backends_to_try = [cv2.CAP_DSHOW]
            elif isinstance(src, str) and src.startswith("rtsp"):
                backends_to_try = [cv2.CAP_FFMPEG]
            else:
                backends_to_try = [cv2.CAP_V4L2, None]

            for b in backends_to_try:
                args = (src,) if b is None else (src, b)
                try:
                    cap = cv2.VideoCapture(*args)
                    if cap and cap.isOpened():
                        # Read initial test frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            # Try setting resolution if USB camera
                            if isinstance(src, int):
                                try:
                                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                                    ret2, frame2 = cap.read()
                                    if ret2 and frame2 is not None and frame2.size > 0:
                                        test_frame = frame2
                                except Exception:
                                    pass

                            with self.lock:
                                self.cap = cap
                                self.active_source = src
                                self.current_frame = test_frame
                            print(f"[INFO] Successfully started {self.label} on Source: {src} (backend: {b})")
                            return True
                        else:
                            cap.release()
                except Exception as e:
                    print(f"[DEBUG] {self.label} failed on source {src} with backend {b}: {e}")
                    pass

        print(f"[WARN] Could not open {self.label} on any source in {self.sources}. Using simulation placeholder.")
        return False

    def start(self):
        self.connect()
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        consecutive_failures = 0
        while self.running:
            with self.lock:
                cap = self.cap

            if cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        consecutive_failures = 0
                        with self.lock:
                            self.current_frame = frame
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            print(f"[WARN] {self.label} lost signal ({consecutive_failures} blank reads). Reconnecting...")
                            self.connect()
                            consecutive_failures = 0
                        time.sleep(0.01)
                except Exception:
                    consecutive_failures += 1
                    time.sleep(0.01)
            else:
                # Generate synthetic test pattern if camera is offline/disconnected
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                src_display = self.active_source if self.active_source is not None else (self.sources[0] if self.sources else "N/A")
                cv2.putText(frame, f"{self.label} ({src_display}) - Disconnected / Simulation",
                            (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                t_str = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Time: {t_str}", (40, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                with self.lock:
                    self.current_frame = frame
                time.sleep(0.05)

    def read_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            else:
                return np.zeros((720, 1280, 3), dtype=np.uint8)

    def is_connected(self):
        with self.lock:
            return self.running and self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None


# 3-Camera Manager Instances with clean, non-conflicting source lists
qr_cam_stream = CameraStream([CAMERA_QR_INDEX, 0], "Camera 1 (QR Scanner)")
bill_cam_stream = CameraStream([CAMERA_BILL_TEXTURE_INDEX, 1], "Camera 2 (Side Bill OCR & Texture)")
top_cam_stream = CameraStream([CAMERA_TOP_INDEX, 2, DEFAULT_RTSP], "Camera 3 (Top Banner & Corner Label)")


def init_cameras():
    qr_cam_stream.start()
    bill_cam_stream.start()
    top_cam_stream.start()

