"""
camera.py — Thread-safe side camera capture module.

Supports both pypylon-driven Basler USB3 industrial cameras and standard USB webcams
using OpenCV as a fallback. Runs a background loop to continuously fetch the latest
frames and minimize capture latency.
"""

import threading
import time
import cv2
import numpy as np

try:
    from pypylon import pylon
    HAS_PYPYLON = True
except ImportError:
    HAS_PYPYLON = False


class ThreadedCamera:
    """
    Continuous frame grabber that supports:
      - Basler USB3 cameras (via pypylon SDK)
      - Standard USB webcams (via OpenCV VideoCapture)
    Always yields the latest frame in a thread-safe manner.
    """
    def __init__(self, index=0, serial=None, width=1920, height=1080, fps=30):
        self.index = index
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.basler_cam = None
        self.basler_converter = None
        self.use_basler = False
        self.frame = None
        self.running = False
        self.error_msg = ""
        self.lock = threading.Lock()
        self.thread = None

        # Attempt to establish camera connection
        self._connect_camera()

        # Start background grab loop
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True, name="SideCameraGrabThread")
        self.thread.start()

    def _connect_camera(self):
        """Safely configures and opens either a Basler or USB webcam device."""
        with self.lock:
            # Safely release any existing handles
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.basler_cam is not None:
                try:
                    if self.basler_cam.IsOpen():
                        self.basler_cam.Close()
                except Exception:
                    pass
                self.basler_cam = None
                self.basler_converter = None
                self.use_basler = False

            # 1. Try connecting via Basler/pypylon if available
            if HAS_PYPYLON:
                try:
                    print("[camera] Attempting to find Basler Side Camera...")
                    tl_factory = pylon.TlFactory.GetInstance()
                    if self.serial:
                        devices = tl_factory.EnumerateDevices()
                        matches = [d for d in devices if d.GetSerialNumber() == self.serial]
                        if not matches:
                            raise RuntimeError(f"Basler camera with serial {self.serial} not found.")
                        p_device = tl_factory.CreateDevice(matches[0])
                    else:
                        p_device = tl_factory.CreateFirstDevice()

                    self.basler_cam = pylon.InstantCamera(p_device)
                    self.basler_cam.Open()

                    self.basler_converter = pylon.ImageFormatConverter()
                    self.basler_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                    self.basler_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                    self.use_basler = True
                    self.error_msg = ""
                    print("[camera] Successfully connected to Basler Side Camera!")
                    return True
                except Exception as e:
                    print(f"[camera] Basler initialization failed: {str(e)}. Falling back to standard webcam.")
                    self.use_basler = False
                    self.basler_cam = None

            # 2. Fallback to standard OpenCV webcam capture
            print(f"[camera] Attempting to open webcam index {self.index}...")
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                # Fallback to explicit CAP_V4L2 for Linux compatibility
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                self.error_msg = f"Failed to open webcam index {self.index}"
                print(f"[camera] ERROR: {self.error_msg}")
                self.cap = None
                return False

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.error_msg = ""
            print(f"[camera] Successfully connected to standard webcam index {self.index} ({self.width}x{self.height} @ {self.fps}fps)")
            return True

    def _grab_loop(self):
        """Infinite grabber loop operating in the background."""
        delay = 1.0 / self.fps
        while self.running:
            start_time = time.time()
            
            with self.lock:
                use_basler = self.use_basler
                basler_cam = self.basler_cam
                cap_to_read = self.cap

            if use_basler and basler_cam is not None:
                try:
                    # Basler single shot grab (timeout 2000ms)
                    result = basler_cam.GrabOne(2000)
                    if result.GrabSucceeded():
                        converted = self.basler_converter.Convert(result)
                        frame = converted.GetArray()
                        with self.lock:
                            self.frame = frame
                            self.error_msg = ""
                    else:
                        with self.lock:
                            self.error_msg = "Basler grab result failed"
                    result.Release()
                except Exception as e:
                    with self.lock:
                        self.error_msg = f"Basler grab exception: {str(e)}"
            elif cap_to_read is not None:
                try:
                    ok, frame = cap_to_read.read()
                    if ok and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.error_msg = ""
                    else:
                        with self.lock:
                            self.error_msg = f"Failed to read frame from index {self.index}"
                except Exception as e:
                    with self.lock:
                        self.error_msg = f"Webcam exception: {str(e)}"
            else:
                # If neither is available, sleep and retry connection
                time.sleep(0.5)
                self._connect_camera()

            # Maintain constant frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0.01, delay - elapsed)
            time.sleep(sleep_time)

    def read(self):
        """Returns the latest captured frame and current error state."""
        with self.lock:
            return self.frame, self.error_msg

    def change_camera(self, new_idx, serial=None):
        """Swaps parameters and resets the connection."""
        self.index = new_idx
        self.serial = serial
        return self._connect_camera()

    def release(self):
        """Terminates thread and releases all camera bindings."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.basler_cam is not None:
                try:
                    if self.basler_cam.IsOpen():
                        self.basler_cam.Close()
                except Exception:
                    pass
                self.basler_cam = None
                self.basler_converter = None
