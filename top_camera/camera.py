"""
camera.py — Thread-safe live webcam capture module.

Launches a background thread to poll frames from cv2.VideoCapture continuously,
ensuring low latency (always returns the latest frame) and preventing UI blocking.
Supports dynamic camera index switching at runtime.
"""

import threading
import time
import cv2

class ThreadedCamera:
    """
    Grabs frames from a webcam index in a background thread.
    Exposes methods to read the latest frame, swap camera indices, and release resources.
    """
    def __init__(self, index=0, width=1920, height=1080, fps=30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.error_msg = ""

        # Initial camera connection attempt
        self._connect_camera()
        
        # Start background grab thread
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True, name="WebcamGrabThread")
        self.thread.start()

    def _connect_camera(self):
        """Safely opens the VideoCapture device and configures properties."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            print(f"[camera] Attempting to open webcam index {self.index}...")
            # Try default backend first, fallback to V4L2 on Linux if needed
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                # Let's try explicit CAP_V4L2 as a fallback
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
                
            if not self.cap.isOpened():
                self.error_msg = f"Failed to open camera index {self.index}"
                print(f"[camera] ERROR: {self.error_msg}")
                self.cap = None
                return False

            # Configure properties for sane bandwidth & frame rate
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimizes latency
            except Exception:
                pass
            
            self.error_msg = ""
            print(f"[camera] Successfully connected to camera {self.index} ({self.width}x{self.height} @ {self.fps}fps)")
            return True

    def _grab_loop(self):
        """Continuous frame grabber loop."""
        delay = 1.0 / self.fps
        while self.running:
            start_time = time.time()
            cap_to_read = None
            
            with self.lock:
                cap_to_read = self.cap

            if cap_to_read is not None:
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
                        self.error_msg = f"Exception in read: {str(e)}"
            else:
                # If camera is not connected, rest briefly and retry
                time.sleep(0.5)
                self._connect_camera()

            # Maintain frame rate timing
            elapsed = time.time() - start_time
            sleep_time = max(0.001, delay - elapsed)
            time.sleep(sleep_time)

    def read(self):
        """
        Returns the latest frame (numpy array) and any error message.
        Thread-safe.
        """
        with self.lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return frame_copy, self.error_msg

    def change_camera(self, new_index):
        """
        Dynamically changes the camera index.
        Re-establishes the connection without restarting the thread.
        """
        with self.lock:
            if self.index == new_index and self.cap is not None:
                return True
            self.index = new_index
            self.frame = None
        return self._connect_camera()

    def release(self):
        """Stops the grab thread and releases resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print("[camera] Camera resources released cleanly.")
