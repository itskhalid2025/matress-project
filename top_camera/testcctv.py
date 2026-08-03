import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================================
# Force TCP transport for RTSP CCTV camera stream
# ==========================================================
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RTSP_URL = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
)

# ==========================================================
# Load trained YOLO model
# ==========================================================
MODEL_PATH = r"C:\matress-project-matress\top_camera\bestdimension.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "bestdimension.pt")

print(f"[testcctv] Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_HALF = torch.cuda.is_available()

# ==========================================================
# Initialize CCTV / RTSP Camera with USB fallback
# ==========================================================
print(f"[testcctv] Connecting to CCTV RTSP stream: {RTSP_URL}...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
camera_type = "RTSP CCTV"

if cap.isOpened():
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[testcctv] RTSP CCTV connected successfully.")
else:
    print("[testcctv] RTSP stream failed. Falling back to USB webcam (Index 0)...")
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera_type = "USB Cam (Index 0)"
        print("[testcctv] Connected to USB webcam.")
    else:
        print("[testcctv] ERROR: Could not open CCTV stream or USB webcam.")
        sys.exit(1)

print("\nPress 'q' to quit.")

# ==========================================================
# Live CCTV Detection Loop
# ==========================================================
fps_counter = 0
current_fps = 0.0
last_fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[testcctv] Frame read failed. Retrying...")
        time.sleep(0.01)
        continue

    # Calculate FPS
    fps_counter += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        current_fps = round(fps_counter / (now - last_fps_time), 1)
        fps_counter = 0
        last_fps_time = now

    # Run YOLO inference on the full frame
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.10,
        device=YOLO_DEVICE,
        half=YOLO_HALF,
        verbose=False
    )

    # Plot bounding boxes
    if len(results) > 0:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()

    # Overlay FPS & Camera Info
    info_txt = f"FPS: {current_fps} | Camera: {camera_type}"
    cv2.putText(annotated_frame, info_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(annotated_frame, info_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("CCTV Mattress Detection - YOLO", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()