"""
step1_live_border_check.py — Step 1 Live Webcam Black Border Detection Check.

Opens your USB Webcam (Index 0) directly and renders real-time 4-edge
border detection overlays (Top, Right, Bottom, Left) with P1-P4 corner points.
"""

import os
import sys
import time
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dimension_engine import MattressDimensionEngine


def connect_webcam(cam_index=0):
    """Directly opens USB Webcam without RTSP network delays."""
    print(f"[live_border] Opening USB Webcam (Index {cam_index})...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[live_border] USB Webcam Index {cam_index} connected successfully!")
        return cap, f"USB Webcam (Index {cam_index})"

    print(f"[live_border] ERROR: Could not open USB Webcam Index {cam_index}.")
    return None, "None"


def main():
    cap, camera_type = connect_webcam(0)
    if cap is None:
        sys.exit(1)

    color_modes = ["black", "auto", "red", "yellow", "white", "green"]
    color_idx = 0

    engine = MattressDimensionEngine(border_color_mode=color_modes[color_idx])

    window_name = "Step 1 Live Black Border Detection Check"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print(" STEP 1 LIVE WEBCAM BLACK BORDER DETECTION RUNNING")
    print(f" Camera Source: {camera_type}")
    print(" Controls: 'c'=Cycle Color Mode | 's'=Save Snapshot | 'q'=Quit")
    print("=" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        active_color = color_modes[color_idx]
        engine.calibrator.border_color_mode = active_color

        # Run Step 1 Border Detection Overlay
        annotated, border_info = engine.detect_and_draw_border_overlay(frame)

        # Bottom info bar
        h_img = annotated.shape[0]
        cv2.rectangle(annotated, (0, h_img - 35), (annotated.shape[1], h_img), (0, 0, 0), -1)
        status_txt = f"CAM: {camera_type} | MODE: {active_color.upper()} | BORDER DETECTED: {border_info['detected']}"
        cv2.putText(annotated, status_txt, (15, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[live_border] Exiting live border check...")
            break
        elif key == ord('c'):
            color_idx = (color_idx + 1) % len(color_modes)
            print(f"[live_border] Switched color mode to: {color_modes[color_idx].upper()}")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(CURRENT_DIR, f"live_border_check_{timestamp}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"[live_border] Saved camera border snapshot: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
