"""
live_camera_dimension.py — Guided 3-State Live Camera Dimension System.

3-State Operator Workflow:
  STATE 1: BORDER CALIBRATION — Detects border & inputs physical border width/length one-by-one.
  STATE 2: WAITING FOR MATTRESS — Prompts operator: "TABLE CALIBRATED! Place Mattress Inside."
  STATE 3: MEASURING MATTRESS — Calculates & displays exact mattress dimensions in cm & inches.

Interactive Controls:
  [SPACE] / [ENTER] — Confirm current state & advance to next phase
  'w' — Set Reference Border Width (cm)
  'l' — Set Reference Border Length (cm)
  'c' — Cycle Border Color ('red' -> 'yellow' -> 'black' -> 'white' -> 'green' -> 'auto')
  'r' — Reset & measure next mattress
  'b' — Re-calibrate border corners
  'd' — Toggle 4-quadrant multi-channel debug window
  's' — Save high-res snapshot & JSON measurement report
  'q' — Quit
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dimension_engine import MattressDimensionEngine

# Force TCP transport for RTSP CCTV IP cameras
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Default CCTV RTSP camera URL from project settings
DEFAULT_RTSP_URL = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
)

# Workflow State Constants
STATE_BORDER_CALIBRATION = 1
STATE_WAITING_FOR_MATTRESS = 2
STATE_MEASURING_MATTRESS = 3


def connect_camera(rtsp_url=None, cam_index=0):
    """Connects to RTSP CCTV stream with fallback to local USB webcam."""
    url = rtsp_url or DEFAULT_RTSP_URL
    print(f"[live_camera] Attempting RTSP CCTV connection: {url}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[live_camera] RTSP CCTV stream connected successfully.")
        return cap, "RTSP CCTV"

    print(f"[live_camera] RTSP failed. Falling back to USB Webcam (Index {cam_index})...")
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
    cap = cv2.VideoCapture(cam_index, backend)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[live_camera] Connected to USB Webcam Index {cam_index}.")
        return cap, f"USB Cam (Index {cam_index})"

    print("[live_camera] ERROR: Could not open RTSP stream or USB webcam.")
    return None, "None"


def draw_state_overlay(img, state, ref_w_cm, ref_h_cm, border_color_mode, res=None):
    """Renders visual status banners according to current operational phase."""
    h_img, w_img = img.shape[:2]
    annotated = img.copy()

    # Top Banner Header
    cv2.rectangle(annotated, (0, 0), (w_img, 45), (20, 20, 20), -1)

    if state == STATE_BORDER_CALIBRATION:
        header_txt = f"STATE 1: BORDER SETUP | Tape: {border_color_mode.upper()} | Input: W={ref_w_cm} cm, L={ref_h_cm} cm"
        cv2.putText(annotated, header_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        # Guidance Banner
        cv2.rectangle(annotated, (0, h_img - 70), (w_img, h_img), (0, 0, 0), -1)
        guide_txt = "Press 'w' to edit Width | Press 'l' to edit Length | Press SPACE to Lock Calibration"
        cv2.putText(annotated, guide_txt, (20, h_img - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    elif state == STATE_WAITING_FOR_MATTRESS:
        header_txt = f"STATE 2: TABLE READY | Ref Frame Locked: {ref_w_cm} cm x {ref_h_cm} cm"
        cv2.putText(annotated, header_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Large Prompt Box in Center
        cx, cy = w_img // 2, h_img // 2
        cv2.rectangle(annotated, (cx - 380, cy - 65), (cx + 380, cy + 65), (30, 30, 30), -1)
        cv2.rectangle(annotated, (cx - 380, cy - 65), (cx + 380, cy + 65), (0, 255, 255), 3)

        p1_txt = "TABLE CALIBRATION LOCKED (" + str(ref_w_cm) + " cm x " + str(ref_h_cm) + " cm)"
        p2_txt = "PLEASE PLACE MATTRESS INSIDE THE REFERENCE BORDER"
        p3_txt = "[ Press SPACE when Mattress is Ready ]"

        cv2.putText(annotated, p1_txt, (cx - 340, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated, p2_txt, (cx - 355, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, p3_txt, (cx - 240, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        # Guidance Banner
        cv2.rectangle(annotated, (0, h_img - 40), (w_img, h_img), (0, 0, 0), -1)
        cv2.putText(annotated, "Press 'b' to re-calibrate border | Press SPACE when mattress is loaded",
                    (20, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    elif state == STATE_MEASURING_MATTRESS:
        header_txt = f"STATE 3: DIMENSION MEASUREMENT COMPLETE | Ref: {ref_w_cm}x{ref_h_cm} cm"
        cv2.putText(annotated, header_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Bottom Guidance Banner
        cv2.rectangle(annotated, (0, h_img - 45), (w_img, h_img), (0, 0, 0), -1)
        cv2.putText(annotated, "Press 'r' to measure NEXT mattress | Press 's' to save report | Press 'b' to reset border",
                    (20, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def prompt_user_input(window_name, current_val, prompt_title):
    """Simple GUI prompt using OpenCV text dialog or console fallback."""
    print(f"\n[INPUT PROMPT] {prompt_title} (Current: {current_val} cm):")
    try:
        new_val_str = input(f"Enter new value in cm (or press ENTER to keep {current_val}): ").strip()
        if new_val_str:
            return float(new_val_str)
    except Exception as e:
        print(f"Invalid input ({e}), keeping {current_val} cm.")
    return float(current_val)


def main():
    parser = argparse.ArgumentParser(description="Guided 3-State Mattress Dimension System")
    parser.add_argument("--rtsp", type=str, default=DEFAULT_RTSP_URL, help="RTSP CCTV Camera Stream URL")
    parser.add_argument("--cam-index", type=int, default=0, help="USB Webcam index (default 0)")
    parser.add_argument("--ref-width", type=float, default=100.0, help="Physical reference border width in cm")
    parser.add_argument("--ref-height", type=float, default=120.0, help="Physical reference border height in cm")
    parser.add_argument("--color", type=str, default="red", help="Border color ('red', 'yellow', 'black', 'white', 'green', 'auto')")
    args = parser.parse_args()

    cap, camera_type = connect_camera(args.rtsp, args.cam_index)
    if cap is None:
        sys.exit(1)

    color_modes = ["red", "yellow", "black", "white", "green", "auto"]
    current_color_idx = color_modes.index(args.color.lower()) if args.color.lower() in color_modes else 0

    ref_w_cm = float(args.ref_width)
    ref_h_cm = float(args.ref_height)

    engine = MattressDimensionEngine(
        ref_width_cm=ref_w_cm,
        ref_height_cm=ref_h_cm,
        border_color_mode=color_modes[current_color_idx]
    )

    current_state = STATE_BORDER_CALIBRATION
    show_debug_window = False
    last_res = None

    window_name = "Guided Mattress Dimensioning System"
    debug_window_name = "Multi-Channel Filter Debug Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print(" GUIDED 3-STATE MATTRESS DIMENSION SYSTEM INITIALIZED")
    print("=" * 70)
    print(" STATE 1: BORDER CALIBRATION — Detect border & input metric size")
    print(" STATE 2: WAITING FOR MATTRESS — Prompt: Place mattress inside frame")
    print(" STATE 3: MEASURING MATTRESS — Auto-calculate width, length, area")
    print("=" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        active_color = color_modes[current_color_idx]

        if current_state == STATE_BORDER_CALIBRATION:
            # Detect reference border & warp view
            try:
                warped, corners, scale_x, scale_y, M = engine.calibrator.calibrate_and_warp(frame)
                annotated_display = frame.copy()
                if corners is not None:
                    pts = np.int32(corners)
                    cv2.polylines(annotated_display, [pts], isClosed=True, color=(0, 215, 255), thickness=3)
                    for p in pts:
                        cv2.circle(annotated_display, tuple(p), 7, (0, 255, 0), -1)
                annotated_display = draw_state_overlay(annotated_display, current_state, ref_w_cm, ref_h_cm, active_color)
            except Exception as e:
                annotated_display = frame.copy()
                cv2.putText(annotated_display, f"SEARCHING BORDER ({active_color.upper()}): {e}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                annotated_display = draw_state_overlay(annotated_display, current_state, ref_w_cm, ref_h_cm, active_color)

        elif current_state == STATE_WAITING_FOR_MATTRESS:
            annotated_display = draw_state_overlay(frame, current_state, ref_w_cm, ref_h_cm, active_color)

        elif current_state == STATE_MEASURING_MATTRESS:
            last_res, annotated_warped, debug_grid = engine.process_frame(
                frame,
                ref_width_cm=ref_w_cm,
                ref_height_cm=ref_h_cm,
                border_color_mode=active_color
            )
            annotated_display = draw_state_overlay(annotated_warped, current_state, ref_w_cm, ref_h_cm, active_color, last_res)

            if show_debug_window and debug_grid is not None:
                cv2.imshow(debug_window_name, debug_grid)

        cv2.imshow(window_name, annotated_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[live_camera] Exiting live dimension system...")
            break
        elif key == 32 or key == 13:  # SPACE or ENTER key
            if current_state == STATE_BORDER_CALIBRATION:
                current_state = STATE_WAITING_FOR_MATTRESS
                print(f"[WORKFLOW] Border Calibration LOCKED ({ref_w_cm} cm x {ref_h_cm} cm). Advancing to STATE 2 (Place Mattress Prompt).")
            elif current_state == STATE_WAITING_FOR_MATTRESS:
                current_state = STATE_MEASURING_MATTRESS
                print("[WORKFLOW] Mattress loaded. Advancing to STATE 3 (Dimension Calculation).")
        elif key == ord('w'):
            ref_w_cm = prompt_user_input(window_name, ref_w_cm, "Physical Border Width (cm)")
            engine.calibrator.update_reference_dimensions(ref_w_cm, ref_h_cm)
            print(f"[CALIBRATION] Reference Width updated to: {ref_w_cm} cm")
        elif key == ord('l'):
            ref_h_cm = prompt_user_input(window_name, ref_h_cm, "Physical Border Length (cm)")
            engine.calibrator.update_reference_dimensions(ref_w_cm, ref_h_cm)
            print(f"[CALIBRATION] Reference Length updated to: {ref_h_cm} cm")
        elif key == ord('c'):
            current_color_idx = (current_color_idx + 1) % len(color_modes)
            print(f"[live_camera] Border color mode set to: {color_modes[current_color_idx].upper()}")
        elif key == ord('r'):
            current_state = STATE_WAITING_FOR_MATTRESS
            print("[WORKFLOW] Resetting to STATE 2. Ready to measure NEXT mattress!")
        elif key == ord('b'):
            current_state = STATE_BORDER_CALIBRATION
            print("[WORKFLOW] Resetting to STATE 1. Ready to re-calibrate Border Corners!")
        elif key == ord('d'):
            show_debug_window = not show_debug_window
            if not show_debug_window and cv2.getWindowProperty(debug_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(debug_window_name)
            elif show_debug_window:
                cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
            print(f"[live_camera] Debug grid window: {'ENABLED' if show_debug_window else 'DISABLED'}")
        elif key == ord('s') and last_res:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_img = os.path.join(CURRENT_DIR, f"mattress_report_{timestamp}.jpg")
            save_json = os.path.join(CURRENT_DIR, f"mattress_report_{timestamp}.json")
            cv2.imwrite(save_img, annotated_display)
            with open(save_json, "w") as f:
                json.dump({"timestamp": timestamp, "results": last_res}, f, indent=2)
            print(f"[REPORT] Saved visual snapshot: {save_img}")
            print(f"[REPORT] Saved JSON metrics : {save_json}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
