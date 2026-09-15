import os
import time
import datetime

os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration & Ground Truth Constants
# -----------------------------------------------------------------------------
PIXELS_PER_CM = 5.3241

# Actual Ground Truth Dimensions
ACTUAL_W_CM = 110.0
ACTUAL_L_CM = 208.0

# Convert actual to inches (1 inch = 2.54 cm)
ACTUAL_W_IN = round(ACTUAL_W_CM / 2.54, 1)
ACTUAL_L_IN = round(ACTUAL_L_CM / 2.54, 1)

# Global application state 
app_state = {
    "frozen": False,
    "annotated": None,
    "mask": None,
    "dims": None  # Will store (dim_w_cm, dim_w_in, dim_l_cm, dim_l_in)
}

# -----------------------------------------------------------------------------
# UI & Interaction Handlers
# -----------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    """Listens for mouse clicks on the virtual buttons."""
    global app_state
    if event == cv2.EVENT_LBUTTONDOWN:
        # CAPTURE button bounds
        if 30 <= x <= 200 and 30 <= y <= 80:
            app_state["frozen"] = True
            print("📸 Frame Frozen! Displaying accuracy report.")
        
        # RESUME button bounds
        elif 220 <= x <= 390 and 30 <= y <= 80:
            app_state["frozen"] = False
            print("▶️ Live Feed Resumed.")


def draw_ui(img):
    """Draws interactive buttons, status text, and the accuracy report panel."""
    global app_state

    # 1. CAPTURE Button (Green)
    cv2.rectangle(img, (30, 30), (200, 80), (30, 180, 50), -1)
    cv2.rectangle(img, (30, 30), (200, 80), (255, 255, 255), 2)
    cv2.putText(img, "CAPTURE", (55, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 2. RESUME Button (Blue)
    cv2.rectangle(img, (220, 30), (390, 80), (180, 70, 30), -1)
    cv2.rectangle(img, (220, 30), (390, 80), (255, 255, 255), 2)
    cv2.putText(img, "RESUME", (255, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 3. Status Indicator
    if app_state["frozen"]:
        cv2.putText(img, "[FROZEN] - Press 'r' to resume", (30, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(img, "[LIVE] - Press Spacebar to capture", (30, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 4. Accuracy Report Panel (Shows ONLY when frame is frozen)
    if app_state["frozen"]:
        h, w = img.shape[:2]
        panel_x = w - 540
        panel_y = 30
        
        # Draw translucent black background for report
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 510, panel_y + 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + 510, panel_y + 200), (255, 255, 255), 2)
        
        if app_state["dims"] is not None:
            dim_w_cm, dim_w_in, dim_l_cm, dim_l_in = app_state["dims"]
            
            # 1. Calculate Absolute Error in inches
            error_w = abs(dim_w_in - ACTUAL_W_IN)
            error_l = abs(dim_l_in - ACTUAL_L_IN)
            
            # 2. Calculate Error Percentage: (Error / Actual) * 100
            err_pct_w = (error_w / ACTUAL_W_IN) * 100
            err_pct_l = (error_l / ACTUAL_L_IN) * 100
            
            # 3. Accuracy is 100% minus the Error Percentage
            acc_w = max(0.0, 100.0 - err_pct_w)
            acc_l = max(0.0, 100.0 - err_pct_l)
            
            # Line 1: Actual Dimensions
            cv2.putText(img, "ACTUAL:", (panel_x + 15, panel_y + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, f"W: {ACTUAL_W_CM}cm ({ACTUAL_W_IN}in) | L: {ACTUAL_L_CM}cm ({ACTUAL_L_IN}in)", 
                        (panel_x + 15, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Line 2: Measured Dimensions
            cv2.putText(img, "MEASURED:", (panel_x + 15, panel_y + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, f"W: {dim_w_cm}cm ({dim_w_in}in) | L: {dim_l_cm}cm ({dim_l_in}in)", 
                        (panel_x + 15, panel_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Line 3: Accuracy & Error
            cv2.putText(img, f"ACCURACY W: {acc_w:.1f}% (Err: {err_pct_w:.1f}%)", 
                        (panel_x + 15, panel_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(img, f"ACCURACY L: {acc_l:.1f}% (Err: {err_pct_l:.1f}%)", 
                        (panel_x + 15, panel_y + 188), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(img, "NO CARDBOARD DETECTED", (panel_x + 15, panel_y + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# -----------------------------------------------------------------------------
# Core Image Processing
# -----------------------------------------------------------------------------
def segment_cardboard_from_glass_and_floor(frame, floor_cutoff_ratio=0.15):
    """
    Isolates a cardboard sheet/box from a reflective glass background
    and foreground floor using CIELAB/HSV color thresholding and contour analysis.
    Returns: annotated_frame, mask, bounding_rect, and dimensions tuple
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 1. Ignore Foreground Floor
    analysis_mask = np.ones((h, w), dtype=np.uint8) * 255
    if floor_cutoff_ratio > 0:
        cutoff_y = int(h * (1.0 - floor_cutoff_ratio))
        analysis_mask[cutoff_y:, :] = 0

    # 2. Color Space Separation
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    _, _, b_chan = cv2.split(lab)
    _, lab_b_thresh = cv2.threshold(b_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 40, 40])
    upper_brown = np.array([35, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    cardboard_raw_mask = cv2.bitwise_and(lab_b_thresh, hsv_mask)
    cardboard_raw_mask = cv2.bitwise_and(cardboard_raw_mask, analysis_mask)

    # 3. Morphological Cleaning
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean_mask = cv2.morphologyEx(cardboard_raw_mask, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.dilate(clean_mask, kernel_close, iterations=1)

    # 4. Contour Geometry & Bounding
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return annotated, clean_mask, None, None

    cardboard_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cardboard_contour)

    if area < 30000:
        return annotated, clean_mask, None, None

    hull = cv2.convexHull(cardboard_contour)
    rot_rect = cv2.minAreaRect(hull)
    box_pts = np.int32(cv2.boxPoints(rot_rect))
    (cx, cy), (box_w, box_h), angle = rot_rect

    # 5. Calculate Measured Dimensions
    dim_w_px = min(box_w, box_h)
    dim_l_px = max(box_w, box_h)
    
    # Convert pixels to cm, then subtract 1 cm from the width
    raw_w_cm = dim_w_px / PIXELS_PER_CM
    dim_w_cm = round(max(0.0, raw_w_cm - 1.0), 1)
    
    dim_l_cm = round(dim_l_px / PIXELS_PER_CM, 1)

    # Convert the adjusted width to inches
    dim_w_in = round(dim_w_cm / 2.54, 1)
    dim_l_in = round(dim_l_cm / 2.54, 1)

    # Draw live bounding box on the target
    cv2.drawContours(annotated, [box_pts], 0, (0, 255, 0), 3)
    cv2.circle(annotated, (int(cx), int(cy)), 7, (0, 255, 255), -1)

    # Pack dimensions into a tuple to be used by the UI
    dims = (dim_w_cm, dim_w_in, dim_l_cm, dim_l_in)

    return annotated, clean_mask, rot_rect, dims


# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
def main():
    global app_state

    # -------------------------------------------------------------------------
    # Create absolute path to Desktop
    # -------------------------------------------------------------------------
    desktop_dir = os.path.expanduser("~/Desktop")
    os.makedirs(desktop_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # =========================================================================
    # HD CAMERA OPTIMIZATION
    # =========================================================================
    # 1. Force the camera hardware to use MJPG compression (Prevents USB bottleneck/lag)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    
    # 2. Set to true 1080p Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 3. Request 30 FPS from the camera (if supported)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # =========================================================================

    cv2.namedWindow("Cardboard Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Segmentation Mask", cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    # -------------------------------------------------------------------------
    # Setup Video Recorder (Full HD)
    # -------------------------------------------------------------------------
    frame_h, frame_w = frame.shape[:2]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save directly to Desktop
    video_filename = os.path.join(desktop_dir, f"session_record_{timestamp}.mp4")
    
    # Using 'mp4v' codec. We set it to ~20 fps which is usually a safe speed for 
    # Raspberry Pi real-time processing so the video doesn't play back in fast-forward.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_w, frame_h))
    
    print(f"🔴 REC: HD Automatic recording started -> {video_filename}")
    print(f"   Resolution confirmed: {frame_w}x{frame_h}")

    cv2.imshow("Cardboard Detection", frame)
    cv2.waitKey(1)
    cv2.setMouseCallback("Cardboard Detection", mouse_callback)

    print("=" * 60)
    print(" 🚀 INTERACTIVE CARDBOARD DETECTOR (HD)")
    print(" • Click 'CAPTURE' (or press SPACE) to freeze and see accuracy.")
    print(" • Click 'RESUME' (or press 'R') to go back to live feed.")
    print(" • Press 'S' to save a snapshot to Desktop.")
    print(" • Press 'Q' or ESC to exit and save the HD video.")
    print("=" * 60)

    while True:
        # Update pipeline only if NOT frozen
        if not app_state["frozen"]:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, mask, rect, dims = segment_cardboard_from_glass_and_floor(
                frame, 
                floor_cutoff_ratio=0.12 
            )
            
            app_state["annotated"] = annotated
            app_state["mask"] = mask
            app_state["dims"] = dims
        
        # Render the UI on top of the current state
        display_frame = app_state["annotated"].copy()
        draw_ui(display_frame)

        # Write the final Full HD frame to the MP4 file
        out_video.write(display_frame)

        cv2.imshow("Cardboard Detection", display_frame)
        cv2.imshow("Binary Segmentation Mask", app_state["mask"])

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key in [ord('c'), 32]:
            app_state["frozen"] = True
            print("📸 Frame Frozen! Accuracy report displayed.")
        elif key == ord('r'):
            app_state["frozen"] = False
            print("▶️ Live Feed Resumed.")
        elif key == ord('s'):
            snapshot_name = os.path.join(desktop_dir, f"snapshot_{datetime.datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(snapshot_name, display_frame)
            print(f"💾 Snapshot saved to: {snapshot_name}")

    # Cleanup resources
    print("Stopping HD recording and closing...")
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ HD Video saved successfully to: {video_filename}")


if __name__ == "__main__":
    main()
