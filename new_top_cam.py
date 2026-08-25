#!/usr/bin/env python3
"""
================================================================================
FILE: new_top_cam.py
================================================================================
DESCRIPTION:
    Gradient-based Mattress & ROI Segmentation Script for Top Camera System.
    Uses Image Spatial Gradients (Sobel Operators / Gradient Magnitude) instead
    of rigid HSV color ranges to differentiate foreground mattress from the
    background.

WHY GRADIENT METHOD?
    - Color-based (HSV) segmentation fails when lighting changes or when background
      colors overlap with mattress fabric (e.g. white cardboard vs white mattress).
    - Gradient methods compute spatial intensity variations (edges/boundaries).
      Mattress boundaries present strong gradient peaks regardless of color.

PIPELINE:
    1. Extract ROI (or full frame) from video stream / camera.
    2. Convert to Grayscale & apply edge-preserving noise reduction (Gaussian filter).
    3. Compute Horizontal (Gx) and Vertical (Gy) Sobel Gradients.
    4. Calculate Gradient Magnitude M = sqrt(Gx^2 + Gy^2) normalized to [0, 255].
    5. Threshold Gradient Magnitude (Otsu Thresholding) to extract edge boundary map.
    6. Apply Morphological Closing (dilation then erosion) to bridge edge gaps.
    7. Find outer contours, select largest contour, and fill polygon to build solid binary mask.
    8. Bitwise-AND mask with original ROI to produce clean segmented mattress foreground.
================================================================================
"""

import cv2
import numpy as np
import time
import os

# Force TCP transport for RTSP stream stability (essential for CCTV IP cameras)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def compute_gradient_magnitude(roi):
    """
    Computes spatial gradient magnitude of an input ROI using Sobel operators.
    
    Args:
        roi (numpy.ndarray): Input BGR image slice.
        
    Returns:
        gradient_mag (numpy.ndarray): 8-bit normalized gradient magnitude map (0-255).
        gray (numpy.ndarray): Grayscale blurred image.
    """
    # 1. Convert ROI to Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian Blur to smooth high-frequency noise while retaining major edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Compute Sobel gradients in X and Y directions (64-bit float precision)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 4. Calculate Gradient Magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # 5. Normalize magnitude map to 8-bit unsigned integer (0 - 255)
    gradient_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return gradient_mag, gray


def segment_mattress_gradient(roi, min_area=5000):
    """
    Segments the mattress ROI using the Gradient Method.
    
    Args:
        roi (numpy.ndarray): BGR image slice of ROI.
        min_area (int): Minimum contour area to filter noise.
        
    Returns:
        segmented_roi (numpy.ndarray): Masked ROI displaying isolated mattress foreground.
        gradient_mag (numpy.ndarray): Normalized gradient magnitude map.
        solid_mask (numpy.ndarray): Binary solid mask of detected mattress.
        contour_info (dict): Bounding rectangle information and dimensions.
    """
    # Step 1: Compute Gradient Magnitude
    gradient_mag, _ = compute_gradient_magnitude(roi)
    
    # Step 2: Threshold the Gradient Magnitude map (Otsu's auto threshold)
    _, binary_edges = cv2.threshold(gradient_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Morphological Operations to connect boundary edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    closed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Step 4: Find outer contours from edge map
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = roi.shape[:2]
    solid_mask = np.zeros((h, w), dtype=np.uint8)
    contour_info = None
    
    if contours:
        # Find largest external contour (assumed mattress boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area >= min_area:
            # Create Convex Hull to ensure solid boundary without interior gaps
            hull = cv2.convexHull(largest_contour)
            
            # Fill the interior of contour/hull to create a solid binary mask
            cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
            # Extract oriented minimum area bounding box for dimensions & alignment
            rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.int32(box_points)
            
            center, (width_px, height_px), angle = rect
            contour_info = {
                'box_points': box_points,
                'width_px': round(width_px, 1),
                'height_px': round(height_px, 1),
                'angle': round(angle, 1),
                'area': area
            }
            
    # Step 5: Segment mattress foreground using bitwise-AND
    segmented_roi = cv2.bitwise_and(roi, roi, mask=solid_mask)
    
    return segmented_roi, gradient_mag, solid_mask, contour_info


def create_synthetic_frame():
    """Generates a synthetic frame with a mattress on background for demonstration/testing."""
    frame = np.full((720, 1280, 3), (80, 80, 80), dtype=np.uint8)  # Dark background floor
    
    # Draw background texture lines
    for y in range(0, 720, 40):
        cv2.line(frame, (0, y), (1280, y), (90, 90, 90), 1)
        
    # Draw simulated light-colored mattress (320, 180) to (960, 580)
    x1, y1, x2, y2 = 320, 180, 960, 580
    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 235, 240), -1)
    
    # Draw blue logo on mattress
    cv2.rectangle(frame, (x1 + 40, y1 + 40), (x1 + 180, y1 + 90), (180, 80, 20), -1)
    cv2.putText(frame, "MATTRESS", (x1 + 50, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Border pattern on mattress edge
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 210), 3)
    
    return frame, (x1, y1, x2, y2)


def main():
    print("=" * 65)
    print("GRADIENT-BASED MATTRESS SEGMENTATION SYSTEM (new_top_cam.py)")
    print("=" * 65)
    
    # Candidate RTSP URLs (Environment Variable -> Password Encoded -> Raw Password -> Local Webcam)
    env_url = os.environ.get("MATTRESS_RTSP_URL", None)
    candidates = []
    if env_url:
        candidates.append(env_url)
    candidates.extend([
        "rtsp://admin:Admin%4012345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
        0  # USB / integrated webcam
    ])
    
    cap = None
    connected_source = None
    for src in candidates:
        print(f"[INFO] Attempting connection to stream source: {src}")
        try:
            temp_cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
            # Short test read to verify live frames are received
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    connected_source = src
                    print(f"[SUCCESS] Connected live stream from: {src}\n")
                    break
            temp_cap.release()
        except Exception as err:
            print(f"[WARN] Connection to {src} failed: {err}")
    
    use_synthetic = cap is None or not cap.isOpened()
    if use_synthetic:
        print("[INFO] No active camera stream accessible. Falling back to DEMO synthetic mode.")
        print("[INFO] Press 'q' inside any window to exit.\n")

    while True:
        if use_synthetic:
            frame, bbox = create_synthetic_frame()
            x1, y1, x2, y2 = bbox
            time.sleep(0.03)  # ~30 FPS timing
        else:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed. Retrying...")
                time.sleep(0.1)
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)

        # 1. Extract Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2].copy()

        # 2. Run Gradient Segmentation Pipeline
        segmented_roi, gradient_mag, solid_mask, contour_info = segment_mattress_gradient(roi)

        # 3. Visualization overlays
        annotated_roi = roi.copy()
        if contour_info is not None:
            cv2.drawContours(annotated_roi, [contour_info['box_points']], 0, (0, 255, 0), 2)
            lbl = f"W: {contour_info['width_px']}px | H: {contour_info['height_px']}px"
            cv2.putText(annotated_roi, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display OpenCV Windows
        cv2.imshow("Original ROI with Box", annotated_roi)
        cv2.imshow("Gradient Magnitude (Sobel)", gradient_mag)
        cv2.imshow("Binary Gradient Mask", solid_mask)
        cv2.imshow("Segmented Mattress (Foreground)", segmented_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Execution stopped cleanly.")


if __name__ == "__main__":
    main()
