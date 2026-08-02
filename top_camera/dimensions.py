"""
dimensions.py — Hybrid AI + OpenCV Mattress Dimension Checking Module.

Adapted to use standard OpenCV webcam frame buffers. Loads a local YOLOv8 model for
semantic target isolation, then uses classical contour mathematics to compute the
rotated bounding box width and height in centimeters.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# Model Initialization
# ==============================================================================
# Resolve the local path to yolov8n.pt for complete project self-containment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "yolov8n.pt")
_YOLO_MODEL = None

def get_model():
    """Lazy-loads the YOLO model to save memory if dimensions are not checked."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model weight file not found at local path: {MODEL_PATH}. "
                "Ensure yolov8n.pt has been copied to the top_camera folder."
            )
        print(f"[dimensions] Loading YOLOv8 model from {MODEL_PATH}...")
        _YOLO_MODEL = YOLO(MODEL_PATH)
    return _YOLO_MODEL


def measure_dimensions(img, pixels_per_cm=None, edge_correction=1.1, min_contour_area=50000, fixed_crop=None):
    """
    Measures the width and height of a mattress in the given BGR frame.
    
    Parameters:
        img (np.ndarray): The input BGR frame from the webcam.
        pixels_per_cm (float): Calibration ratio (pixels/cm). If None, dimension estimation is skipped.
        edge_correction (float): Dynamic scale factor to account for contour offsets.
        min_contour_area (int): Area threshold in pixels to filter out noise.
        fixed_crop (tuple): ROI crop window (y0, y1, x0, x1) to restrict analysis.
        
    Returns:
        width (float): Calculated width in cm (or None if uncalibrated/undetected).
        height (float): Calculated height in cm (or None if uncalibrated/undetected).
        annotated_img (np.ndarray): The frame with bounding box and dimension labels drawn.
    """
    if img is None or img.size == 0:
        return None, None, img

    # Work on a copy to avoid mutating the original stream frame
    annotated_img = img.copy()
    h_orig, w_orig = img.shape[:2]

    try:
        # Determine analysis region (crop if provided)
        if fixed_crop:
            y0, y1, x0, x1 = fixed_crop
            # Safe bounding checks
            y0, y1 = max(0, y0), min(h_orig, y1)
            x0, x1 = max(0, x0), min(w_orig, x1)
            crop_img = img[y0:y1, x0:x1]
        else:
            y0, y1, x0, x1 = 0, h_orig, 0, w_orig
            crop_img = img

        if crop_img.size == 0:
            print("[dimensions] ERROR: Crop region has zero area.")
            return None, None, annotated_img

        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        crop_area = crop_img.shape[0] * crop_img.shape[1]

        # ----------------------------------------------------------------------
        # STAGE 1: YOLO AI Target Isolation & Bounding Box Extraction
        # ----------------------------------------------------------------------
        model = get_model()
        yolo_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)

        results = model.predict(source=crop_img, save=False, conf=0.15, verbose=False)
        valid_box = None
        max_box_area = 0

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                # Skip human operator detections (Class 0)
                if class_id == 0:
                    continue

                x1_b, y1_b, x2_b, y2_b = box.xyxy[0].cpu().numpy()
                area = (x2_b - x1_b) * (y2_b - y1_b)

                if area > max_box_area:
                    max_box_area = area
                    valid_box = box

        if valid_box is not None:
            x1_b, y1_b, x2_b, y2_b = valid_box.xyxy[0].cpu().numpy()
            pad = 25
            x1_pad = max(0, int(x1_b) - pad)
            y1_pad = max(0, int(y1_b) - pad)
            x2_pad = min(crop_img.shape[1], int(x2_b) + pad)
            y2_pad = min(crop_img.shape[0], int(y2_b) + pad)

            cv2.rectangle(yolo_mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 255, -1)
        else:
            # Fallback to full-frame mask if no YOLO object was detected
            yolo_mask[:] = 255

        # ----------------------------------------------------------------------
        # STAGE 2: High-Contrast Edge Detection (Canny + Adaptive Morph)
        # ----------------------------------------------------------------------
        crop_blur = cv2.GaussianBlur(crop_gray, (7, 7), 0)
        
        # Combine Canny edge detection + Otsu thresholding for multi-object versatility
        edges = cv2.Canny(crop_blur, 30, 120)
        _, thresh = cv2.threshold(crop_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        combined_binary = cv2.bitwise_or(edges, thresh)
        
        # Morphological dilation to close gaps in outer object boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        crop_clean = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
        crop_clean = cv2.dilate(crop_clean, kernel, iterations=1)

        # ----------------------------------------------------------------------
        # STAGE 3: Mask Merging and Convex Hull Geometry Estimation
        # ----------------------------------------------------------------------
        final_mask = cv2.bitwise_and(crop_clean, crop_clean, mask=yolo_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[dimensions] No contours found inside mask.")
            return None, None, annotated_img

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        roi_found = False
        min_area_thresh = min(min_contour_area, 2000)

        for c in contours:
            area = cv2.contourArea(c)

            # Area must be within reasonable bounds (not tiny noise, not full table background)
            if min_area_thresh < area < (crop_area * 0.85):
                # Smooth outer boundary using Convex Hull
                hull = cv2.convexHull(c)
                
                # Compute minimum area rotated bounding box
                rect = cv2.minAreaRect(hull)
                (cx, cy), (w, h), angle = rect

                if w < 5 or h < 5:
                    continue

                # Project coordinates back onto original frame system
                cx_orig = cx + x0
                cy_orig = cy + y0
                
                # Draw the red convex hull contour
                hull_offset = hull + np.array([x0, y0])
                cv2.drawContours(annotated_img, [hull_offset], -1, (0, 0, 255), 2)

                # Draw the green rotated bounding box
                box_pts = cv2.boxPoints(rect)
                box_pts_offset = np.int32(box_pts + np.array([x0, y0]))
                cv2.drawContours(annotated_img, [box_pts_offset], 0, (0, 255, 0), 3)

                # Render yellow center point
                cv2.circle(annotated_img, (int(cx_orig), int(cy_orig)), 6, (0, 255, 255), -1)

                # If uncalibrated, return raw pixel dimensions
                if pixels_per_cm is None or pixels_per_cm <= 0:
                    warn_text = "UNCALIBRATED: Showing pixels"
                    cv2.putText(annotated_img, warn_text, (int(cx_orig) - 140, int(cy_orig) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(annotated_img, f"W: {round(w,1)} px", (int(cx_orig) - 100, int(cy_orig)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(annotated_img, f"L: {round(h,1)} px", (int(cx_orig) - 100, int(cy_orig) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    return w, h, annotated_img

                # Calibrate dimension math: convert pixels to cm & inches
                raw_w = w / pixels_per_cm
                raw_h = h / pixels_per_cm

                nW = round(raw_w * edge_correction, 1)
                nH = round(raw_h * edge_correction, 1)

                nW_in = round(nW / 2.54, 1)
                nH_in = round(nH / 2.54, 1)

                # Render text tags on frame
                cv2.putText(annotated_img, f'W: {nW} cm ({nW_in} in)', (int(cx_orig) - 120, int(cy_orig) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2)
                cv2.putText(annotated_img, f'L: {nH} cm ({nH_in} in)', (int(cx_orig) - 120, int(cy_orig) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2)

                roi_found = True
                return nW, nH, annotated_img

        if not roi_found:
            print("[dimensions] No geometric contour matched constraints.")
            
    except Exception as e:
        print(f"[dimensions] Error during processing: {str(e)}")

    return None, None, annotated_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dimensions.py individually")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file to test (optional)")
    parser.add_argument("--save", type=str, default="dimensions_result.jpg", help="Output path for annotated image")
    parser.add_argument("--pixels-per-cm", type=float, default=10.0, help="Calibration pixels/cm (default: 10.0)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" 📐 DIMENSIONS.PY INDIVIDUAL TEST")
    print("="*60)

    frame = None
    if args.image and os.path.exists(args.image):
        print(f"[test] Loading image from disk: {args.image}")
        frame = cv2.imread(args.image)
    else:
        print("[test] Grabbing live frame from Picamera2...")
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (1456, 1088), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            import time; time.sleep(1.5)
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            picam.stop()
            picam.close()
        except Exception as e:
            print(f"[test] Picamera2 failed: {e}. Trying OpenCV V4L2 fallback...")
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

    if frame is None:
        print("❌ ERROR: Could not get frame from camera or image file.")
        exit(1)

    print(f"[test] Input frame shape: {frame.shape}")
    print("[test] Measuring dimensions...")

    w, h, annotated = measure_dimensions(frame, pixels_per_cm=args.pixels_per_cm)

    print("-" * 40)
    if w is not None and h is not None:
        print(f"✅ MEASUREMENT SUCCESSFUL:")
        print(f"   Width:  {w} cm (or px if uncalibrated)")
        print(f"   Height: {h} cm (or px if uncalibrated)")
    else:
        print("⚠️ No valid object contour detected in frame.")

    cv2.imwrite(args.save, annotated)
    print(f"📸 Saved annotated output image to: {os.path.abspath(args.save)}")
    print("="*60 + "\n")

