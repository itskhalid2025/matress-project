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
        # STAGE 1: YOLO AI Detection Masking
        # ----------------------------------------------------------------------
        model = get_model()
        yolo_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)

        # Run inference (conf=0.15 matches legacy configuration)
        results = model.predict(source=crop_img, save=False, conf=0.15, verbose=False)
        valid_box = None
        max_box_area = 0

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                # Skip human detections (Class 0) so operators don't corrupt the ROI mask
                if class_id == 0:
                    continue

                x1_box, y1_box, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                area = (x2_box - x1_box) * (y2_box - y1_box)

                if area > max_box_area:
                    max_box_area = area
                    valid_box = box

        if valid_box is not None:
            x1_box, y1_box, x2_box, y2_box = valid_box.xyxy[0].cpu().numpy()
            
            # Pad the bounding box slightly outward to ensure we don't clip the mattress edges
            pad = 40
            x1_pad = max(0, int(x1_box) - pad)
            y1_pad = max(0, int(y1_box) - pad)
            x2_pad = min(crop_img.shape[1], int(x2_box) + pad)
            y2_pad = min(crop_img.shape[0], int(y2_box) + pad)

            cv2.rectangle(yolo_mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 255, -1)
        else:
            # Fallback to full-frame if YOLO doesn't detect any object
            yolo_mask[:] = 255

        # ----------------------------------------------------------------------
        # STAGE 2: Classical OpenCV Geometry
        # ----------------------------------------------------------------------
        crop_blur = cv2.GaussianBlur(crop_gray, (9, 9), 0)
        
        # Otsu thresholding handles variable conveyor lighting dynamically
        _, crop_thresh = cv2.threshold(crop_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological clean up to remove stitches/texture lines
        kernel = np.ones((5, 5), np.uint8)
        crop_clean = cv2.erode(crop_thresh, kernel, iterations=1)
        crop_clean = cv2.dilate(crop_clean, kernel, iterations=2)

        # ----------------------------------------------------------------------
        # STAGE 3: Mask Merging and Convex Hull Estimation
        # ----------------------------------------------------------------------
        final_mask = cv2.bitwise_and(crop_clean, crop_clean, mask=yolo_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[dimensions] No contours found inside mask.")
            return None, None, annotated_img

        # Sort contours by area to inspect the largest ones first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        roi_found = False

        for c in contours:
            area = cv2.contourArea(c)

            # Area must be significant and not take up the entire crop area (prevent background bleed)
            if min_contour_area < area < (crop_area * 0.90):
                # Smooth the shape outline using a Convex Hull
                hull = cv2.convexHull(c)
                
                # Get the minimum area rotated bounding box
                rect = cv2.minAreaRect(hull)
                (cx, cy), (w, h), angle = rect

                if w == 0 or h == 0:
                    continue

                # Project coordinates back onto the original uncropped frame coordinate system
                cx_orig = cx + x0
                cy_orig = cy + y0
                
                # Draw the convex hull contour
                hull_offset = hull + np.array([x0, y0])
                cv2.drawContours(annotated_img, [hull_offset], -1, (0, 0, 255), 2)

                # Draw the rotated bounding box
                box_pts = cv2.boxPoints(rect)
                box_pts_offset = np.int32(box_pts + np.array([x0, y0]))
                cv2.drawContours(annotated_img, [box_pts_offset], 0, (0, 255, 0), 3)

                # Render a center indicator dot
                cv2.circle(annotated_img, (int(cx_orig), int(cy_orig)), 5, (0, 255, 255), -1)

                # If calibration is missing, render warning guidelines
                if pixels_per_cm is None or pixels_per_cm <= 0:
                    warn_text = "UNCALIBRATED: Size showing in pixels"
                    cv2.putText(annotated_img, warn_text, (int(cx_orig) - 180, int(cy_orig) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(annotated_img, f"Width: {round(w,1)} px", (int(cx_orig) - 100, int(cy_orig)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(annotated_img, f"Length: {round(h,1)} px", (int(cx_orig) - 100, int(cy_orig) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    return w, h, annotated_img

                # Calibrate dimension math: convert pixels to cm
                raw_w = w / pixels_per_cm
                raw_h = h / pixels_per_cm

                # Apply edge correction factor and round to 1 decimal place
                nW = round(raw_w * edge_correction, 1)
                nH = round(raw_h * edge_correction, 1)

                # Draw width and height tags
                cv2.putText(annotated_img, f'Width: {nW} cm', (int(cx_orig) - 100, int(cy_orig) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.putText(annotated_img, f'Length: {nH} cm', (int(cx_orig) - 100, int(cy_orig) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                roi_found = True
                return nW, nH, annotated_img

        if not roi_found:
            print("[dimensions] No geometric contour matched constraints.")
            
    except Exception as e:
        print(f"[dimensions] Error during processing: {str(e)}")

    return None, None, annotated_img
