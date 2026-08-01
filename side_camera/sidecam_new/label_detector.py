"""
label_detector.py — Robust Hybrid YOLO11n & Computer Vision Label Localization Module.

Pipeline:
  1. Safe loading of YOLO11n (best.pt) model.
  2. Bounding Box detection with confidence threshold.
  3. Hybrid Rectification: Crop detected region with margin, apply HSV/LAB thresholding 
     and perspective transform to generate a flat, deskewed label image.
  4. Dual Fallback:
     - Fallback A: Use the simple YOLO rectangular crop (rotated 90° if vertical).
     - Fallback B: If YOLO fails to detect any labels, fall back to pure classic CV detection 
       across the entire frame.
  5. Premium Bounding Box Annotation drawing helper.
"""

import os
import cv2
import numpy as np

# Safe import of ultralytics for YOLO inference
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Global reference to cache loaded model instance
_yolo_model = None


def get_yolo_model():
    """Lazy loads and caches the YOLO model from local best.pt."""
    global _yolo_model
    if not HAS_YOLO:
        return None
    if _yolo_model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[YOLO] Loading model from {MODEL_PATH}...")
            try:
                _yolo_model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"[YOLO ERROR] Failed to load model instance: {e}")
        else:
            print(f"[YOLO ERROR] Model file not found at: {MODEL_PATH}")
    return _yolo_model


def detect_and_crop_label(image, conf_threshold=0.25):
    """
    Locates the label tag using YOLO11n (best.pt), attempts perspective rectification on the crop,
    runs a 4-way rotation sweep to auto-orient the label horizontally based on QR decoding,
    and falls back to standard cropping or CV-only detection if YOLO fails.

    Returns:
        (cropped_label_img, box_points, label_found_bool, qr_data_dict_or_None)
        If label is not detected, returns (None, None, False, None).
    """
    if image is None or image.size == 0:
        return None, None, False, None

    img_h, img_w = image.shape[:2]

    # Try YOLO first
    if HAS_YOLO:
        model = get_yolo_model()
        if model is not None:
            try:
                results = model(image, conf=conf_threshold, verbose=False)
                if results and len(results[0].boxes) > 0:
                    # Select the bounding box with the highest confidence
                    best_box = None
                    best_conf = -1.0
                    for box in results[0].boxes:
                        c = box.conf[0].item()
                        if c > best_conf:
                            best_conf = c
                            best_box = box

                    if best_box is not None:
                        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Bounding box points in full image coordinates
                        box_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                        # Expand the crop slightly to ensure we capture the whole label and its border
                        padding = 20
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(img_w, x2 + padding)
                        crop_y2 = min(img_h, y2 + padding)

                        yolo_crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                        # Attempt classic CV-based perspective rectification inside the YOLO crop
                        rectified_crop = _try_classic_rectification(yolo_crop)
                        base_crop = rectified_crop if rectified_crop is not None else yolo_crop

                        # Run 4-way rotation check using QR code detection to auto-orient horizontal reading
                        from qrtest import inspect_qr_code
                        
                        best_rotated_crop = None
                        detected_qr_data = None
                        
                        # 0, 90 CW, 180, 90 CCW
                        for rot_name, rot_code in [
                            ("0 deg", None),
                            ("90 deg CW", cv2.ROTATE_90_CLOCKWISE),
                            ("180 deg", cv2.ROTATE_180),
                            ("90 deg CCW", cv2.ROTATE_90_COUNTERCLOCKWISE)
                        ]:
                            if rot_code is not None:
                                test_crop = cv2.rotate(base_crop, rot_code)
                            else:
                                test_crop = base_crop.copy()
                                
                            qr_res = inspect_qr_code(test_crop)
                            if qr_res and qr_res.get("qr_found"):
                                print(f"[YOLO/Orientation] QR code successfully decoded at {rot_name} rotation!")
                                best_rotated_crop = test_crop
                                detected_qr_data = qr_res
                                break
                                
                        if best_rotated_crop is not None:
                            return best_rotated_crop, box_pts, True, detected_qr_data
                        else:
                            # Fallback: if QR not detected at any rotation, just check orientation by aspect ratio
                            print("[YOLO/Orientation] WARNING: QR code not decoded at any rotation. Defaulting orientation by aspect ratio.")
                            rh, rw = base_crop.shape[:2]
                            # If taller than wide, rotate 90 CW to make it horizontal
                            if rh > rw:
                                base_crop = cv2.rotate(base_crop, cv2.ROTATE_90_CLOCKWISE)
                            return base_crop, box_pts, True, None
            except Exception as e:
                print(f"[YOLO Exception during inference]: {e}")

    # Fallback B: Pure Classic CV (HSV/LAB + Contours) on the full image
    print("[YOLO] Falling back to classic CV-based label detection...")
    classic_res = _detect_and_crop_classic(image)
    if classic_res and len(classic_res) >= 3:
        return classic_res[0], classic_res[1], classic_res[2], None
    return None, None, False, None


def draw_label_bounding_box(image, box_pts, label_text="Label Detected"):
    """
    Draws a premium glowing neon bounding box and a tag labeled text on the image.
    """
    if box_pts is None or len(box_pts) < 4:
        return

    pts = np.array(box_pts, dtype=np.int32)
    # Draw glowing cyan rectangle (cyan in BGR is (254, 242, 0))
    cv2.polylines(image, [pts], isClosed=True, color=(254, 242, 0), thickness=4, lineType=cv2.LINE_AA)
    cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Draw labeled text tag above the box
    x_min = min([pt[0] for pt in box_pts])
    y_min = min([pt[1] for pt in box_pts])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Text container coordinates
    tx1 = x_min
    ty1 = max(0, y_min - text_h - 10)
    tx2 = x_min + text_w + 12
    ty2 = y_min

    # Overlay container with transparent alpha blending
    overlay = image.copy()
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (254, 242, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Draw the text label
    cv2.putText(image, label_text, (tx1 + 6, ty2 - 6), font, font_scale, (10, 15, 25), thickness, cv2.LINE_AA)


def _try_classic_rectification(crop_image):
    """
    Applies white mask thresholding and perspective transform on the cropped label region
    to correct any perspective distortion or skew.
    """
    if crop_image is None or crop_image.size == 0:
        return None

    h, w = crop_image.shape[:2]
    total_area = h * w

    # Color Space Conversion
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    # White Label Mask
    lower_hsv_white = np.array([0, 0, 140], dtype=np.uint8)
    upper_hsv_white = np.array([180, 80, 255], dtype=np.uint8)
    hsv_white_mask = cv2.inRange(hsv, lower_hsv_white, upper_hsv_white)

    l_channel = lab[:, :, 0]
    _, lab_white_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)

    white_mask = cv2.bitwise_and(hsv_white_mask, lab_white_mask)

    # Morphological Grouping
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_open)

    # Find external contours inside crop
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * 0.1):  # Label should occupy a decent portion of YOLO crop
            continue
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < 20 or rh < 20:
            continue

        aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect_ratio > 5.5:
            continue

        if area > max_area:
            max_area = area
            best_cnt = cnt

    if best_cnt is not None:
        rect = cv2.minAreaRect(best_cnt)
        rectified_crop, _ = _warp_perspective_crop(crop_image, rect)
        
        if rectified_crop is not None and rectified_crop.size > 0:
            # Check orientation
            rh, rw = rectified_crop.shape[:2]
            if rh > (rw * 1.4):
                rectified_crop = cv2.rotate(rectified_crop, cv2.ROTATE_90_CLOCKWISE)
            return rectified_crop

    return None


def _detect_and_crop_classic(image):
    """
    Classic color & contour-based label locator (fallback if YOLO is unavailable or fails).
    """
    if image is None or image.size == 0:
        return None, None, False

    img_h, img_w = image.shape[:2]
    total_area = img_h * img_w

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lower_hsv_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_hsv_white = np.array([180, 80, 255], dtype=np.uint8)
    hsv_white_mask = cv2.inRange(hsv, lower_hsv_white, upper_hsv_white)

    l_channel = lab[:, :, 0]
    _, lab_white_mask = cv2.threshold(l_channel, 160, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(hsv_white_mask, lab_white_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_white = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)
    morphed_white = cv2.morphologyEx(morphed_white, cv2.MORPH_OPEN, kernel_open)

    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_gray, 50, 150)

    contours, _ = cv2.findContours(morphed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * 0.002) or area > (total_area * 0.28):
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < 30 or rh < 30:
            continue

        aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect_ratio > 5.5:
            continue

        cnt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_edge_density = cv2.mean(canny_edges, mask=cnt_mask)[0] / 255.0
        mean_brightness = cv2.mean(gray, mask=cnt_mask)[0]

        area_ratio = area / total_area
        area_score = 1.0 - abs(area_ratio - 0.04)
        score = (mean_edge_density * 5.0) + (mean_brightness / 255.0 * 2.0) + (area_score * 2.0)

        if mean_brightness >= 140 and mean_edge_density >= 0.025:
            candidates.append({
                'contour': cnt,
                'rect': rect,
                'score': score
            })

    if not candidates:
        return None, None, False

    candidates.sort(key=lambda c: c['score'], reverse=True)
    best_candidate = candidates[0]
    best_rect = best_candidate['rect']

    rectified_crop, box_pts = _warp_perspective_crop(image, best_rect)
    if rectified_crop is None or rectified_crop.size == 0:
        return None, None, False

    rh, rw = rectified_crop.shape[:2]
    if rh > (rw * 1.4):
        rectified_crop = cv2.rotate(rectified_crop, cv2.ROTATE_90_CLOCKWISE)

    return rectified_crop, box_pts, True


def _warp_perspective_crop(image, rect):
    """
    Performs perspective transform warping on an oriented bounding box.
    """
    box = cv2.boxPoints(rect)
    pts = box.astype('float32')

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src_pts = np.array([tl, tr, br, bl], dtype='float32')

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 20 or maxHeight < 20:
        return None, None

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, box.astype(int).tolist()
