"""
reference_calibration.py — Physical Reference Frame Detection & Perspective Calibration.

Detects the physical reference tape/border on the inspection table, extracts the 4 corners,
computes top-down perspective warp, and calculates dynamic pixel-to-metric scale ratios.
"""

import cv2
import numpy as np
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(config_path=None):
    """Loads configuration parameters from JSON file."""
    path = config_path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "ref_width_cm": 100.0,
        "ref_height_cm": 120.0,
        "border_color_mode": "red",
        "min_contour_area_ratio": 0.05,
        "color_hsv_ranges": {
            "red": [[[0, 100, 100], [10, 255, 255]], [[160, 100, 100], [180, 255, 255]]],
            "yellow": [[[15, 100, 100], [35, 255, 255]]],
            "green": [[[35, 50, 50], [85, 255, 255]]],
            "blue": [[[90, 80, 80], [130, 255, 255]]],
            "black": [[[0, 0, 0], [180, 255, 70]]],
            "white": [[[0, 0, 180], [180, 40, 255]]]
        }
    }


def order_points(pts):
    """
    Orders 4 points (x, y) into a consistent order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates: top-right has smallest difference (y - x or x - y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


class ReferenceCalibrator:
    """
    Handles physical reference border detection, 4-corner perspective warping,
    and automatic scale factor calculation (cm per pixel).
    """

    def __init__(self, ref_width_cm=100.0, ref_height_cm=120.0, border_color_mode="black", edge_lengths=None, config_path=None):
        self.config = load_config(config_path)
        self.border_color_mode = (border_color_mode or self.config.get("border_color_mode", "black")).lower()
        
        # 4-Edge asymmetric metric inputs (Top, Right, Bottom, Left in cm)
        self.edge_lengths = edge_lengths or {
            "top": float(ref_width_cm or 100.0),
            "right": float(ref_height_cm or 120.0),
            "bottom": float(ref_width_cm or 100.0),
            "left": float(ref_height_cm or 120.0)
        }
        self.ref_width_cm = (self.edge_lengths["top"] + self.edge_lengths["bottom"]) / 2.0
        self.ref_height_cm = (self.edge_lengths["left"] + self.edge_lengths["right"]) / 2.0

    def update_edge_lengths(self, top_cm=None, right_cm=None, bottom_cm=None, left_cm=None):
        """Updates individual edge lengths (cm) one-by-one or all together."""
        if top_cm is not None:
            self.edge_lengths["top"] = float(top_cm)
        if right_cm is not None:
            self.edge_lengths["right"] = float(right_cm)
        if bottom_cm is not None:
            self.edge_lengths["bottom"] = float(bottom_cm)
        if left_cm is not None:
            self.edge_lengths["left"] = float(left_cm)
        
        self.ref_width_cm = (self.edge_lengths["top"] + self.edge_lengths["bottom"]) / 2.0
        self.ref_height_cm = (self.edge_lengths["left"] + self.edge_lengths["right"]) / 2.0

    def update_reference_dimensions(self, ref_width_cm, ref_height_cm):
        """Updates reference dimensions for backward compatibility."""
        self.update_edge_lengths(
            top_cm=ref_width_cm,
            bottom_cm=ref_width_cm,
            left_cm=ref_height_cm,
            right_cm=ref_height_cm
        )

    def detect_border_mask(self, img_hsv):
        """Generates a binary mask of the reference border based on color preset or HSV limits."""
        hsv_ranges = self.config.get("color_hsv_ranges", {})
        color = self.border_color_mode

        if color in hsv_ranges:
            mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
            for lower, upper in hsv_ranges[color]:
                lower_b = np.array(lower, dtype=np.uint8)
                upper_b = np.array(upper, dtype=np.uint8)
                sub_mask = cv2.inRange(img_hsv, lower_b, upper_b)
                mask = cv2.bitwise_or(mask, sub_mask)
            return mask

        # Fallback for auto/geometric mode: use adaptive morphological edges
        gray = img_hsv[:, :, 2]  # Value channel
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        return mask

    def find_reference_corners(self, img):
        """
        Locates the 4 corners of the outer reference frame border.
        Returns:
            corners (np.ndarray or None): 4 ordered corner points (TL, TR, BR, BL).
            border_contour (np.ndarray or None): The raw detected contour.
        """
        h_orig, w_orig = img.shape[:2]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = self.detect_border_mask(img_hsv)

        # Morphological close to bridge gaps in tape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback: edge detection across full gray frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Sort by contour area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_area = (h_orig * w_orig) * 0.15

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                return order_points(pts), c

            # Fallback: Minimum area bounding rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            return order_points(box), c

        # Ultimate fallback: return image boundaries with small margin
        margin = 15
        pts = np.array([
            [margin, margin],
            [w_orig - margin, margin],
            [w_orig - margin, h_orig - margin],
            [margin, h_orig - margin]
        ], dtype="float32")
        return pts, None

    def calibrate_and_warp(self, img, target_width_px=1000):
        """
        Detects reference frame, warps image to top-down view, and calculates scale factors.

        Returns:
            warped_img (np.ndarray): Perspective-corrected top-down frame.
            corners (np.ndarray): 4 corners on original image.
            scale_x (float): cm per pixel (Horizontal).
            scale_y (float): cm per pixel (Vertical).
            warp_matrix (np.ndarray): Perspective transformation matrix.
        """
        corners, border_contour = self.find_reference_corners(img)
        if corners is None:
            raise ValueError("Reference border could not be detected in image.")

        # Compute target height pixels proportional to metric aspect ratio
        aspect_ratio = self.ref_height_cm / self.ref_width_cm
        target_height_px = int(target_width_px * aspect_ratio)

        dst_pts = np.array([
            [0, 0],
            [target_width_px - 1, 0],
            [target_width_px - 1, target_height_px - 1],
            [0, target_height_px - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(corners.astype("float32"), dst_pts)
        warped = cv2.warpPerspective(img, M, (target_width_px, target_height_px))

        # Calculate dynamic scale factors (cm per pixel)
        scale_x = self.ref_width_cm / float(target_width_px)
        scale_y = self.ref_height_cm / float(target_height_px)

        return warped, corners, scale_x, scale_y, M

    def get_edge_details(self, corners):
        """
        Extracts 4 individual edge line segments, midpoints, and labels for rendering overlay.
        corners: ordered [TL, TR, BR, BL]
        """
        if corners is None or len(corners) != 4:
            return None

        tl, tr, br, bl = corners

        edges = {
            "top": {
                "pt1": tuple(np.int32(tl)),
                "pt2": tuple(np.int32(tr)),
                "mid": (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2)),
                "length_cm": self.edge_lengths["top"],
                "label": f"1. TOP: {self.edge_lengths['top']} cm"
            },
            "right": {
                "pt1": tuple(np.int32(tr)),
                "pt2": tuple(np.int32(br)),
                "mid": (int((tr[0] + br[0]) / 2), int((tr[1] + br[1]) / 2)),
                "length_cm": self.edge_lengths["right"],
                "label": f"2. RIGHT: {self.edge_lengths['right']} cm"
            },
            "bottom": {
                "pt1": tuple(np.int32(br)),
                "pt2": tuple(np.int32(bl)),
                "mid": (int((br[0] + bl[0]) / 2), int((br[1] + bl[1]) / 2)),
                "length_cm": self.edge_lengths["bottom"],
                "label": f"3. BOTTOM: {self.edge_lengths['bottom']} cm"
            },
            "left": {
                "pt1": tuple(np.int32(bl)),
                "pt2": tuple(np.int32(tl)),
                "mid": (int((bl[0] + tl[0]) / 2), int((bl[1] + tl[1]) / 2)),
                "length_cm": self.edge_lengths["left"],
                "label": f"4. LEFT: {self.edge_lengths['left']} cm"
            }
        }
        return edges
