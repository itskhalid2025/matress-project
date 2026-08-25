"""
dimension_calculator.py — Pixel Gap Arithmetic & Metric Dimension Calculator.

Calculates mattress physical width and length by computing pixel gaps relative
to the reference frame border and multiplying by dynamic scale factors (cm/pixel).
"""

import cv2
import numpy as np


class DimensionCalculator:
    """
    Calculates metric dimensions using pixel gap subtraction:
    Dimension = Reference Metric Size - (Gap1 + Gap2) * Scale Ratio
    """

    def __init__(self, ref_width_cm=100.0, ref_height_cm=120.0):
        self.ref_width_cm = float(ref_width_cm)
        self.ref_height_cm = float(ref_height_cm)

    def calculate_dimensions(self, fused_mask, scale_x, scale_y, warped_shape):
        """
        Computes metric dimensions from the fused binary mask.

        Parameters:
            fused_mask (np.ndarray): Binary mask of the mattress.
            scale_x (float): Horizontal scale factor (cm per pixel).
            scale_y (float): Vertical scale factor (cm per pixel).
            warped_shape (tuple): Shape of top-down warped workspace (H, W).

        Returns:
            result_dict (dict): Dictionary containing calculated dimensions, pixel gaps, and contour data.
        """
        h_warped, w_warped = warped_shape[:2]

        contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "success": False,
                "error": "No mattress contour detected inside reference frame."
            }

        # Filter out minor noise contours, select largest inner object (mattress)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        mattress_contour = contours[0]

        # Orthogonal Bounding Box
        x, y, w_box, h_box = cv2.boundingRect(mattress_contour)

        # Pixel gaps relative to reference border edges
        gap_left_px = x
        gap_right_px = w_warped - (x + w_box)
        gap_top_px = y
        gap_bottom_px = h_warped - (y + h_box)

        # Calculated metric size via calibrated pixel-to-metric translation
        width_cm = round(w_box * scale_x, 2)
        height_cm = round(h_box * scale_y, 2)

        # Rotated Minimum Area Bounding Box
        rect = cv2.minAreaRect(mattress_contour)
        (cx, cy), (rot_w_px, rot_h_px), angle = rect
        box_pts = cv2.boxPoints(rect)

        rot_w_cm = round(rot_w_px * scale_x, 2)
        rot_h_cm = round(rot_h_px * scale_y, 2)

        # Ensure width is smaller than length convention
        final_width_cm = min(width_cm, height_cm)
        final_length_cm = max(width_cm, height_cm)

        # Convert to inches
        width_in = round(final_width_cm / 2.54, 1)
        length_in = round(final_length_cm / 2.54, 1)

        area_sq_m = round((final_width_cm * final_length_cm) / 10000.0, 3)

        return {
            "success": True,
            "width_cm": float(final_width_cm),
            "length_cm": float(final_length_cm),
            "width_in": float(width_in),
            "length_in": float(length_in),
            "area_sq_m": float(area_sq_m),
            "pixel_gaps": {
                "left_px": int(gap_left_px),
                "right_px": int(gap_right_px),
                "top_px": int(gap_top_px),
                "bottom_px": int(gap_bottom_px)
            },
            "metric_gaps_cm": {
                "left_cm": round(float(gap_left_px * scale_x), 2),
                "right_cm": round(float(gap_right_px * scale_x), 2),
                "top_cm": round(float(gap_top_px * scale_y), 2),
                "bottom_cm": round(float(gap_bottom_px * scale_y), 2)
            },
            "rotated_box": {
                "width_cm": float(min(rot_w_cm, rot_h_cm)),
                "length_cm": float(max(rot_w_cm, rot_h_cm)),
                "angle": round(float(angle), 1),
                "center_px": [round(float(cx), 1), round(float(cy), 1)],
                "box_points": box_pts.tolist()
            },
            "contour": mattress_contour,
            "bounding_rect": [int(x), int(y), int(w_box), int(h_box)]
        }
