"""
dimension_engine.py — Unified API & Visual Diagnostic Engine for Mattress Measurement.

Combines reference frame calibration, multi-channel image processing, and
matrix dimension arithmetic into a single high-performance pipeline.
"""

import cv2
import numpy as np
import json
import os
try:
    from .reference_calibration import ReferenceCalibrator, load_config
    from .multi_channel_processor import MultiChannelProcessor
    from .dimension_calculator import DimensionCalculator
except ImportError:
    from reference_calibration import ReferenceCalibrator, load_config
    from multi_channel_processor import MultiChannelProcessor
    from dimension_calculator import DimensionCalculator


class MattressDimensionEngine:
    """
    Main entry point for mattress dimension checking.
    Accepts camera images and optional dynamic reference frame inputs.
    """

    def __init__(self, ref_width_cm=100.0, ref_height_cm=120.0, border_color_mode="black", edge_lengths=None, config_path=None):
        self.config_path = config_path
        self.calibrator = ReferenceCalibrator(ref_width_cm, ref_height_cm, border_color_mode, edge_lengths, config_path)
        self.processor = MultiChannelProcessor()
        self.calculator = DimensionCalculator(self.calibrator.ref_width_cm, self.calibrator.ref_height_cm)

    def process_frame(self, img_bgr, ref_width_cm=None, ref_height_cm=None, border_color_mode=None):
        """
        Processes a raw BGR frame and computes physical mattress dimensions.

        Parameters:
            img_bgr (np.ndarray): Input camera BGR frame.
            ref_width_cm (float, optional): Dynamic reference width in cm.
            ref_height_cm (float, optional): Dynamic reference height in cm.
            border_color_mode (str, optional): Border color preset ("red", "yellow", "black", "white", "auto").

        Returns:
            res (dict): Calculation results dictionary.
            annotated_img (np.ndarray): Primary annotated output frame.
            debug_grid (np.ndarray): 4-quadrant multi-channel debug visualization grid.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {"success": False, "error": "Invalid or empty image frame."}, img_bgr, None

        # Dynamically update reference frame metrics if supplied
        if ref_width_cm and ref_height_cm:
            self.calibrator.update_reference_dimensions(ref_width_cm, ref_height_cm)
            self.calculator.ref_width_cm = float(ref_width_cm)
            self.calculator.ref_height_cm = float(ref_height_cm)

        if border_color_mode:
            self.calibrator.border_color_mode = str(border_color_mode).lower()

        # Step 1: Detect border & Perspective Warp
        try:
            warped, corners, scale_x, scale_y, M = self.calibrator.calibrate_and_warp(img_bgr)
        except Exception as e:
            annotated = img_bgr.copy()
            cv2.putText(annotated, f"CALIBRATION ERROR: {str(e)}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return {"success": False, "error": str(e)}, annotated, None

        # Step 2: Multi-Channel Filter & Fusion on warped frame
        fused_mask, debug_channels = self.processor.process_pipeline(warped)

        # Mask out outer 3% perimeter of fused_mask to clear any residual reference tape
        tape_margin_h = int(warped.shape[0] * 0.03)
        tape_margin_w = int(warped.shape[1] * 0.03)
        fused_mask[:tape_margin_h, :] = 0
        fused_mask[-tape_margin_h:, :] = 0
        fused_mask[:, :tape_margin_w] = 0
        fused_mask[:, -tape_margin_w:] = 0

        # Step 3: Dimension Calculation
        res = self.calculator.calculate_dimensions(fused_mask, scale_x, scale_y, warped.shape)

        # Step 4: Visual Overlay Generation
        annotated_img, debug_grid = self.render_visualizations(img_bgr, warped, corners, fused_mask,
                                                               debug_channels, res, scale_x, scale_y)

        return res, annotated_img, debug_grid

    def render_visualizations(self, original_img, warped_img, corners, fused_mask, debug_channels, res, scale_x, scale_y):
        """Renders rich color-coded annotations and 4-quadrant diagnostic debug grid."""
        annotated = original_img.copy()

        # Draw detected Reference Border (Blue)
        if corners is not None:
            pts = np.int32(corners)
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 100, 0), thickness=3)
            for i, p in enumerate(pts):
                cv2.circle(annotated, tuple(p), 6, (0, 255, 255), -1)

        # Draw scale info banner
        banner_text = (f"REF FRAME: {self.calibrator.ref_width_cm}x{self.calibrator.ref_height_cm} cm | "
                       f"MODE: {self.calibrator.border_color_mode.upper()} | "
                       f"SCALE: {round(scale_x, 4)} cm/px")
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(annotated, banner_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Render Warped Overlay with Mattress Contour
        annotated_warped = warped_img.copy()

        if res.get("success"):
            box_pts = np.int32(res["rotated_box"]["box_points"])
            cv2.drawContours(annotated_warped, [box_pts], 0, (0, 255, 0), 3)

            w_cm = res["width_cm"]
            l_cm = res["length_cm"]
            w_in = res["width_in"]
            l_in = res["length_in"]

            cx, cy = int(warped_img.shape[1] / 2), int(warped_img.shape[0] / 2)

            label_w = f"W: {w_cm} cm ({w_in} in)"
            label_l = f"L: {l_cm} cm ({l_in} in)"

            cv2.rectangle(annotated_warped, (cx - 150, cy - 35), (cx + 150, cy + 35), (0, 0, 0), -1)
            cv2.putText(annotated_warped, label_w, (cx - 130, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(annotated_warped, label_l, (cx - 130, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Draw Mattress Contour on Original Frame
            contour = res["contour"]
            if isinstance(contour, list):
                contour = np.array(contour, dtype=np.int32)
            elif not isinstance(contour, np.ndarray):
                contour = np.array(contour, dtype=np.int32)
            cv2.drawContours(annotated_warped, [contour], -1, (0, 0, 255), 2)

        # Create 4-Quadrant Multi-Channel Debug Grid
        h_w, w_w = warped_img.shape[:2]
        thumb_w, thumb_h = 480, int(480 * (h_w / float(w_w)))

        ch1_color = cv2.cvtColor(debug_channels["ch1_high_contrast"], cv2.COLOR_GRAY2BGR)
        ch2_color = cv2.cvtColor(debug_channels["ch2_grayscale_otsu"], cv2.COLOR_GRAY2BGR)
        ch3_color = cv2.cvtColor(debug_channels["ch3_noise_reduction"], cv2.COLOR_GRAY2BGR)
        fused_color = cv2.cvtColor(fused_mask, cv2.COLOR_GRAY2BGR)

        cv2.putText(ch1_color, "Ch1: High-Contrast CLAHE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(ch2_color, "Ch2: Grayscale & Otsu", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(ch3_color, "Ch3: Glare Noise Filter", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(fused_color, "Fused Binary Contour Mask", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        top_row = np.hstack([cv2.resize(ch1_color, (thumb_w, thumb_h)), cv2.resize(ch2_color, (thumb_w, thumb_h))])
        bot_row = np.hstack([cv2.resize(ch3_color, (thumb_w, thumb_h)), cv2.resize(fused_color, (thumb_w, thumb_h))])
        debug_grid = np.vstack([top_row, bot_row])

        return annotated_warped, debug_grid

    def detect_and_draw_border_overlay(self, img_bgr, active_edge=None):
        """
        Step 1: Detects black reference border and draws 4 interactive edge overlays.
        Returns:
            annotated_img (np.ndarray): Image with 4 colored edge overlays and mid-point labels.
            border_info (dict): Edge details (corners, edge lengths, midpoints).
        """
        annotated = img_bgr.copy()
        corners, border_contour = self.calibrator.find_reference_corners(img_bgr)
        
        if corners is None:
            cv2.putText(annotated, "SEARCHING FOR BLACK BORDER...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated, {"detected": False, "corners": None, "edges": None}

        edges = self.calibrator.get_edge_details(corners)
        
        # Header banner
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (20, 20, 20), -1)
        cv2.putText(annotated, f"STEP 1: BLACK BORDER DETECTED | Mode: {self.calibrator.border_color_mode.upper()}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        edge_colors = {
            "top": (0, 255, 255),      # Yellow
            "right": (255, 150, 0),    # Orange
            "bottom": (255, 0, 255),   # Magenta
            "left": (0, 200, 255)      # Cyan
        }

        for edge_key, info in edges.items():
            is_active = (active_edge and str(active_edge).lower() == edge_key)
            color = (0, 255, 0) if is_active else edge_colors.get(edge_key, (0, 255, 255))
            thickness = 6 if is_active else 3

            # Draw edge line segment
            cv2.line(annotated, info["pt1"], info["pt2"], color, thickness)

            # Draw midpoint label box
            mx, my = info["mid"]
            label = info["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (mx - 10, my - th - 10), (mx + tw + 10, my + 10), (0, 0, 0), -1)
            cv2.rectangle(annotated, (mx - 10, my - th - 10), (mx + tw + 10, my + 10), color, 2)
            cv2.putText(annotated, label, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Corner Vertex Points (P1, P2, P3, P4)
        for i, pt in enumerate(corners):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(annotated, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(annotated, (x, y), 10, (0, 0, 0), 2)
            cv2.putText(annotated, f"P{i+1}", (x + 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated, {
            "detected": True,
            "corners": corners.tolist(),
            "edges": {k: v["length_cm"] for k, v in edges.items()},
            "edge_details": edges
        }
