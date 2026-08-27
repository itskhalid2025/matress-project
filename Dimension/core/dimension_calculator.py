"""
dimension_calculator.py — Scan-Line Longest-Run Mattress Dimension Calculator.

Instead of contour tracing (which merges marble veins with the mattress),
this uses a SCAN-LINE approach:
  - For each row, find the longest continuous horizontal run of white pixels
  - For each column, find the longest continuous vertical run of white pixels
  - The mattress = the largest continuous block of long runs
  - Short scattered runs (marble veins, noise) are automatically filtered out
"""

import cv2
import numpy as np


class DimensionCalculator:
    """
    Scan-Line Longest-Run dimension calculator.
    
    Immune to marble vein noise, scattered reflections, and curvy contour deviations
    because it measures the longest continuous stretch, not contour shape.
    """

    def __init__(self, ref_width_cm=100.0, ref_height_cm=120.0, mattress_distance_cm=100.0, table_distance_cm=130.0):
        self.ref_width_cm = float(ref_width_cm)
        self.ref_height_cm = float(ref_height_cm)
        self.mattress_distance_cm = float(mattress_distance_cm or 100.0)
        self.table_distance_cm = float(table_distance_cm or 130.0)

    def _find_longest_run_per_row(self, mask):
        """
        For each row, find the longest continuous horizontal run of white (255) pixels.
        
        Returns: list of (row_index, run_start_col, run_end_col, run_length)
        """
        h, w = mask.shape
        results = []
        
        for r in range(h):
            row = mask[r, :]
            best_start, best_end, best_len = 0, 0, 0
            start = None
            
            for c in range(w):
                if row[c] > 0:
                    if start is None:
                        start = c
                else:
                    if start is not None:
                        run_len = c - start
                        if run_len > best_len:
                            best_start, best_end, best_len = start, c, run_len
                        start = None
            
            # Handle run that extends to right edge
            if start is not None:
                run_len = w - start
                if run_len > best_len:
                    best_start, best_end, best_len = start, w, run_len
            
            results.append((r, best_start, best_end, best_len))
        
        return results

    def _find_longest_vertical_stretch(self, row_runs, frame_width, min_run_ratio=0.15):
        """
        Find the longest continuous vertical stretch of rows that have 
        significant horizontal runs (> min_run_ratio of frame width).
        
        This identifies the mattress height region.
        Short runs from marble veins are filtered out.
        """
        min_run_px = frame_width * min_run_ratio
        
        # Mark rows as "significant" (has a long enough horizontal run)
        significant = [(r, s, e, l) for r, s, e, l in row_runs if l >= min_run_px]
        
        if not significant:
            return None
        
        # Group into consecutive vertical stretches (allow up to 5px gaps for noise)
        groups = []
        current_group = [significant[0]]
        
        for i in range(1, len(significant)):
            row_gap = significant[i][0] - significant[i-1][0]
            if row_gap <= 5:  # Allow small vertical gaps (quilting seams, shadows)
                current_group.append(significant[i])
            else:
                groups.append(current_group)
                current_group = [significant[i]]
        groups.append(current_group)
        
        # Select the longest vertical group = mattress
        best_group = max(groups, key=len)
        return best_group

    def _compute_robust_bounds(self, group):
        """
        From the longest-run group, compute robust mattress bounds 
        using median of run starts/ends (rejects outlier deviations).
        """
        starts = [s for _, s, e, l in group]
        ends = [e for _, s, e, l in group]
        
        # Use 10th and 90th percentile instead of raw min/max to reject curvy outliers
        left = int(np.percentile(starts, 10))
        right = int(np.percentile(ends, 90))
        top = group[0][0]
        bottom = group[-1][0]
        
        return left, top, right - left, bottom - top

    def calculate_dimensions(self, fused_mask, scale_x, scale_y, warped_shape, distance_correction_ratio=1.0):
        """
        Computes mattress dimensions using the Scan-Line Longest-Run algorithm.
        
        Instead of tracing contours (which merge marble veins with the mattress),
        this scans each row for the longest continuous stretch of white pixels.
        The mattress appears as a large block of long runs. Marble veins, noise,
        and curvy deviations are automatically filtered out.

        Parameters:
            fused_mask (np.ndarray): Binary mask (can be noisy — algorithm handles it).
            scale_x (float): Horizontal scale factor (cm per pixel).
            scale_y (float): Vertical scale factor (cm per pixel).
            warped_shape (tuple): Shape of warped workspace (H, W).
            distance_correction_ratio (float): Camera distance correction factor.

        Returns:
            result_dict (dict): Calculated dimensions with scan-line metadata.
        """
        h_warped, w_warped = warped_shape[:2]

        # Step 1: Scan each row for the longest horizontal run
        row_runs = self._find_longest_run_per_row(fused_mask)

        # Step 2: Find the longest continuous vertical stretch of significant rows
        best_group = self._find_longest_vertical_stretch(row_runs, w_warped, min_run_ratio=0.15)

        if best_group is None or len(best_group) < 20:
            return {
                "success": False,
                "error": "No continuous mattress region detected (scan-line algorithm found no significant runs)."
            }

        # Step 3: Compute robust rectangular bounds using percentile filtering
        x, y, w_box, h_box = self._compute_robust_bounds(best_group)
        
        # Safety clamp
        x = max(0, x)
        y = max(0, y)
        w_box = min(w_box, w_warped - x)
        h_box = min(h_box, h_warped - y)

        if w_box < 10 or h_box < 10:
            return {
                "success": False,
                "error": "Detected region too small to be a mattress."
            }

        # Step 4: Apply camera height perspective correction
        eff_scale_x = scale_x * float(distance_correction_ratio or 1.0)
        eff_scale_y = scale_y * float(distance_correction_ratio or 1.0)

        # Calculated metric size
        width_cm = round(w_box * eff_scale_x, 2)
        height_cm = round(h_box * eff_scale_y, 2)

        # Pixels per cm ratio
        pixels_per_cm_x = round(1.0 / eff_scale_x, 2) if eff_scale_x > 0 else 0.0
        pixels_per_cm_y = round(1.0 / eff_scale_y, 2) if eff_scale_y > 0 else 0.0

        # Ensure width < length convention
        final_width_cm = min(width_cm, height_cm)
        final_length_cm = max(width_cm, height_cm)

        # Convert to inches
        width_in = round(final_width_cm / 2.54, 1)
        length_in = round(final_length_cm / 2.54, 1)

        area_sq_m = round((final_width_cm * final_length_cm) / 10000.0, 3)

        # Build a rectangular contour for visualization (clean box, no curvy noise)
        box_contour = np.array([
            [x, y],
            [x + w_box, y],
            [x + w_box, y + h_box],
            [x, y + h_box]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Rotated bounding box from scan-line bounds
        cx = float(x + w_box / 2.0)
        cy = float(y + h_box / 2.0)
        box_pts = np.array([
            [x, y],
            [x + w_box, y],
            [x + w_box, y + h_box],
            [x, y + h_box]
        ], dtype=np.float32)

        return {
            "success": True,
            "width_cm": float(final_width_cm),
            "length_cm": float(final_length_cm),
            "width_in": float(width_in),
            "length_in": float(length_in),
            "area_sq_m": float(area_sq_m),
            "pixels_per_cm": round(float((pixels_per_cm_x + pixels_per_cm_y) / 2.0), 2),
            "pixels_per_cm_x": float(pixels_per_cm_x),
            "pixels_per_cm_y": float(pixels_per_cm_y),
            "mattress_distance_cm": float(self.mattress_distance_cm),
            "bounding_box_pixels": {
                "w_px": int(w_box),
                "h_px": int(h_box)
            },
            "scan_line_stats": {
                "total_significant_rows": len(best_group),
                "median_run_length_px": int(np.median([l for _, _, _, l in best_group])),
                "min_run_length_px": int(min([l for _, _, _, l in best_group])),
                "max_run_length_px": int(max([l for _, _, _, l in best_group]))
            },
            "pixel_gaps": {
                "left_px": int(x),
                "right_px": int(w_warped - (x + w_box)),
                "top_px": int(y),
                "bottom_px": int(h_warped - (y + h_box))
            },
            "metric_gaps_cm": {
                "left_cm": round(float(x * eff_scale_x), 2),
                "right_cm": round(float((w_warped - (x + w_box)) * eff_scale_x), 2),
                "top_cm": round(float(y * eff_scale_y), 2),
                "bottom_cm": round(float((h_warped - (y + h_box)) * eff_scale_y), 2)
            },
            "rotated_box": {
                "width_cm": float(final_width_cm),
                "length_cm": float(final_length_cm),
                "angle": 0.0,
                "center_px": [round(cx, 1), round(cy, 1)],
                "box_points": box_pts.tolist()
            },
            "contour": box_contour,
            "bounding_rect": [int(x), int(y), int(w_box), int(h_box)]
        }
