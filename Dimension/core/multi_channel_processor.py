"""
multi_channel_processor.py — Enhanced ROI & Multi-Channel Mattress Isolation Pipeline.

Designed for the specific scenario:
  - Black outer frame (tape border)
  - Black sheet beneath mattress (may have marble/veined patterns)
  - White/light-colored mattress on top

Uses 3 channels with HIGH luminance cutoffs + aggressive morphological
cleanup to isolate ONLY the mattress, rejecting marble veins and noise.
"""

import cv2
import numpy as np


class MultiChannelProcessor:
    """
    Enhanced 3-channel image processor tuned for white mattress on dark background.
    
    Strategy:
    1. Channel 1 (Lab Luminance): High-cutoff L-channel filter (L > 185) to reject marble veins
    2. Channel 2 (Adaptive Brightness): Large-block adaptive threshold to capture uniform bright regions
    3. Channel 3 (Saturation-Filtered Grayscale): Bilateral + high-threshold, excluding saturated/colored areas
    
    Fusion: Strict agreement + area-based contour filtering to reject scattered noise.
    """

    def __init__(self, clahe_clip_limit=3.0, clahe_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def process_high_contrast_channel(self, img_bgr):
        """
        Channel 1: Lab L-Channel with HIGH Luminance Cutoff.
        
        White mattress fabric: L ≈ 200-255
        Black sheet marble veins: L ≈ 140-175
        Black sheet base: L ≈ 20-80
        
        Cutoff at L > 185 cleanly separates mattress from marble veins.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        
        # High luminance cutoff — only truly bright white fabric passes
        _, mask = cv2.threshold(l_chan, 185, 255, cv2.THRESH_BINARY)
        
        # Remove small isolated bright spots (marble vein fragments)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # Fill small holes inside the mattress (quilting pattern gaps)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        return mask

    def process_grayscale_threshold_channel(self, img_bgr):
        """
        Channel 2: Adaptive Block Threshold for Uniform Bright Regions.
        
        Uses adaptive thresholding with a large block size to detect the mattress
        as a single uniform bright region, ignoring thin marble vein lines.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # High fixed threshold — mattress fabric is significantly brighter than everything else
        _, bright_mask = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        
        # Aggressive morphological cleanup to merge mattress regions and reject streaks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        return bright_mask

    def process_noise_reduction_channel(self, img_bgr):
        """
        Channel 3: Bilateral-Filtered High-Threshold Channel.
        
        Bilateral filter preserves mattress edges while smoothing out marble veins.
        Then a high threshold isolates only the mattress.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Heavy bilateral filtering — smooths marble veins while keeping mattress edges
        bilateral = cv2.bilateralFilter(gray, 11, 85, 85)
        bilateral = cv2.bilateralFilter(bilateral, 9, 75, 75)  # Double-pass for extra smoothing
        
        # High threshold
        _, mask = cv2.threshold(bilateral, 175, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask

    def fuse_channels(self, mask1, mask2, mask3):
        """
        Strict agreement fusion with area-based contour filtering.
        
        Only pixels that appear in at least 2 of 3 channels are kept.
        Then small scattered contours (< 10% of frame area) are rejected.
        """
        # Majority vote: pixel must appear in at least 2 out of 3 channels
        vote = (mask1.astype(np.uint16) + mask2.astype(np.uint16) + mask3.astype(np.uint16))
        fused = np.zeros_like(mask1)
        fused[vote >= 510] = 255  # 510 = 2 * 255, i.e. at least 2 channels agree
        
        # Heavy morphological closing to fill quilting pattern gaps inside mattress
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Light opening to remove small bridge artifacts
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Area-based contour filtering: reject any contour < 10% of total frame area
        frame_area = fused.shape[0] * fused.shape[1]
        min_contour_area = frame_area * 0.10
        
        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(fused)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_contour_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        # Final smooth closing pass
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_final, iterations=1)
        
        return clean_mask

    def process_pipeline(self, img_bgr):
        """
        Executes the full enhanced 3-channel filtering and fusion pipeline.

        Returns:
            fused_mask (np.ndarray): Clean binary mask isolating ONLY the mattress.
            debug_channels (dict): Individual channel masks for visual diagnostics.
        """
        ch1_mask = self.process_high_contrast_channel(img_bgr)
        ch2_mask = self.process_grayscale_threshold_channel(img_bgr)
        ch3_mask = self.process_noise_reduction_channel(img_bgr)

        fused_mask = self.fuse_channels(ch1_mask, ch2_mask, ch3_mask)

        debug_channels = {
            "ch1_high_contrast": ch1_mask,
            "ch2_grayscale_otsu": ch2_mask,
            "ch3_noise_reduction": ch3_mask,
            "fused_mask": fused_mask
        }

        return fused_mask, debug_channels
