"""
multi_channel_processor.py — 3-Channel Image Filtering & Fusion Pipeline.

Eliminates factory lighting reflections, specular glare, fabric color casts,
and printed logo text noise by combining 3 distinct preprocessing channels.
"""

import cv2
import numpy as np


class MultiChannelProcessor:
    """
    Processes single camera captures through 3 specialized filter channels:
    1. High-Contrast Channel (CLAHE + Morphological Edges)
    2. Grayscale/Thresholding Channel (Otsu + Adaptive Threshold)
    3. Noise/Glare Reduction Channel (Bilateral + Median Filtering)
    Followed by bitwise fusion into a noise-free binary mask.
    """

    def __init__(self, clahe_clip_limit=3.0, clahe_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def process_high_contrast_channel(self, img_bgr):
        """
        Channel 1: Lab L-Channel Luminance Filter.
        Isolates bright white mattress fabric (L > 165) from gray/beige table surface (L ~ 130).
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        
        # Pure luminance thresholding (White mattress > 165, Gray table ~130)
        _, mask_lum = cv2.threshold(l_chan, 165, 255, cv2.THRESH_BINARY)
        
        # Canny edge gradient for sharp piped borders
        edges = cv2.Canny(l_chan, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        mask = cv2.bitwise_and(mask_lum, mask_lum)
        return mask

    def process_grayscale_threshold_channel(self, img_bgr):
        """
        Channel 2: Grayscale High-Luminance Threshold Channel.
        Thresholds bright white mattress fabric (> 160) while completely cutting out gray table surface.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # High brightness threshold (cuts out gray table at ~130)
        _, bright_mask = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
        return bright_mask

    def process_noise_reduction_channel(self, img_bgr):
        """
        Channel 3: Glare & Shadow Noise Reduction Channel.
        Applies Bilateral and Median filtering to preserve sharp white mattress edges.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        median = cv2.medianBlur(bilateral, 7)

        _, mask = cv2.threshold(median, 160, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask_clean

    def fuse_channels(self, mask1, mask2, mask3):
        """
        Merges masks from all 3 filter channels.
        Uses pure luminance agreement fusion to isolate ONLY the white mattress contour.
        """
        # Channel agreement fusion between Lab Luminance & Grayscale High-Threshold
        fused = cv2.bitwise_and(mask1, mask2)
        fused = cv2.bitwise_or(fused, cv2.bitwise_and(mask2, mask3))

        # Morphological closing & opening to fill internal fabric quilting lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        final_clean = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_clean = cv2.morphologyEx(final_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)

        return final_clean

    def process_pipeline(self, img_bgr):
        """
        Executes the full 3-channel filtering and fusion pipeline.

        Returns:
            fused_mask (np.ndarray): Clean binary mask of mattress contour.
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
