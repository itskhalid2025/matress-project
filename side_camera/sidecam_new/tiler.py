"""
tiler.py — Sliding Window Image Tiling Module for sidecam_new.

Divides full-resolution camera frames into overlapping tiles (640x640 pixels, 25-30% overlap).
Optimized for complete coverage without splitting QR codes or text lines across tile boundaries.
"""

import cv2
import numpy as np


def generate_tiles(image, tile_size=640, overlap_pct=0.25):
    """
    Splits an image into overlapping square tiles of size `tile_size` x `tile_size`.
    
    Args:
        image (np.ndarray): Full resolution input BGR image.
        tile_size (int): Tile width and height in pixels (default 640).
        overlap_pct (float): Overlap percentage between adjacent tiles (default 0.25 = 25%).
        
    Returns:
        list of tuples: [(tile_index, tile_bgr, (x, y, w, h)), ...]
    """
    if image is None or image.size == 0:
        return []

    img_h, img_w = image.shape[:2]

    # Calculate stride step size
    stride = int(tile_size * (1.0 - overlap_pct))
    stride = max(1, stride)

    # Compute top-left (x, y) coordinates for all sliding windows
    x_coords = []
    curr_x = 0
    while curr_x + tile_size <= img_w:
        x_coords.append(curr_x)
        curr_x += stride
    # Ensure rightmost edge is covered
    if not x_coords or (img_w - tile_size) > x_coords[-1]:
        x_coords.append(max(0, img_w - tile_size))

    y_coords = []
    curr_y = 0
    while curr_y + tile_size <= img_h:
        y_coords.append(curr_y)
        curr_y += stride
    # Ensure bottom edge is covered
    if not y_coords or (img_h - tile_size) > y_coords[-1]:
        y_coords.append(max(0, img_h - tile_size))

    tiles = []
    tile_idx = 1

    for y in y_coords:
        for x in x_coords:
            # Crop tile from full resolution image
            tile_crop = image[y:y + tile_size, x:x + tile_size].copy()
            actual_h, actual_w = tile_crop.shape[:2]
            
            tiles.append((tile_idx, tile_crop, (x, y, actual_w, actual_h)))
            tile_idx += 1

    return tiles
