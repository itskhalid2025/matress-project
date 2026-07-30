"""
mattress/crop.py — front-end localisation (G8).

For the POC on collected reference frames, the documented FIXED-CROP switch is
used (controlled-line mode). This is a scope choice, NOT the banner defence —
banner exclusion is appearance-based in imageio.build_exclusion_masks. The
dynamic localise/deskew path for the cluttered dev site is Task 3 and is left
as a clean seam.
"""
import config as cfg


def localise_cover(frame_bgr, production_mode=True):
    h, w = frame_bgr.shape[:2]
    if w != cfg.CAPTURE_W or h != cfg.CAPTURE_H:
        raise ValueError(
            f"G2 scale violation: frame is {w}x{h}, "
            f"enrolled scale is {cfg.CAPTURE_W}x{cfg.CAPTURE_H}. "
            f"Never mix resolutions — features are scale-sensitive.")
    y0, y1, x0, x1 = cfg.FIXED_CROP
    return frame_bgr[y0:y1, x0:x1]


# --- Task 3 seam (dynamic crop + deskew for the cluttered dev site) ---
def localise_cover_dynamic(frame_bgr):
    raise NotImplementedError(
        "Task 3: min-area-rect localisation + deskew. Do not implement with a "
        "naive biggest-bright-blob detector (locks onto adjacent mattresses); "
        "crop with the ROTATED rect after warping, not the pre-rotation bbox.")
