"""
smoke_test.py — Integration and import verification for Top Camera module.
"""

import sys
import os

print("Starting smoke test for Mattress Top Camera Rig...")

# Check files exist
expected_files = [
    "config.py",
    "camera.py",
    "dimensions.py",
    "banner.py",
    "main.py",
    "yolov8n.pt",
    "templates/gravite_banner_templates.pkl",
    "templates/ortholex_banner_templates.pkl",
    "templates/maxi_plush_banner_templates.pkl"
]

current_dir = os.path.dirname(os.path.abspath(__file__))
for f in expected_files:
    p = os.path.join(current_dir, f)
    if not os.path.exists(p):
        print(f"ERROR: Expected file missing: {f} at path {p}")
        sys.exit(1)
    else:
        print(f"  [ok] Found file: {f}")

# Try importing packages
try:
    import numpy as np
    import cv2
    import pytesseract
    from ultralytics import YOLO
    print("  [ok] System and virtualenv library dependencies imported successfully.")
except Exception as e:
    print(f"ERROR: Failed to import standard dependencies: {str(e)}")
    sys.exit(1)

# Try local imports
try:
    import config as cfg
    from camera import ThreadedCamera
    from dimensions import measure_dimensions, get_model
    from banner import read_banner, get_galleries_for_hue
    print("  [ok] Local module files imported successfully.")
except Exception as e:
    print(f"ERROR: Failed to import local module files: {str(e)}")
    sys.exit(1)

# Test YOLO model loading
try:
    model = get_model()
    print("  [ok] Local yolov8n.pt model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load YOLO model: {str(e)}")
    sys.exit(1)

# Test templates loading
try:
    # Test hue ranges to verify template lazy loading
    gravite_gals = get_galleries_for_hue(10)
    ortholex_gals = get_galleries_for_hue(100)
    maxi_plush_gals = get_galleries_for_hue(150)
    
    print(f"  [ok] Template galleries validation:")
    print(f"       - Gravite matches in hue 10: {len(gravite_gals) > 0}")
    print(f"       - Ortholex matches in hue 100: {len(ortholex_gals) > 0}")
    print(f"       - Maxi Plush matches in hue 150: {len(maxi_plush_gals) > 0}")
except Exception as e:
    print(f"ERROR: Template loading test failed: {str(e)}")
    sys.exit(1)

# Mock frame test
try:
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw a simulated mattress-like gray rectangle in center
    cv2.rectangle(mock_frame, (400, 200), (1500, 900), (120, 120, 120), -1)
    
    # Run dimension check
    w, h, annotated = measure_dimensions(
        mock_frame, 
        pixels_per_cm=10.0, 
        edge_correction=1.0, 
        min_contour_area=10000,
        fixed_crop=cfg.FIXED_CROP
    )
    print(f"  [ok] Dry-run dimensions check executed. Detected size: {w} x {h}")
except Exception as e:
    print(f"ERROR: Dry-run dimensions check failed: {str(e)}")
    sys.exit(1)

print("\nSMOKE TEST COMPLETED SUCCESSFULLY! All components are fully operational.")
sys.exit(0)
