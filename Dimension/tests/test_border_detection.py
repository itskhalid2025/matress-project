"""
test_border_detection.py — Step 1 Black Border Detection & Overlay Test Tool.

Tests black border detection, corner vertex extraction, and 4-edge midpoint label overlays.
"""

import os
import sys
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dimension_engine import MattressDimensionEngine
from generate_test_samples import generate_synthetic_inspection_sample


def test_black_border_detection():
    print("=" * 70)
    print(" STEP 1 TEST: Black Border Detection & 4-Edge Overlay")
    print("=" * 70)

    # Generate synthetic image with a BLACK border
    sample_path, meta = generate_synthetic_inspection_sample(
        filename="test_step1_black_border.jpg",
        ref_border_cm=(100.0, 120.0),
        mattress_cm=(80.0, 100.0),
        border_color="black"
    )

    img = cv2.imread(sample_path)
    engine = MattressDimensionEngine(ref_width_cm=100.0, ref_height_cm=120.0, border_color_mode="black")

    # Update 4 edge lengths (Top=100cm, Right=120cm, Bottom=100cm, Left=120cm)
    engine.calibrator.update_edge_lengths(top_cm=100.0, right_cm=120.0, bottom_cm=100.0, left_cm=120.0)

    # Step 1: Detect Black Border & Render 4-Edge Overlay
    annotated, border_info = engine.detect_and_draw_border_overlay(img, active_edge="top")

    print(f" Border Detected : {border_info['detected']}")
    if border_info["detected"]:
        print(" 4 Corners (P1, P2, P3, P4):")
        for i, pt in enumerate(border_info["corners"]):
            print(f"   P{i+1}: ({round(pt[0], 1)}, {round(pt[1], 1)})")

        print(" Detected 4 Edges:")
        for edge_name, cm in border_info["edges"].items():
            print(f"   - {edge_name.upper()} Edge: {cm} cm")

    out_file = os.path.join(CURRENT_DIR, "step1_border_overlay.jpg")
    cv2.imwrite(out_file, annotated)
    print(f"\n Visual Overlay output saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    test_black_border_detection()
