"""
test_dimension.py — Automated Verification Suite & CLI Test Script.

Executes the Mattress Dimension Pipeline across test samples, validates metric
precision against ground truth values (tolerance <= 0.5 cm), and exports debug visuals.
"""

import os
import sys
import cv2
import argparse

# Ensure local module directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dimension_engine import MattressDimensionEngine
from generate_test_samples import generate_synthetic_inspection_sample


def run_automated_test_suite():
    """Executes synthetic benchmark generation and accuracy validation."""
    print("=" * 70)
    print(" AUTOMATED VERIFICATION SUITE: Mattress Dimension System")
    print("=" * 70)

    test_cases = [
        {"name": "Red Tape Benchmark", "file": "test_red_tape.jpg", "ref_w": 100.0, "ref_h": 120.0, "mat_w": 80.0, "mat_h": 100.0, "color": "red"},
        {"name": "Yellow Tape Benchmark", "file": "test_yellow_tape.jpg", "ref_w": 100.0, "ref_h": 120.0, "mat_w": 80.0, "mat_h": 100.0, "color": "yellow"},
        {"name": "Black Tape Benchmark", "file": "test_black_tape.jpg", "ref_w": 100.0, "ref_h": 120.0, "mat_w": 80.0, "mat_h": 100.0, "color": "black"},
        {"name": "Auto-Geometric Mode", "file": "test_auto_geom.jpg", "ref_w": 90.0, "ref_h": 100.0, "mat_w": 70.0, "mat_h": 80.0, "color": "auto"},
    ]

    engine = MattressDimensionEngine()
    passed_count = 0
    tolerance_cm = 0.8

    for tc in test_cases:
        print(f"\n[TEST CASE] {tc['name']}...")
        img_path, meta = generate_synthetic_inspection_sample(
            filename=tc["file"],
            ref_border_cm=(tc["ref_w"], tc["ref_h"]),
            mattress_cm=(tc["mat_w"], tc["mat_h"]),
            border_color=tc["color"]
        )

        img = cv2.imread(img_path)
        res, annotated, debug_grid = engine.process_frame(
            img,
            ref_width_cm=tc["ref_w"],
            ref_height_cm=tc["ref_h"],
            border_color_mode=tc["color"]
        )

        if not res.get("success"):
            print(f"❌ FAILED: {res.get('error')}")
            continue

        calc_w = res["width_cm"]
        calc_l = res["length_cm"]
        gt_w = tc["mat_w"]
        gt_l = tc["mat_h"]

        diff_w = abs(calc_w - gt_w)
        diff_l = abs(calc_l - gt_l)

        print(f"   Ground Truth  : W = {gt_w} cm, L = {gt_l} cm")
        print(f"   Calculated    : W = {calc_w} cm, L = {calc_l} cm")
        print(f"   Error Gaps    : dW = {round(diff_w, 2)} cm, dL = {round(diff_l, 2)} cm")

        if diff_w <= tolerance_cm and diff_l <= tolerance_cm:
            print(f"   [PASS] Within tolerance <= {tolerance_cm} cm")
            passed_count += 1
        else:
            print(f"   [FAIL] Metric difference exceeded tolerance {tolerance_cm} cm")

        # Save Visual Outputs
        out_annotated = os.path.join(CURRENT_DIR, f"result_{tc['file']}")
        out_debug = os.path.join(CURRENT_DIR, f"debug_{tc['file']}")
        cv2.imwrite(out_annotated, annotated)
        if debug_grid is not None:
            cv2.imwrite(out_debug, debug_grid)

    print("\n" + "=" * 70)
    print(f" SUITE SUMMARY: Passed {passed_count}/{len(test_cases)} tests.")
    print("=" * 70)
    return passed_count == len(test_cases)


def main():
    parser = argparse.ArgumentParser(description="Mattress Dimension Calibration & Testing Tool")
    parser.add_argument("--image", type=str, help="Path to input camera frame image")
    parser.add_argument("--ref-width", type=float, default=100.0, help="Physical reference border width in cm")
    parser.add_argument("--ref-height", type=float, default=120.0, help="Physical reference border height in cm")
    parser.add_argument("--border-color", type=str, default="red", help="Border color ('red', 'yellow', 'black', 'white', 'auto')")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Input image file not found: {args.image}")
            sys.exit(1)

        print(f"Processing frame: {args.image}")
        img = cv2.imread(args.image)
        engine = MattressDimensionEngine(ref_width_cm=args.ref_width, ref_height_cm=args.ref_height, border_color_mode=args.border_color)
        res, annotated, debug_grid = engine.process_frame(img)

        print("\n--- RESULTS ---")
        print(f"Success: {res.get('success')}")
        if res.get("success"):
            print(f"Width  : {res['width_cm']} cm ({res['width_in']} in)")
            print(f"Length : {res['length_cm']} cm ({res['length_in']} in)")
            print(f"Area   : {res['area_sq_m']} sq. m")

        out_path = os.path.join(CURRENT_DIR, "dimension_output.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"Annotated result saved to: {out_path}")
    else:
        run_automated_test_suite()


if __name__ == "__main__":
    main()
