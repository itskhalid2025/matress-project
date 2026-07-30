"""
capture_and_analyze_gs.py -- single-capture, dual-pipeline runner for the
Raspberry Pi Global Shutter camera.

Captures ONE frame from the GS camera via picamera2, then runs, on that
exact same frame, concurrently:
  1. Dimension measurement (YOLO + OpenCV hybrid, refactored from
     top_test2_gs.py to accept an already-captured frame)
  2. Banner SKU identification (mattress.banner_ocr.read_banner, UNCHANGED
     -- it is camera-agnostic and needs no GS-specific edits)

Run via a thread pool, not a process pool: neither pipeline depends on the
other's output, and both are compute-heavy in ways that release Python's
GIL for the actual work (YOLO's underlying torch/C++ ops; pytesseract
shells out to the tesseract binary as a subprocess). Real wall-clock
overlap is expected here, not just cosmetic threading -- but this hasn't
been benchmarked on this specific Pi, so treat the reported elapsed time
as informative, not a guaranteed speedup figure.

============================================================================
CALIBRATION / VERIFICATION STATUS -- read before trusting any output
============================================================================
  - PIXELS_PER_CM (dimension measurement): NOT calibrated for the GS
    camera. This script will raise a clear error until you set it -- see
    the docstring on measure_dimensions() for the recalibration procedure.
    Do NOT reuse a value derived for a different camera/lens/mount height.

  - Banner OCR thresholds (mattress/config.py: SASH_S_THRESH,
    SASH_V_THRESH; banner_ocr.py: WEAK_MAGENTA, WEAK_ORANGE_MAX,
    BAND_HUE_TOL) were calibrated against reference frames from the OLD
    top webcam's color/exposure response. They are UNVERIFIED against the
    GS camera and may need re-tuning. This script does not change or
    re-derive them -- it only wires the GS camera into the existing,
    unmodified pipeline. Run a calibration pass (same spirit as
    qr_feasibility.py: measure real read-rate per SKU against real GS
    captures) before trusting banner results from this camera.

  - CAM_WIDTH/CAM_HEIGHT below default to 1456x1088 (typical for a Sony
    IMX296-based GS module) -- NOT confirmed against this specific board.
    Run `libcamera-hello --list-cameras` and correct these if they don't
    match.

  - mattress/config.py enforces CAPTURE_W=1920/CAPTURE_H=1080 for the
    TEXTURE-IDENTIFICATION pipeline specifically (crop.py's
    localise_cover() hard-rejects any other resolution). The GS camera's
    native 1456x1088 does NOT match this. This script only runs dimension
    measurement + banner OCR -- it does NOT touch texture ID, QR, or
    reconcile. If the GS camera is ever meant to replace the top webcam
    for texture ID too, that resolution mismatch needs its own resolution
    (config change + full re-enrollment against GS reference photos), not
    an assumption that it'll just work.

Run inside the main project venv (source ~/Matress-project/venv/bin/activate)
-- NOT experiments/New/myenv, which is the separate TensorFlow environment
for the unrelated deep-learning track and doesn't have ultralytics/picamera2
wired the same way.
"""

import sys
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- import the banner OCR module from the mattress package -----------
# Adjust _POC_FINAL below if this script doesn't sit next to
# mattress_poc_final/ in your actual layout (e.g. if you place this
# inside legacy_scripts/, mattress_poc_final/ is one level up -- that's
# the default assumed here).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_POC_FINAL = os.path.join(_THIS_DIR, "..", "mattress_poc_final")
if _POC_FINAL not in sys.path:
    sys.path.insert(0, _POC_FINAL)

try:
    from banner_ocr import read_banner
except ImportError as e:
    raise ImportError(
        f"Could not import mattress.banner_ocr ({e}). This script expects "
        f"to sit next to mattress_poc_final/ -- adjust _POC_FINAL above if "
        f"your layout differs, or run from the correct directory."
    )


# ============================================================
# GS CAMERA CONFIGURATION
# ============================================================
CAM_WIDTH = 1456   # <-- verify with `libcamera-hello --list-cameras`
CAM_HEIGHT = 1088

# PLACEHOLDER -- see measure_dimensions() docstring for recalibration steps.
PIXELS_PER_CM = None
EDGE_CORRECTION_FACTOR = 1.1
# ============================================================


def capture_frame(lock_exposure=False, exposure_us=None, gain=None):
    """Grab exactly one frame from the GS camera. Returns a BGR numpy array
    usable directly by both consumers (BGR888 output needs no cvtColor).

    lock_exposure: disables auto-exposure/auto-gain and applies the given
    exposure_us/gain instead. Recommended for a fixed industrial setup
    with consistent lighting, so OCR/threshold results don't drift frame
    to frame -- but this requires you to have already found good values
    for your actual lighting. Passing lock_exposure=True with no explicit
    values just freezes whatever the camera happened to auto-converge to,
    which is not necessarily a good value.
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let auto-exposure/gain settle before any capture

    if lock_exposure:
        controls = {"AeEnable": False}
        if exposure_us is not None:
            controls["ExposureTime"] = exposure_us
        if gain is not None:
            controls["AnalogueGain"] = gain
        picam2.set_controls(controls)
        time.sleep(0.3)  # let manual controls take effect

    frame = picam2.capture_array()
    picam2.stop()
    return frame


def measure_dimensions(img, model):
    """Dimension measurement (YOLO mask + OpenCV geometry), refactored from
    top_test2_gs.py to accept an already-captured frame instead of grabbing
    its own -- so it can share exactly one frame with the banner OCR call.

    Returns {"width_cm": float, "height_cm": float, "annotated": <BGR
    image copy with overlay drawn>}, or None if no shape matched.

    RECALIBRATION REQUIRED before trusting numbers from this:
      1. Run against a flat object of KNOWN width, at production mount
         height/angle.
      2. Measure the object's pixel width in the saved annotated image.
      3. PIXELS_PER_CM = measured_pixel_width / known_width_cm
      4. Set the module-level PIXELS_PER_CM above.
    """
    if PIXELS_PER_CM is None:
        raise RuntimeError(
            "PIXELS_PER_CM not calibrated for this camera yet -- see "
            "measure_dimensions() docstring for the procedure."
        )

    out = img.copy()
    img_area = out.shape[0] * out.shape[1]
    imgGray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    results = model.predict(source=out, save=False, conf=0.15)
    yolo_mask = np.zeros(out.shape[:2], dtype=np.uint8)

    valid_box = None
    max_box_area = 0
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id == 0:  # skip humans so feet/legs don't ruin the mask
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_box_area:
                max_box_area = area
                valid_box = box

    if valid_box is not None:
        x1, y1, x2, y2 = valid_box.xyxy[0].cpu().numpy()
        pad = 40
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(out.shape[1], int(x2) + pad)
        y2 = min(out.shape[0], int(y2) + pad)
        cv2.rectangle(yolo_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    else:
        yolo_mask[:] = 255

    imgBlur = cv2.GaussianBlur(imgGray, (9, 9), 0)
    _, imgThre = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    imgClean = cv2.erode(imgThre, kernel, iterations=1)
    imgClean = cv2.dilate(imgClean, kernel, iterations=2)

    imgFinalMask = cv2.bitwise_and(imgClean, imgClean, mask=yolo_mask)
    contours, _ = cv2.findContours(imgFinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        area = cv2.contourArea(c)
        if 50000 < area < (img_area * 0.75):
            hull = cv2.convexHull(c)
            cv2.drawContours(out, [hull], -1, (0, 0, 255), 2)
            rect = cv2.minAreaRect(hull)
            (cx, cy), (w, h), angle = rect

            raw_w = w / PIXELS_PER_CM
            raw_h = h / PIXELS_PER_CM
            nW = round(raw_w * EDGE_CORRECTION_FACTOR, 1)
            nH = round(raw_h * EDGE_CORRECTION_FACTOR, 1)

            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(out, [box], 0, (0, 255, 0), 3)
            cv2.putText(out, f'W: {nW} cm', (int(cx) - 100, int(cy) - 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)
            cv2.putText(out, f'H: {nH} cm', (int(cx) - 100, int(cy) + 40),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)

            return {"width_cm": nW, "height_cm": nH, "annotated": out}

    return None


def _run_dimensions_safe(img, model):
    """Wrapper so a raised exception (e.g. uncalibrated PIXELS_PER_CM)
    doesn't take down the whole run -- captured and returned instead, so
    the banner result can still be reported."""
    try:
        return {"ok": True, "result": measure_dimensions(img, model)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _run_banner_safe(img):
    try:
        sku, angle, text, tried = read_banner(img)
        return {"ok": True, "sku": sku, "angle": angle,
                "raw_text": text, "attempts": len(tried)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Capture one GS-camera frame, then run dimension "
                    "measurement and banner OCR on it in parallel."
    )
    parser.add_argument("--lock-exposure", action="store_true",
                        help="Disable auto-exposure/gain (recommended once "
                             "you've found good fixed values for your "
                             "production lighting).")
    parser.add_argument("--exposure-us", type=int, default=None,
                        help="Manual exposure time in microseconds "
                             "(only applied with --lock-exposure).")
    parser.add_argument("--gain", type=float, default=None,
                        help="Manual analogue gain "
                             "(only applied with --lock-exposure).")
    parser.add_argument("--save", default="gs_capture.jpg",
                        help="Where to save the raw captured frame.")
    args = parser.parse_args()

    print("Capturing frame from GS camera...")
    frame = capture_frame(lock_exposure=args.lock_exposure,
                          exposure_us=args.exposure_us, gain=args.gain)
    cv2.imwrite(args.save, frame)
    print(f"Saved raw capture to {args.save}")

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("Running dimension measurement + banner OCR in parallel...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_dim = ex.submit(_run_dimensions_safe, frame, model)
        fut_banner = ex.submit(_run_banner_safe, frame)
        dim_outcome = fut_dim.result()
        banner_outcome = fut_banner.result()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.2f}s\n")

    print("=== Dimension measurement ===")
    if not dim_outcome["ok"]:
        print(f"  FAILED: {dim_outcome['error']}")
    elif dim_outcome["result"] is None:
        print("  No shape matched -- check the saved capture manually.")
    else:
        r = dim_outcome["result"]
        print(f"  Width:  {r['width_cm']} cm")
        print(f"  Height: {r['height_cm']} cm")
        annotated_path = "gs_dimensions_annotated.jpg"
        cv2.imwrite(annotated_path, r["annotated"])
        print(f"  Annotated image saved to {annotated_path}")

    print("\n=== Banner OCR ===")
    if not banner_outcome["ok"]:
        print(f"  FAILED: {banner_outcome['error']}")
    else:
        print(f"  SKU:      {banner_outcome['sku']}")
        print(f"  Raw text: {banner_outcome['raw_text']!r}")
        print(f"  Attempts tried: {banner_outcome['attempts']}")


if __name__ == "__main__":
    main()