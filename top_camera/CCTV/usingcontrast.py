"""
Mattress Dimension Checker using Color Contour Detection
===========================================================

WHY THIS WORKS
--------------
When a mattress is photographed against a CONTRASTING background
(e.g. a light mattress on a dark floor, or vice versa), the pixel
intensity/color difference at the mattress's edge is large. That
makes it easy to isolate the mattress shape ("contour") using
thresholding, then measure it.

PIPELINE
--------
1. Load image, convert to grayscale (or HSV if using a color mask)
2. Blur slightly to reduce noise
3. Threshold (Otsu's method auto-picks the best cutoff for
   high-contrast images) -> binary mask of mattress vs background
4. Morphological close/open -> remove small holes & specks
5. Find contours -> keep the largest one (assumed to be the mattress)
6. Fit a rotated bounding box (cv2.minAreaRect) -> gives width/height
   in PIXELS, correct even if mattress is slightly angled in photo
7. Convert pixels -> real units (cm/inch) using a REFERENCE OBJECT
   of known width placed in the same photo (e.g. an A4 sheet = 21cm
   wide, a ruler, or a printed marker). Without a reference object
   you can only get pixel measurements, not real-world size.
8. Draw the contour + box + dimension labels on the image and save it.

USAGE
-----
    python mattress_dimension_contour.py --image mattress.jpg \
        --ref-width 21.0

    --image        path to your photo
    --ref-width    real-world width (in cm) of the reference object
                    in the photo (leave out to only get pixel sizes)
    --units        cm or inch (default cm)
    --invert       add this flag if the mattress is DARKER than the
                    background (script defaults to mattress=lighter)

REQUIREMENTS
------------
    pip install opencv-python numpy scipy --break-system-packages
"""

import argparse
import os
import time
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Force TCP transport for RTSP stream stability
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RTSP_RECONNECT_AFTER_FAILURES = 60  # consecutive failed reads before reconnect
DEFAULT_RTSP = os.environ.get(
    "MATTRESS_RTSP_URL",
    "rtsp://admin:Admin@12345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
)  # prefer setting MATTRESS_RTSP_URL in the environment over hardcoding creds


def order_points(pts):
    """Order 4 box corners as: top-left, top-right, bottom-right, bottom-left."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def midpoint(a, b):
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def get_mattress_contour(image, invert=False):
    """Segment the mattress from a contrasting background and return
    its largest contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu's threshold auto-computes the best cutoff for a
    # bimodal histogram, which is exactly what a high-contrast
    # mattress-vs-background image produces.
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Close small gaps in the mattress edge, then remove small
    # background specks that got picked up as foreground.
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Try --invert, or check lighting/contrast.")

    # Assume the mattress is the largest contour in the frame.
    mattress_contour = max(contours, key=cv2.contourArea)
    return mattress_contour, mask


def measure_contour(image, contour, pixels_per_unit=None, units="cm"):
    """Fit a rotated bounding box to the contour and measure it."""
    box = cv2.minAreaRect(contour)          # ((cx,cy), (w,h), angle)
    box_points = cv2.boxPoints(box)
    box_points = order_points(np.array(box_points, dtype="float32"))
    (tl, tr, br, bl) = box_points

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    height_px = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    width_px = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    result = {
        "box_points": box_points,
        "width_px": width_px,
        "height_px": height_px,
        "midpoints": [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)],
    }

    if pixels_per_unit:
        result["width_real"] = width_px / pixels_per_unit
        result["height_real"] = height_px / pixels_per_unit
        result["units"] = units

    return result


def draw_result(image, mattress_result, ref_result=None):
    out = image.copy()
    box = mattress_result["box_points"].astype("int")
    cv2.drawContours(out, [box], -1, (0, 255, 0), 3)

    for (x, y) in box:
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)

    (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY) = mattress_result["midpoints"]
    cv2.line(out, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(out, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    if "width_real" in mattress_result:
        w_label = f"W: {mattress_result['width_real']:.1f} {mattress_result['units']}"
        h_label = f"H: {mattress_result['height_real']:.1f} {mattress_result['units']}"
    else:
        w_label = f"W: {mattress_result['width_px']:.0f} px"
        h_label = f"H: {mattress_result['height_px']:.0f} px"

    cv2.putText(out, w_label, (int(tlblX - 15), int(tlblY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(out, w_label, (int(tlblX - 15), int(tlblY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.putText(out, h_label, (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(out, h_label, (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    return out


def main():
    parser = argparse.ArgumentParser(description="Measure mattress dimensions via contour detection using CCTV or static image.")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP stream URL (defaults to MATTRESS_RTSP_URL env var or default RTSP URL)")
    parser.add_argument("--image", default=None, help="Path to a static mattress photo (if supplied, runs single image mode)")
    parser.add_argument("--ref-width", type=float, default=None,
                         help="Real-world width (cm) of a reference object also visible in frame.")
    parser.add_argument("--pixels-per-unit", type=float, default=None,
                         help="Skip auto reference detection: directly supply a known pixels-per-cm "
                              "(or pixels-per-inch) ratio, e.g. from a prior calibration.")
    parser.add_argument("--units", default="cm", choices=["cm", "inch"])
    parser.add_argument("--invert", action="store_true",
                         help="Use if mattress is DARKER than the background")
    parser.add_argument("--win-width", type=int, default=1024, help="Display window width in pixels (default: 1024)")
    parser.add_argument("--win-height", type=int, default=576, help="Display window height in pixels (default: 576)")
    parser.add_argument("--output", default="mattress_measured.jpg")
    args = parser.parse_args()

    # Mode 1: Static Image Processing
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

        contour, mask = get_mattress_contour(image, invert=args.invert)
        pixels_per_unit = args.pixels_per_unit
        result = measure_contour(image, contour, pixels_per_unit=pixels_per_unit, units=args.units)

        out_img = draw_result(image, result)
        cv2.imwrite(args.output, out_img)
        cv2.imwrite("mattress_mask_debug.jpg", mask)

        print(f"Mattress dimensions (pixels): {result['width_px']:.1f} x {result['height_px']:.1f}")
        if "width_real" in result:
            print(f"Mattress dimensions ({args.units}): {result['width_real']:.1f} x {result['height_real']:.1f}")
        else:
            print("No pixels-per-unit ratio given -> only pixel size reported. Pass --pixels-per-unit to convert.")
        print(f"Annotated image saved to: {args.output}")
        return

    # Mode 2: Live CCTV RTSP Video Stream
    print(f"[CCTV] Opening RTSP stream: {args.rtsp}")
    print(f"[CCTV] Window display size: {args.win_width}x{args.win_height} (resizable)")
    print("[CCTV] Controls: Press 'q' to exit live view, press 's' to save full-resolution annotated screenshot.")

    window_name = "Mattress Dimension Checker - CCTV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.win_width, args.win_height)

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    consecutive_failures = 0

    while True:
        if cap is None or not cap.isOpened():
            print("[CCTV] RTSP stream disconnected. Retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures % 10 == 0:
                print(f"[CCTV] Failed frame read count: {consecutive_failures}/{RTSP_RECONNECT_AFTER_FAILURES}")
            if consecutive_failures >= RTSP_RECONNECT_AFTER_FAILURES:
                print("[CCTV] RTSP stream unresponsive — attempting reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
            time.sleep(0.005)
            continue

        consecutive_failures = 0

        try:
            contour, mask = get_mattress_contour(frame, invert=args.invert)
            result = measure_contour(frame, contour, pixels_per_unit=args.pixels_per_unit, units=args.units)
            out_img = draw_result(frame, result)
        except Exception:
            out_img = frame.copy()
            cv2.putText(out_img, "Searching for mattress contour...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, out_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(args.output, out_img)
            print(f"Saved full-resolution annotated frame to: {args.output}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()