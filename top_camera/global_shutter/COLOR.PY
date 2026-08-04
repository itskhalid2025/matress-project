"""
Live Dimension Tracking via Webcam — Contour Detection + minAreaRect
------------------------------------------------------------------------
Continuously measures the length & breadth of the largest rectangular
object in frame (e.g., a book), using a reference object for scale.

Usage:
    python live_measure.py <reference_width_mm>

Example:
    python live_measure.py 85.6
    (85.6mm = width of a standard credit card, used as reference)

Controls:
    c   -> (re)calibrate scale using the reference object currently
            visible in frame (place ONLY the reference object, or
            keep it as the leftmost/smallest object, then press c)
    q   -> quit

Setup tips:
    - Keep the reference object (card/coin) visible in every frame,
      to the LEFT of the book, so it's picked out consistently.
    - Shoot top-down or straight-on, camera steady, book flat.
    - Plain, high-contrast background (desk mat, paper) helps a lot.
    - Even lighting -> shadows are the main cause of noisy contours.
    - Keep camera-to-book distance constant; this is 2D, not stereo.
"""

import sys
import time
import cv2
import numpy as np
from scipy.spatial import distance as dist


def order_points(pts):
    """Order 4 box points as: top-left, top-right, bottom-right, bottom-left."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_contours(frame, min_area=1500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    return cnts


def box_dims_px(c):
    """Return ordered box points + pixel width/height for a contour."""
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = order_points(np.array(box, dtype="float32"))

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    pixel_height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    pixel_width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    return box, pixel_width, pixel_height


def main():
    if len(sys.argv) >= 2:
        ref_width_mm = float(sys.argv[1])
    else:
        ref_width_mm = 85.6  # Default: 85.6mm (standard card width)
        print(f"No reference width provided. Using default: {ref_width_mm} mm")
    pixels_per_metric = None  # set on first calibration

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different camera index.")

    prev_time = time.time()

    print("Press 'c' to calibrate using the reference object in frame.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cnts = get_contours(frame)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # largest first

        key = cv2.waitKey(1) & 0xFF

        if cnts:
            # Smallest-area contour among the larger ones = likely the reference
            # (assumes reference object is physically smaller than the book)
            ref_c = min(cnts, key=cv2.contourArea)
            book_c = cnts[0] if cnts[0] is not ref_c else (cnts[1] if len(cnts) > 1 else None)

            # --- Calibrate on 'c' press ---
            if key == ord('c'):
                _, ref_px_w, _ = box_dims_px(ref_c)
                pixels_per_metric = ref_px_w / ref_width_mm
                print(f"Calibrated: {pixels_per_metric:.3f} px/mm")

            # Draw reference object (thin, yellow)
            box, _, _ = box_dims_px(ref_c)
            cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 255), 1)
            cv2.putText(frame, "REF", tuple(box[0].astype("int")),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw & measure the book (thick, green)
            if book_c is not None:
                box, px_w, px_h = box_dims_px(book_c)
                cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 3)

                if pixels_per_metric:
                    length_mm = px_h / pixels_per_metric
                    breadth_mm = px_w / pixels_per_metric
                    text = f"L:{length_mm:.1f}mm  B:{breadth_mm:.1f}mm"
                else:
                    text = "Press 'c' to calibrate (show reference only)"

                (tl, tr, br, bl) = box
                cv2.putText(frame, text, (int(tl[0]), int(tl[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "BOOK", (int(tl[0]), int(tl[1] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS overlay
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Live Dimension Tracking", frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()