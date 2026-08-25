"""
generate_test_samples.py — Synthetic Test Image Benchmark Generator.

Generates realistic inspection table camera frames featuring:
- Physical reference borders (red, yellow, black, white, green tape)
- Mattresses of exact ground truth metric dimensions
- Fabric textures, specular lighting glare, and shadow gradients.
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(__file__)


def generate_synthetic_inspection_sample(
    filename="test_mattress_sample.jpg",
    table_size_px=(1280, 960),
    ref_border_cm=(100.0, 120.0),
    mattress_cm=(80.0, 100.0),
    border_color="red",
    glare_intensity=0.4
):
    """
    Generates a synthetic camera capture with known ground truth dimensions.
    """
    w_img, h_img = table_size_px
    ref_w_cm, ref_h_cm = ref_border_cm
    mat_w_cm, mat_h_cm = mattress_cm

    # Table background (dark industrial table)
    img = np.full((h_img, w_img, 3), 40, dtype=np.uint8)

    # Add table surface noise/texture
    noise = np.random.randint(-15, 15, (h_img, w_img, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Define reference frame border margin
    margin_x = 100
    margin_y = 80
    ref_w_px = w_img - (2 * margin_x)
    ref_h_px = h_img - (2 * margin_y)

    scale_x = ref_w_cm / float(ref_w_px)
    scale_y = ref_h_cm / float(ref_h_px)

    # Reference Tape Color Mapping
    color_map = {
        "red": (30, 30, 220),
        "yellow": (0, 215, 255),
        "green": (50, 200, 50),
        "blue": (220, 100, 20),
        "black": (15, 15, 15),
        "white": (240, 240, 240)
    }
    bgr_color = color_map.get(border_color.lower(), (30, 30, 220))

    # Draw Physical Reference Frame Tape
    tape_thickness = 18
    p1 = (margin_x, margin_y)
    p2 = (margin_x + ref_w_px, margin_y + ref_h_px)
    cv2.rectangle(img, p1, p2, bgr_color, tape_thickness)

    # Calculate Mattress Bounding Box Pixels based on Ground Truth CM
    mat_w_px = int(mat_w_cm / scale_x)
    mat_h_px = int(mat_h_cm / scale_y)

    # Center mattress inside reference frame
    mat_x0 = margin_x + (ref_w_px - mat_w_px) // 2
    mat_y0 = margin_y + (ref_h_px - mat_h_px) // 2
    mat_x1 = mat_x0 + mat_w_px
    mat_y1 = mat_y0 + mat_h_px

    # Draw Mattress (White fabric with light quilted texture)
    mat_fabric = np.full((mat_h_px, mat_w_px, 3), 220, dtype=np.uint8)
    # Quilted pattern lines
    for y in range(0, mat_h_px, 30):
        cv2.line(mat_fabric, (0, y), (mat_w_px, y), (200, 200, 200), 1)
    for x in range(0, mat_w_px, 30):
        cv2.line(mat_fabric, (x, 0), (x, mat_h_px), (200, 200, 200), 1)

    # Add printed brand text/logo pattern on fabric
    cv2.putText(mat_fabric, "MATTRESS PRO-LINE", (mat_w_px // 4, mat_h_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (170, 170, 170), 2)

    img[mat_y0:mat_y1, mat_x0:mat_x1] = mat_fabric

    # Simulate Specular Lighting Glare spots across the mattress
    if glare_intensity > 0:
        glare_mask = np.zeros((h_img, w_img), dtype=np.float32)
        center_glare = (mat_x0 + int(mat_w_px * 0.4), mat_y0 + int(mat_h_px * 0.3))
        cv2.circle(glare_mask, center_glare, 140, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (101, 101), 0)

        glare_3ch = np.dstack([glare_mask * 255 * glare_intensity] * 3).astype(np.uint8)
        img = cv2.add(img, glare_3ch)

    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)
    print(f"[generate_test_samples] Generated benchmark image: {save_path}")
    print(f"                       Ground Truth Mattress: {mat_w_cm} cm x {mat_h_cm} cm")
    print(f"                       Reference Border: {ref_w_cm} cm x {ref_h_cm} cm ({border_color.upper()} tape)")

    return save_path, {
        "ground_truth_width_cm": mat_w_cm,
        "ground_truth_length_cm": mat_h_cm,
        "ref_width_cm": ref_w_cm,
        "ref_height_cm": ref_h_cm,
        "border_color": border_color
    }


if __name__ == "__main__":
    generate_synthetic_inspection_sample("test_red_tape.jpg", border_color="red")
    generate_synthetic_inspection_sample("test_yellow_tape.jpg", border_color="yellow")
    generate_synthetic_inspection_sample("test_black_tape.jpg", border_color="black")
