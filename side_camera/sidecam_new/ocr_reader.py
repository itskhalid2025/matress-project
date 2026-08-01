"""
ocr_reader.py — Tesseract OCR Reader Module for sidecam_new.

Performs Tesseract OCR on the cropped label image after preprocessing:
  - Grayscale
  - CLAHE
  - Adaptive / Otsu Thresholding
  - Noise Removal
"""

import os
import cv2
import numpy as np

# Configure pytesseract path safely for Windows
try:
    import pytesseract
    # Default Windows installation path
    TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_WIN_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_WIN_PATH
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def preprocess_for_ocr(image):
    """
    Applies image preprocessing pipeline tailored for Tesseract OCR:
    Grayscale -> CLAHE -> Thresholding -> Noise Removal
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Contrast Enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Otsu Binarization
    _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Adaptive Thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Combine both binary images for crisp character shapes
    combined = cv2.bitwise_and(thresh_otsu, thresh_adaptive)

    # 4. Noise Removal using Median Blur
    denoised = cv2.medianBlur(combined, 3)

    return denoised, gray


def read_label_ocr(label_image):
    """
    Runs Tesseract OCR on the cropped label ROI and returns the extracted text string.
    """
    if not HAS_PYTESSERACT:
        print("[OCR Warning] pytesseract module is not installed.")
        return "OCR Module Not Installed"

    if label_image is None or label_image.size == 0:
        return ""

    # Preprocess cropped label
    processed, gray = preprocess_for_ocr(label_image)

    extracted_text = ""
    
    # Try reading from preprocessed binary image first
    try:
        config_options = "--psm 6"  # Assume uniform block of text
        extracted_text = pytesseract.image_to_string(processed, lang="eng", config=config_options)
    except Exception as e:
        print(f"[OCR Exception on processed image]: {e}")

    # Fallback to grayscale if binary image yield is very low
    if not extracted_text or len(extracted_text.strip()) < 5:
        try:
            extracted_text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
        except Exception as e:
            print(f"[OCR Exception on gray image]: {e}")

    cleaned = extracted_text.strip()
    return cleaned if cleaned else "No OCR text detected"


def print_ocr_results(ocr_text):
    """Prints OCR output to terminal formatted as requested."""
    print("\n=========================")
    print("OCR RESULT\n")
    print(ocr_text if ocr_text else "No text extracted")
    print("=========================\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        txt = read_label_ocr(img)
        print_ocr_results(txt)
