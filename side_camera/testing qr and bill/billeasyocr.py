import platform
import cv2
import easyocr
import numpy as np

# Initialize EasyOCR. Set gpu=False if there's no CUDA GPU available
# (e.g. once this runs on a Raspberry Pi) -- EasyOCR falls back to CPU on
# its own either way, but setting it explicitly skips the CUDA check and
# avoids a startup warning.
reader = easyocr.Reader(['en'], gpu=True)

# Global flag to trigger OCR processing
trigger_ocr = False

# Mouse callback function to detect button clicks
def click_button(event, x, y, flags, param):
    global trigger_ocr
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is inside the button boundaries (x: 10 to 190, y: 10 to 50)
        if 10 <= x <= 190 and 10 <= y <= 50:
            trigger_ocr = True

# CAP_DSHOW is a Windows-only backend. Pick V4L2 automatically on Linux
# (Raspberry Pi included) so this doesn't silently fail to open the camera
# later on.
backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
cap = cv2.VideoCapture(0, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam. Check the device index and connection.")

# Create a named window and bind the mouse click function to it
cv2.namedWindow('Webcam OCR')
cv2.setMouseCallback('Webcam OCR', click_button)

print("Click the 'Do Processing' button on the camera screen to run OCR.")
print("Press 'q' on your keyboard to quit.")

def preprocess(img):
    """Enhance text contrast using CLAHE and high-pass sharpening."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 2)
    # Combine original and blurred for a high-contrast sharpened output
    sharpened = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Save a clean copy of the frame before drawing the UI overlays
    clean_frame = frame.copy()

    # 1. Create the visual "Do Processing" button overlay (Top-Left corner)
    cv2.rectangle(frame, (10, 10), (190, 50), (0, 255, 0), -1)
    cv2.putText(frame, "Do Processing", (25, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 2. Check if the button was clicked
    if trigger_ocr:
        print("\n--- Processing bill text... ---")

        p = preprocess(clean_frame)

        # rotation_info tells EasyOCR to test each DETECTED text box at
        # 90/180/270 degrees and keep whichever orientation the recognizer
        # scores highest on. Detection still runs once on the frame as
        # captured, so this replaces four full readtext() passes with one --
        # and it handles a bill that's sideways or upside down without you
        # needing to pre-rotate the whole image and guess which rotation
        # "won" overall.
        results = reader.readtext(p, rotation_info=[90, 180, 270])

        annotated = clean_frame.copy()

        if not results:
            print("No text found. Check lighting and focus.")
        else:
            print("--- OCR Output ---")
            for (bbox, text, prob) in results:
                clean_text = text.strip()
                # Filter out obvious low-confidence/single-character noise
                if len(clean_text) <= 1 and not clean_text.isdigit():
                    continue
                if prob < 0.25:
                    continue

                print(f"{clean_text} (Confidence: {prob:.2f})")

                # Box coordinates are relative to the frame you passed in
                # (clean_frame), regardless of which internal rotation the
                # recognizer picked for that box -- so they draw straight
                # onto it, no re-rotation bookkeeping needed.
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_text, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show the annotated image for 4 seconds
            cv2.imshow('Webcam OCR', annotated)
            cv2.waitKey(4000)

        # Reset the click trigger
        trigger_ocr = False

    # Show live feed (if not paused for showing results)
    cv2.imshow('Webcam OCR', frame)

    # Break loop if 'q' is pressed on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()