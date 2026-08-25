import cv2
import numpy as np

# 1. Hardcoded RTSP URL with URL-encoded password (@ replaced with %40)
RTSP_URL = "rtsp://admin:Admin%4012345@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"

def process_stream():
    # Connect to the IP Camera
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Could not open the RTSP stream. Check network and credentials.")
        return

    print("Stream connected successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Stream may have disconnected.")
            break

        # Resize frame to reduce computation and prevent lag (Optional but recommended)
        # You can change these dimensions based on your needs
        frame = cv2.resize(frame, (800, 600))

        # 2. Convert to Grayscale (simplifies processing for white objects)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Apply heavy Gaussian Blur to smooth out the marble veins and cardboard textures
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 4. Thresholding: Separates the white mattress from the dark background
        # You may need to tweak the '160' value depending on your camera's exact lighting
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

        # 5. Morphological Operations: Close gaps and remove small background noise
        kernel = np.ones((9, 9), np.uint8)
        # MORPH_OPEN removes small bright spots in the dark background
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # MORPH_CLOSE fills in dark spots inside the bright mattress (like logos/lines)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 6. Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize an empty, pitch-black mask
        mattress_mask = np.zeros_like(gray)

        if contours:
            # Assume the mattress is the largest white object in the frame
            largest_contour = max(contours, key=cv2.contourArea)

            # Optional: Filter out tiny flashes of light by setting a minimum area size
            if cv2.contourArea(largest_contour) > 5000:
                # 7. Draw a SOLID fill over the contour boundary to bypass all internal patterns/logos
                cv2.drawContours(mattress_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 8. Create the Inverse Mask (Background isolation)
        background_mask = cv2.bitwise_not(mattress_mask)

        # --- Visualizing the Results (Optional but helpful for debugging) ---
        
        # Apply the masks to the original frame to extract the physical pixels
        mattress_only = cv2.bitwise_and(frame, frame, mask=mattress_mask)
        background_only = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Display windows
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Foreground (Solid Mattress Mask)", mattress_mask)
        cv2.imshow("Background (Inverse Mask)", background_mask)
        cv2.imshow("Extracted Mattress", mattress_only)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_stream()