import cv2

# Try index 1 first (usually the external USB camera on a Mac)
# If it fails, change this to 0 (built-in camera) or 2.
CAMERA_INDEX = 0

def test_pc_webcam():
    print(f"Opening camera at index {CAMERA_INDEX}...")
    
    # AVFoundation is the native macOS backend, OpenCV selects it by default
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Request 720p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}. Try changing the index to 0.")
        return

    print("Camera opened successfully! Press 'q' on your keyboard to exit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            continue
            
        # Display the resulting frame in a window
        cv2.imshow(f"Webcam Test - Index {CAMERA_INDEX}", frame)

        # Wait for 1 ms and check if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pc_webcam()