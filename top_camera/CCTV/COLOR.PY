# Source - https://stackoverflow.com/q/77457415
# Posted by yong jie, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-04, License - CC BY-SA 4.0

import cv2
import numpy as np

def get_object_dimensions(contour, reference_width, reference_distance):
    # Calculate the bounding box around the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Assuming the camera calibration is done, convert pixel dimensions to centimeters
    pixel_width = w
    pixel_height = h
    # You may need to calibrate these conversion factors based on your camera and setup
    width_cm = pixel_width * pixel_to_cm_width
    height_cm = pixel_height * pixel_to_cm_height

    # Calculate the current distance based on the reference width and current measured width
    current_distance = (reference_width * reference_distance) / width_cm

    return width_cm, height_cm, current_distance

# Camera setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Calibration factors for converting pixel dimensions to centimeters
pixel_to_cm_width = 0.1  # Adjust based on your calibration
pixel_to_cm_height = 0.1  # Adjust based on your calibration

# Initial reference distance (distance at which the width measurement is accurate)
initial_reference_distance = 50.0  # Adjust based on your setup
reference_width = 10.0  # Adjust based on your setup

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over detected contours
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 1000:
            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get object dimensions and current distance in centimeters
            width_cm, height_cm, current_distance = get_object_dimensions(contour, reference_width, initial_reference_distance)

            # Display the dimensions and current distance
            cv2.putText(frame, f'Width: {width_cm:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Height: {height_cm:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {current_distance:.2f} cm', (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection and Measurement', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows() 
    