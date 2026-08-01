import os
import cv2
from qrtest import inspect_qr_code

# Folder containing captured images
CAPTURE_FOLDER = "C:\matress-project-matress\side_camera\sidecam_new\captures"

# Get the newest image
images = [
    os.path.join(CAPTURE_FOLDER, f)
    for f in os.listdir(CAPTURE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    print("❌ No images found inside captures/")
    exit()

latest_image = max(images, key=os.path.getctime)

print("=" * 60)
print("Testing Image:")
print(latest_image)
print("=" * 60)

# Load image
img = cv2.imread(latest_image)

if img is None:
    print("❌ Failed to load image.")
    exit()

print(f"Image Size : {img.shape[1]} x {img.shape[0]}")

# Run QR inspection
result = inspect_qr_code(latest_image)

print("\n========== QR RESULT ==========")

if result["qr_found"]:
    print("✅ QR FOUND")
    print(f"Product Name      : {result['product_name']}")
    print(f"Batch Number      : {result['batch_no']}")
    print(f"Inventory Item ID : {result['inventory_item_id']}")
    print(f"Raw QR            : {result['raw_text']}")
else:
    print("❌ QR NOT FOUND")

print("===============================\n")

# Display the image
cv2.imshow("Captured Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()