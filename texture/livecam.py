import cv2
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

# ===========================================
# SETTINGS
# ===========================================

MODEL_PATH = "best_model.pth"

CLASS_NAMES = [
    "Dual harmony",
    "Gravite",
    "Maxi plush",
    "Maxi pro",
    "Memorise",
    "Ortholex",
    "Purity plus",
    "Velvet"
]

DEVICE = torch.device("cpu")

# ===========================================
# IMAGE TRANSFORM
# ===========================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ===========================================
# LOAD MODEL
# ===========================================

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model Loaded Successfully")

# ===========================================
# OPEN CAMERA
# ===========================================

# Change 0 to 1 if using USB webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot open camera")

    exit()

# ===========================================
# LIVE LOOP
# ===========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb)

    img = transform(img)

    img = img.unsqueeze(0)

    img = img.to(DEVICE)

    # ----------------------------------

    with torch.no_grad():

        output = model(img)

        probabilities = torch.softmax(output,1)

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    label = CLASS_NAMES[prediction.item()]

    confidence = confidence.item()*100

    # ----------------------------------

    text = f"{label} ({confidence:.2f}%)"

    color = (0,255,0)

    cv2.putText(

        frame,

        text,

        (20,40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        color,

        2

    )

    cv2.imshow(
        "Mattress Texture Classification",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

# ===========================================

cap.release()

cv2.destroyAllWindows()