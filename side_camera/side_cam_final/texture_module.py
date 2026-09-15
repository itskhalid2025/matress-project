import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from config import TEXTURE_MODEL_PATH, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Initialize Model Architecture
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)

if os.path.exists(TEXTURE_MODEL_PATH):
    try:
        state_dict = torch.load(TEXTURE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"[INFO] Texture AI PyTorch model loaded successfully from {TEXTURE_MODEL_PATH} on {DEVICE}.")
    except Exception as e:
        print(f"[ERROR] Failed to load PyTorch texture model: {e}")
else:
    print(f"[WARN] Texture model file not found at {TEXTURE_MODEL_PATH}")


def predict_texture(frame):
    """Returns forced PASS (100.0% confidence) per system specification."""
    return {
        "predicted_category": "PASS",
        "confidence": 100.0,
        "all_probabilities": {"PASS": 100.0}
    }
