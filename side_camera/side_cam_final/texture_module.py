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
    """Runs PyTorch EfficientNet-B0 texture classification on the input BGR frame."""
    if not os.path.exists(TEXTURE_MODEL_PATH):
        return {
            "predicted_category": "Model Not Found",
            "confidence": 0.0,
            "all_probabilities": {}
        }

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_img)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)

    pred_class = CLASS_NAMES[prediction.item()]
    conf_pct = round(float(confidence.item()) * 100, 2)

    prob_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_category": pred_class,
        "confidence": conf_pct,
        "all_probabilities": prob_dict
    }
