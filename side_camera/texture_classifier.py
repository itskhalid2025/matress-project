"""
texture_classifier.py — PyTorch EfficientNet-B0 Mattress Fabric Texture Classifier.

Loads `texture/best_model.pth` and predicts mattress model from fabric quilting pattern.
"""

import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

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

SKU_MAP = {
    "Dual harmony": "dual_harmony",
    "Gravite": "gravite",
    "Maxi plush": "maxi_plush",
    "Maxi pro": "maxi_pro",
    "Memorise": "memorise",
    "Ortholex": "ortholex",
    "Purity plus": "purity_plus",
    "Velvet": "velvet"
}


class TextureClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default path relative to workspace
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "texture"))
            model_path = os.path.join(base_dir, "best_model.pth")

        self.model_path = model_path
        self.device = torch.device("cpu")
        self.available = False
        self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[TextureClassifier] Model weights not found at {self.model_path}")
            return

        try:
            model = efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features,
                len(CLASS_NAMES)
            )
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.available = True
            print("[TextureClassifier] EfficientNet_B0 PyTorch Model Loaded Successfully!")
        except Exception as e:
            print(f"[TextureClassifier] Failed to load model: {e}")
            self.available = False

    def predict(self, frame_bgr):
        """
        Classifies fabric texture from a BGR image frame.
        
        Returns:
            dict: {
                "sku": canonical_sku_str,
                "class_name": raw_class_name,
                "confidence": float (0-100)
            } or None if unavailable/error
        """
        if not self.available or self.model is None or frame_bgr is None:
            return None

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor_img)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

            confidence = round(float(conf.item()) * 100.0, 1)
            raw_class = CLASS_NAMES[pred.item()]
            canonical_sku = SKU_MAP.get(raw_class, raw_class.lower().replace(" ", "_"))

            return {
                "sku": canonical_sku,
                "class_name": raw_class,
                "confidence": confidence
            }
        except Exception as e:
            print(f"[TextureClassifier] Prediction error: {e}")
            return None
