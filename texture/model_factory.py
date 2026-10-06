import torch
import torch.nn as nn
import torchvision.models as models
import timm

def count_parameters(model):
    """Returns total trainable parameter count in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model(model_name="efficientnet_b0", num_classes=8, pretrained=True):
    """
    Factory function to instantiate models for texture classification.
    Supports EfficientNet-B0, ResNet18, ResNet50, MobileNetV3-Large, ConvNeXt-Tiny, ViT-Tiny.
    """
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)

    elif model_name == "vit_tiny":
        # Using timm lightweight vision transformer
        model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
