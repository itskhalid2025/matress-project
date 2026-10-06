import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on PIL image.
    Enhances subtle texture and fabric weave details in mattress images.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            # Grayscale
            enhanced = self.clahe.apply(img_np)
            return Image.fromarray(enhanced)
        
        # RGB: convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_enhanced)

class SharpenTransform:
    """
    Applies subtle unsharp masking filter to highlight fabric edge contrast.
    """
    def __init__(self, alpha=1.5, beta=-0.5):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (5, 5), 1.0)
        sharpened = cv2.addWeighted(img_np, self.alpha, blurred, self.beta, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened)

def get_transforms(pipeline_name="standard", image_size=224):
    """
    Returns train and val transforms based on requested preprocessing pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if pipeline_name == "standard":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif pipeline_name == "clahe_enhanced":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            CLAHETransform(clip_limit=2.5, tile_grid_size=(8, 8)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            CLAHETransform(clip_limit=2.5, tile_grid_size=(8, 8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif pipeline_name == "heavy_aug":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.20, p=0.30),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif pipeline_name == "edge_sharpened":
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            SharpenTransform(alpha=1.4, beta=-0.4),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            SharpenTransform(alpha=1.4, beta=-0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        raise ValueError(f"Unknown pipeline_name: {pipeline_name}")

    return train_transform, val_transform
