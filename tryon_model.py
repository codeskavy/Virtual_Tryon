# tryon_model.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# Resize + ToTensor
def preprocess_image(image_path, size=(128, 128)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Find gray region (agnostic)
def create_mask(image, region_color=(128, 128, 128)):
    image_np = np.array(image)
    mask = np.all(image_np == region_color, axis=-1)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return mask


# Apply overlay on gray area
def apply_mask(base_image, overlay_image, mask):
    if base_image.size() != mask.size():
        mask = F.interpolate(mask.unsqueeze(0), size=base_image.shape[2:], mode='bilinear', align_corners=False).squeeze(0)
    masked_overlay = overlay_image * mask
    result_image = base_image * (1 - mask) + masked_overlay
    return result_image


# Main pipeline
def generate_tryon_result(base_image_path, mask_image_path, overlay_image_path):
    base_image = preprocess_image(base_image_path)
    overlay_image = preprocess_image(overlay_image_path)

    base_image_pil = transforms.ToPILImage()(base_image.squeeze(0))
    mask = create_mask(base_image_pil)

    result_tensor = apply_mask(base_image, overlay_image, mask)
    result_image = transforms.ToPILImage()(result_tensor.squeeze(0))

    return result_image
