import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def create_agnostic_mask(original_image, mask_image):
    """Create agnostic mask from human image"""
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image
        
    if isinstance(mask_image, Image.Image):
        mask_np = np.array(mask_image.convert('L'))
    else:
        mask_np = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) if len(mask_image.shape) == 3 else mask_image
    
    _, mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    grey_image = np.ones_like(original_np) * 128
    
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
    inv_mask_3d = 1 - mask_3d
    
    masked_grey = grey_image * mask_3d
    masked_original = original_np * inv_mask_3d
    result_image = masked_grey + masked_original
    
    return result_image.astype(np.uint8)

def preprocess_image_for_torch(image, size=(128, 128)):
    """Convert PIL image to torch tensor"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def combine_cloth_with_mask(cloth_image, mask_image):
    """Combine clothing with mask"""
    cloth_tensor = preprocess_image_for_torch(cloth_image)
    mask_tensor = preprocess_image_for_torch(mask_image.convert('RGB'))
    
    output = cloth_tensor * (mask_tensor > 0.5).float()
    output_image = transforms.ToPILImage()(output.squeeze())
    return output_image