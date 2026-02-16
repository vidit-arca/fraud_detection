import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pdf2image import convert_from_path
import os

def get_preprocessing_transforms(augment=False):
    """
    Returns preprocessing transforms.
    When augment=True (training), applies data augmentation to increase effective
    training set size and improve model robustness.
    """
    base_transforms = [
        transforms.Resize((256, 256)),  # Larger base resolution
    ]
    
    if augment:
        base_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),  # Slight jitter
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    
    base_transforms.extend([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(base_transforms)


def create_patches(image_tensor, grid_size=(5, 5)):
    """
    Splits the image tensor (C, H, W) into a grid of patches (N*M, C, h, w).
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
        
    c, h, w = image_tensor.shape
    n_rows, n_cols = grid_size
    patch_h, patch_w = h // n_rows, w // n_cols
    
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            start_y, start_x = i * patch_h, j * patch_w
            patch = image_tensor[:, start_y:start_y+patch_h, start_x:start_x+patch_w]
            patches.append(patch)
            
    return torch.stack(patches)


def create_multiscale_patches(image_tensor, grid_sizes=None, target_patch_size=(32, 32)):
    """
    Creates patches at multiple scales for richer representation.
    Default: 3x3 (coarse) + 5x5 (medium) + 7x7 (fine) = 83 patches.
    All patches are resized to target_patch_size for uniform feature extraction.
    """
    if grid_sizes is None:
        grid_sizes = [(3, 3), (5, 5), (7, 7)]
    
    all_patches = []
    for gs in grid_sizes:
        patches = create_patches(image_tensor, grid_size=gs)
        # Resize all patches to uniform size
        patches = F.interpolate(patches, size=target_patch_size, mode='bilinear', align_corners=False)
        all_patches.append(patches)
    
    return torch.cat(all_patches, dim=0)


def load_and_preprocess_image(image_path, transform=None, augment=False):
    """
    Loads an image (or PDF) and applies the transformations.
    """
    if transform is None:
        transform = get_preprocessing_transforms(augment=augment)
        
    try:
        if image_path.lower().endswith('.pdf'):
            images = convert_from_path(image_path, first_page=1, last_page=1)
            if not images:
                print(f"Error: No images extracted from PDF {image_path}")
                return None
            image = images[0].convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
            
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
