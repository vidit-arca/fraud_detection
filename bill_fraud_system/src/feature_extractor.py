import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import io
import cv2

class FeatureExtractor(nn.Module):
    """
    Hybrid feature extractor combining:
    1. Multi-layer deep features from EfficientNet-B0
    2. Error Level Analysis (ELA) features — forensic technique
    3. Noise analysis features — captures compression/editing artifacts
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'efficientnet_b0':
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = base.features
            self.avgpool = base.avgpool
            # Tap intermediate layers for multi-scale representation
            self.tap_indices = [2, 4, 6, 8]
        elif model_name == 'resnet50':
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            base.fc = nn.Identity()
            self.backbone = base
            self.tap_indices = None
        else:
            raise ValueError(f"Model {model_name} not supported.")
            
        self.eval()
        self.to(self.device)

    def _extract_multilayer(self, x):
        """Extract features from multiple EfficientNet layers."""
        tapped = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i + 1 in self.tap_indices:
                pooled = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                pooled = pooled.flatten(1)
                tapped.append(pooled)
        return torch.cat(tapped, dim=1)

    def extract(self, tensor_image):
        """Extract deep features from patches."""
        tensor_image = tensor_image.to(self.device)
        with torch.no_grad():
            if self.model_name == 'efficientnet_b0':
                features = self._extract_multilayer(tensor_image)
            else:
                features = self.backbone(tensor_image)
            return features.cpu().numpy()


class ForensicFeatureExtractor:
    """
    Extracts forensic features from a PIL image:
    - Error Level Analysis (ELA): Detects regions that were edited/re-saved
    - Noise analysis: Captures compression artifact patterns
    - Local variance analysis: Detects inconsistent editing
    """
    
    @staticmethod
    def compute_ela(pil_image, quality=90):
        """
        Error Level Analysis: Re-compress image and measure the difference.
        Tampered regions show different error levels than original ones.
        Returns an ELA image as numpy array.
        """
        # Save to JPEG buffer at specified quality
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Re-load the re-compressed image
        resaved = Image.open(buffer)
        
        # Compute pixel-wise difference
        original_np = np.array(pil_image).astype(np.float32)
        resaved_np = np.array(resaved).astype(np.float32)
        
        # Amplify difference
        ela = np.abs(original_np - resaved_np) * (255.0 / (100 - quality + 1))
        ela = np.clip(ela, 0, 255).astype(np.uint8)
        
        return ela

    @staticmethod
    def compute_noise_features(pil_image):
        """
        Extract noise-level features using high-pass filtering.
        Different editing operations leave distinct noise patterns.
        """
        img_np = np.array(pil_image.convert('L'))  # Keep as uint8
        
        # Apply Laplacian (high-pass filter) to extract noise
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F).astype(np.float64)
        
        features = {
            'noise_mean': np.mean(np.abs(laplacian)),
            'noise_std': np.std(laplacian),
            'noise_kurtosis': float(np.mean((laplacian - np.mean(laplacian))**4) / 
                                     (np.std(laplacian)**4 + 1e-8)),
            'noise_skewness': float(np.mean((laplacian - np.mean(laplacian))**3) / 
                                      (np.std(laplacian)**3 + 1e-8)),
        }
        
        return features

    @staticmethod
    def extract_patch_forensics(pil_image, grid_size=(5, 5)):
        """
        Extract forensic features from each patch region of the image.
        Returns a feature vector per patch capturing local ELA and noise patterns.
        """
        img_w, img_h = pil_image.size
        n_rows, n_cols = grid_size
        patch_h = img_h // n_rows
        patch_w = img_w // n_cols
        
        # Compute ELA for the whole image once
        ela = ForensicFeatureExtractor.compute_ela(pil_image, quality=90)
        ela_gray = np.mean(ela, axis=2) if len(ela.shape) == 3 else ela
        
        # Compute noise map
        img_gray = np.array(pil_image.convert('L'))  # Keep as uint8
        laplacian = np.abs(cv2.Laplacian(img_gray, cv2.CV_64F).astype(np.float64))
        
        patch_features = []
        for i in range(n_rows):
            for j in range(n_cols):
                y1, y2 = i * patch_h, (i + 1) * patch_h
                x1, x2 = j * patch_w, (j + 1) * patch_w
                
                # ELA features for this patch
                ela_patch = ela_gray[y1:y2, x1:x2]
                ela_mean = np.mean(ela_patch)
                ela_std = np.std(ela_patch)
                ela_max = np.max(ela_patch)
                
                # Noise features for this patch
                noise_patch = laplacian[y1:y2, x1:x2]
                noise_mean = np.mean(noise_patch)
                noise_std = np.std(noise_patch)
                
                # Pixel statistics
                img_patch = img_gray[y1:y2, x1:x2]
                intensity_mean = np.mean(img_patch)
                intensity_std = np.std(img_patch)
                
                patch_features.append([
                    ela_mean, ela_std, ela_max,
                    noise_mean, noise_std,
                    intensity_mean, intensity_std,
                ])
        
        return np.array(patch_features, dtype=np.float32)

    @staticmethod
    def extract_image_forensics(pil_image):
        """
        Extract global forensic features for the image.
        Returns a single feature vector capturing overall image characteristics.
        """
        # ELA features at multiple quality levels
        ela_features = []
        for q in [70, 80, 90, 95]:
            ela = ForensicFeatureExtractor.compute_ela(pil_image, quality=q)
            ela_gray = np.mean(ela, axis=2) if len(ela.shape) == 3 else ela
            ela_features.extend([
                np.mean(ela_gray),
                np.std(ela_gray),
                np.max(ela_gray),
                np.percentile(ela_gray, 95),
                np.percentile(ela_gray, 99),
            ])
        
        # Noise features
        noise = ForensicFeatureExtractor.compute_noise_features(pil_image)
        
        # Patch consistency: variance across patches
        patch_forensics = ForensicFeatureExtractor.extract_patch_forensics(pil_image, (5, 5))
        # Standard deviation of ELA means across patches (high = inconsistent editing)
        ela_consistency = np.std(patch_forensics[:, 0])
        # Standard deviation of noise means across patches
        noise_consistency = np.std(patch_forensics[:, 3])
        # Range of ELA
        ela_range = np.max(patch_forensics[:, 0]) - np.min(patch_forensics[:, 0])
        noise_range = np.max(patch_forensics[:, 3]) - np.min(patch_forensics[:, 3])
        
        features = np.array(ela_features + [
            noise['noise_mean'],
            noise['noise_std'],
            noise['noise_kurtosis'],
            noise['noise_skewness'],
            ela_consistency,
            noise_consistency,
            ela_range,
            noise_range,
        ], dtype=np.float32)
        
        return features
