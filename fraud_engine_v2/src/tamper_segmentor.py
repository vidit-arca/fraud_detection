import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

from .signal_forensics import SpatialRichModel

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class DocumentTamperSegmentor(nn.Module):
    """
    TruFor-inspired Dual-Stream Segmentation Network for Document Forgery Localization.
    Combines RGB visual features with SRM (Spatial Rich Model) noise residuals.
    Outputs:
      1. Pixel-level Tampering Probability Heatmap (H, W)
      2. Pixel Reliability Map (H, W)
      3. Global Document Integrity Classification Score (0.0 to 1.0)
    """
    def __init__(self, pretrained=True):
        super(DocumentTamperSegmentor, self).__init__()
        
        # Stream 1: RGB Backbone (EfficientNet-B0 feature extractor)
        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.rgb_stage1 = nn.Sequential(eff.features[0], eff.features[1]) # 16 channels, /2 (256x256)
        self.rgb_stage2 = eff.features[2]                                  # 24 channels, /4 (128x128)
        self.rgb_stage3 = eff.features[3]                                  # 40 channels, /8 (64x64)
        
        # Stream 2: Noise Residual Stream (SRM 3-channel input)
        self.noise_stem = ConvBlock(3, 32)                                # 32 channels, /1 (512x512)
        self.noise_down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 32)) # 32 channels, /2 (256x256)
        self.noise_down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 48)) # 48 channels, /4 (128x128)
        self.noise_down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(48, 80)) # 80 channels, /8 (64x64)
        
        # Cross-Modal Fusion at 1/8 scale (64x64)
        self.fusion = nn.Sequential(
            nn.Conv2d(40 + 80, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (U-Net style upsampling)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 64 channels, /4 (128x128)
        self.dec1 = ConvBlock(64 + 24 + 48, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 32 channels, /2 (256x256)
        self.dec2 = ConvBlock(32 + 16 + 32, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)    # 16 channels, /1 (512x512)
        self.dec3 = ConvBlock(16 + 32, 16)
        
        # Head 1: Pixel Tamper Probability Heatmap (Logits)
        self.tamper_head = nn.Conv2d(16, 1, kernel_size=1)
        
        # Head 2: Pixel Reliability Map
        self.reliability_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Head 3: Global Document Integrity Score
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_tensor, srm_tensor=None):
        """
        rgb_tensor: (B, 3, H, W) normalized image tensor
        srm_tensor: (B, 3, H, W) noise residual tensor (computed automatically if None)
        """
        B, C, H, W = rgb_tensor.shape
        
        if srm_tensor is None:
            # Generate SRM noise on device
            srm_tensor = self._compute_batch_srm(rgb_tensor)
            
        # Stream 1: RGB Encoder
        r1 = self.rgb_stage1(rgb_tensor)     # (B, 16, H/2, W/2)
        r2 = self.rgb_stage2(r1)             # (B, 24, H/4, W/4)
        r3 = self.rgb_stage3(r2)             # (B, 40, H/8, W/8)
        
        # Stream 2: Noise Residual Encoder
        n0 = self.noise_stem(srm_tensor)     # (B, 32, H, W)
        n1 = self.noise_down1(n0)            # (B, 32, H/2, W/2)
        n2 = self.noise_down2(n1)            # (B, 48, H/4, W/4)
        n3 = self.noise_down3(n2)            # (B, 80, H/8, W/8)
        
        # Fusion at 1/8 resolution (64x64)
        fused = self.fusion(torch.cat([r3, n3], dim=1)) # (B, 128, H/8, W/8)
        
        # Global Document Score
        glob_feat = self.global_pool(fused).flatten(1)
        global_tamper_score = self.global_head(glob_feat) # (B, 1)
        
        # Decoder Upsampling
        d1 = self.up1(fused)                                     # (B, 64, H/4, W/4)
        d1 = self.dec1(torch.cat([d1, r2, n2], dim=1))           # (B, 64, H/4, W/4)
        
        d2 = self.up2(d1)                                        # (B, 32, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, r1, n1], dim=1))           # (B, 32, H/2, W/2)
        
        d3 = self.up3(d2)                                        # (B, 16, H, W)
        d3 = self.dec3(torch.cat([d3, n0], dim=1))               # (B, 16, H, W)
        
        # Output Maps
        tamper_logits = self.tamper_head(d3)                     # (B, 1, H, W)
        tamper_map = torch.sigmoid(tamper_logits)                # (B, 1, H, W)
        reliability_map = self.reliability_head(d3)              # (B, 1, H, W)
        
        return {
            "tamper_map": tamper_map,
            "tamper_logits": tamper_logits,
            "reliability_map": reliability_map,
            "global_score": global_tamper_score
        }

    def _compute_batch_srm(self, rgb_tensor):
        """Helper to generate SRM noise residual on torch tensors."""
        # Convert RGB to grayscale
        gray = 0.299 * rgb_tensor[:, 0:1] + 0.587 * rgb_tensor[:, 1:2] + 0.114 * rgb_tensor[:, 2:3]
        
        # Build filter weight tensor from 3 SRM kernels
        kernels = torch.from_numpy(np.stack(SpatialRichModel.SRM_KERNELS, axis=0)).unsqueeze(1) # [3, 1, 5, 5]
        kernels = kernels.to(rgb_tensor.device, dtype=rgb_tensor.dtype)
        
        srm = F.conv2d(gray, kernels, padding=2)
        srm = torch.clamp(srm, -3.0, 3.0)
        return srm


class TamperLocalizationHelper:
    """
    Post-processing tools to convert raw pixel probability heatmaps into
    actionable bounding boxes and visual heatmap overlays.
    """
    @staticmethod
    def extract_tamper_bounding_boxes(tamper_map_np, min_area=30, threshold=0.50, orig_size=None):
        """
        Converts a 2D tampering heatmap (0.0 to 1.0) into bounding boxes.
        orig_size: (orig_w, orig_h) to scale bounding boxes back to original document resolution.
        """
        h_map, w_map = tamper_map_np.shape
        binary_mask = (tamper_map_np > threshold).astype(np.uint8) * 255
        
        # Morphological closing to merge closely spaced character fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scale_x = (orig_size[0] / float(w_map)) if orig_size else 1.0
        scale_y = (orig_size[1] / float(h_map)) if orig_size else 1.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Compute mean confidence inside this component
            comp_mask = np.zeros_like(tamper_map_np, dtype=np.uint8)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            mean_conf = float(np.mean(tamper_map_np[comp_mask > 0]))
            
            orig_x1 = int(x * scale_x)
            orig_y1 = int(y * scale_y)
            orig_x2 = int((x + bw) * scale_x)
            orig_y2 = int((y + bh) * scale_y)
            
            boxes.append({
                "box": [orig_x1, orig_y1, orig_x2, orig_y2],
                "confidence": round(mean_conf, 3),
                "area_px": int(area * scale_x * scale_y),
                "type": "Localized Text / Numerical Tampering",
                "reason": f"Signal noise & pixel texture anomaly detected (Confidence: {mean_conf*100:.1f}%)"
            })
            
        # Sort by confidence descending
        boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        return boxes

    @staticmethod
    def render_visual_overlay(pil_image, tamper_map_np, boxes):
        """
        Creates an annotated visual result overlaying color-coded heatmaps and bounding boxes.
        Returns a PIL Image.
        """
        img_np = np.array(pil_image.convert("RGB"))
        h_orig, w_orig = img_np.shape[:2]
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(tamper_map_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply JET / TURBO colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend heatmap onto image where tampering is suspected (> 0.2)
        alpha = np.clip((heatmap_resized - 0.15) * 1.5, 0.0, 0.65)[:, :, np.newaxis]
        blended = (1.0 - alpha) * img_np.astype(np.float32) + alpha * heatmap_color.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Draw bounding boxes
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            conf = item["confidence"]
            # Red box with label
            cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 30, 30), 2)
            label = f"ALERT {int(conf*100)}%"
            cv2.putText(blended, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 30, 30), 2)
            
        return Image.fromarray(blended)
