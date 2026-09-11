import os
import time
import torch
import numpy as np
from PIL import Image
import cv2

from .tamper_segmentor import DocumentTamperSegmentor, TamperLocalizationHelper
from .signal_forensics import SignalForensicsEngine
from .typography_checker import TypographyInspector

class DocumentFraudEngine:
    """
    Production-grade Multi-Tiered Document Fraud Detection Engine.
    Coordinates Deep Pixel Segmentation, Forensic Signal Analysis, and Typography Inspection.
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = DocumentTamperSegmentor(pretrained=True).to(self.device)
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"✅ Fraud Engine: Segmentor weights loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Fraud Engine: Could not load weights ({e}). Running in pre-trained feature mode.")
        else:
            print("ℹ️ Fraud Engine: Initialized with pre-trained visual/SRM backbone.")
            
        self.model.eval()

    def analyze_document(self, image_input, ocr_json=None, output_dir=None):
        """
        Runs comprehensive multi-layer fraud analysis on an input document image.
        
        image_input: PIL Image or file path string
        ocr_json: Optional OCR JSON data or file path
        output_dir: Optional directory to save generated heatmaps and overlay artifacts
        """
        start_time = time.time()
        
        # 1. Load and prepare image
        if isinstance(image_input, str):
            pil_image = Image.open(image_input).convert("RGB")
            filename = os.path.basename(image_input)
        else:
            pil_image = image_input.convert("RGB")
            filename = f"upload_{int(time.time())}.jpg"
            
        orig_w, orig_h = pil_image.size
        
        # 2. Layer 1: Pixel-Level Tamper Segmentation (TruFor) with TTA
        # Runs 4 forward passes (3 scales + 1 flip) and averages the tamper maps.
        # Scanner artifacts are pixel-location-specific and wash out in the mean.
        # Real tamper regions (structural content edits) persist across all scales.
        tamper_map, global_model_score = self._run_segmentation_tta(pil_image)
        seg_boxes = TamperLocalizationHelper.extract_tamper_bounding_boxes(
            tamper_map, min_area=50, threshold=0.48, orig_size=(orig_w, orig_h)
        )
        
        # Compute tamper map statistics — these are highly discriminative
        # (pixel segmentation head has 88.98% F1; global_score head is not discriminative)
        tamper_area_ratio = float(np.mean(tamper_map > 0.48))   # fraction of pixels above threshold
        tamper_peak_score = float(np.max(tamper_map))            # peak pixel-level confidence
        
        # 3. Layer 2: Forensic Signal Analysis (ELA, SRM Noise, DCT)
        signal_results = SignalForensicsEngine.analyze_document_signals(pil_image)
        signal_score = signal_results["signal_tamper_score"]
        
        # 4. Layer 3: Typographic & Baseline Discrepancies
        typo_results = TypographyInspector.inspect_typography(pil_image, ocr_json)
        typo_score = typo_results["typography_score"]
        typo_boxes = typo_results["findings"]
        
        # Filter for statistically significant anomalies
        # Confidence threshold raised to 0.60 (from 0.50) to compensate for lower min_area.
        significant_seg_boxes = [b for b in seg_boxes if b["confidence"] >= 0.60]
        significant_typo_boxes = [b for b in typo_boxes if b.get("confidence", 0) >= 0.80]
        
        # 5. Fusion & Non-Maximum Suppression (NMS) on bounding boxes
        all_candidate_boxes = significant_seg_boxes + significant_typo_boxes
        merged_boxes = self._merge_and_deduplicate_boxes(all_candidate_boxes)
        
        # 6. Tamper-Map-Driven Decision Gate
        # global_model_score is NOT used: it outputs >= 0.45 for ALL docs (not discriminative).
        # The pixel segmentation head (tamper_map) IS discriminative — use it directly.
        
        ela_std    = signal_results["ela_metrics"]["q90_std"]
        dct_period = signal_results["dct_double_compression_score"]
        noise_disp = signal_results["noise_disparity"]
        
        # --- Tier 1 (Primary): Tamper map localization ---
        # IMPORTANT: Only SEGMENTATION boxes count for the decision.
        # Typo boxes alone cannot trigger is_tampered — typography inspector produces
        # false positives on structured docs (lab reports, tables, grids).
        seg_box_triggered  = len(significant_seg_boxes) > 0
        area_triggered     = tamper_area_ratio > 0.008    # >0.8% of pixels flagged
        # Peak threshold at 0.9995 cleanly splits two populations observed empirically:
        #   Genuine scanner artifacts (ink dots, stamp edges): peak capped at ~0.9991
        #   Actual tampered pixels (text edits, inpainting):   peak always hits 1.0000
        # No area requirement needed — the peak value alone is the discriminator.
        peak_triggered     = tamper_peak_score > 0.9995
        
        seg_triggered = seg_box_triggered or area_triggered or peak_triggered
        
        # Signal forensics: supporting evidence ONLY — not a standalone trigger.
        signal_high = (noise_disp > 0.80) and (ela_std > 25.0) and (dct_period > 0.85)
        
        is_tampered = seg_triggered  # model decision is authoritative
        
        if is_tampered:
            max_patch_conf = max([b["confidence"] for b in merged_boxes], default=float(global_model_score))
            # Segmentation drives 65% of confidence; signal and typo are supporting evidence
            composite_score = float(np.clip(
                0.65 * max_patch_conf +
                0.25 * signal_score +
                0.10 * typo_score,
                0.40, 1.0
            ))
            confidence = min(99.9, 70.0 + 29.9 * composite_score)
        else:
            # Clean document: zero localized tampering detected
            max_patch_conf = 0.0
            composite_score = float(np.clip(
                0.20 * signal_score +
                0.10 * typo_score,
                0.0, 0.35
            ))
            confidence = min(99.9, max(85.0, 99.5 - (composite_score * 30.0)))

        # Formulate human-readable evidence summary
        reasons = []
        if significant_seg_boxes:
            reasons.append(f"Localized text/pixel manipulation found ({len(significant_seg_boxes)} suspicious regions)")
        elif area_triggered:
            reasons.append(f"Diffuse pixel-level manipulation detected (tamper area: {tamper_area_ratio*100:.2f}%)")
        elif peak_triggered:
            reasons.append(f"High-confidence tamper pixel detected (peak: {tamper_peak_score:.3f})")
        if significant_typo_boxes:
            reasons.append(f"Typographical anomalies ({len(significant_typo_boxes)} regions, supporting evidence only)")
        if is_tampered and signal_high:
            reasons.append("Multi-signal forensic anomaly also detected (ELA + DCT + Noise)")
        elif is_tampered and dct_period > 0.60:
            reasons.append("Double JPEG compression & grid shifts detected")
        if is_tampered and ela_std > 12.0:
            reasons.append("Inconsistent compression artifacts (ELA)")
            
        tamper_reason = "; ".join(reasons) if is_tampered else "Document verified authentic. No localized tampering or typographic manipulations detected."
        
        # 7. Render Visual Artifacts
        overlay_image = TamperLocalizationHelper.render_visual_overlay(pil_image, tamper_map, merged_boxes)
        
        overlay_url = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            overlay_filename = f"annotated_{filename.rsplit('.', 1)[0]}.jpg"
            overlay_path = os.path.join(output_dir, overlay_filename)
            overlay_image.save(overlay_path, "JPEG", quality=90)
            overlay_url = f"/uploads/{overlay_filename}"
            
        elapsed = time.time() - start_time
        
        return {
            "status": "TAMPERED" if is_tampered else "GENUINE",
            "confidence": round(confidence, 1),
            "composite_fraud_score": round(composite_score, 4),
            "tamper_detected": is_tampered,
            "tamper_reason": tamper_reason,
            "tamper_regions_count": len(merged_boxes),
            "tamper_boxes": merged_boxes,
            "overlay_url": overlay_url,
            "processing_time": round(elapsed, 2),
            "breakdown": {
                "pixel_segmentation_score": round(float(max_patch_conf), 4),
                "tamper_area_ratio": round(tamper_area_ratio, 6),
                "tamper_peak_score": round(tamper_peak_score, 4),
                "signal_forensics_score": round(float(signal_score), 4),
                "typography_score": round(float(typo_score), 4),
                "dct_periodicity": round(float(dct_period), 4),
                "noise_disparity": round(float(noise_disp), 4),
                "ela_std": round(float(ela_std), 4)
            },
            "document_info": {
                "width": orig_w,
                "height": orig_h,
                "format": pil_image.format or "JPEG"
            }
        }

    def _run_segmentation_tta(self, pil_image):
        """
        Test-Time Augmentation: averages tamper maps across 3 scales + horizontal flip.
        
        Scanner artifacts are pixel-location-specific (JPEG block boundaries, ink dots)
        and produce inconsistent responses across scales — they wash out in the mean.
        Real tamper regions are structural content changes that persist at all scales.
        
        ~4x slower than single-pass inference but eliminates isolated-pixel FPs.
        """
        maps = []
        global_scores = []
        canonical = 512
        
        # Scale augmentations
        for scale in (448, 512, 576):
            t_map, g_score = self._run_segmentation_at_size(pil_image, scale)
            if scale != canonical:
                t_map = cv2.resize(t_map, (canonical, canonical), interpolation=cv2.INTER_LINEAR)
            maps.append(t_map)
            global_scores.append(g_score)
        
        # Horizontal flip augmentation (flip back after inference)
        flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        t_map_flip, g_score_flip = self._run_segmentation_at_size(flipped, canonical)
        maps.append(np.fliplr(t_map_flip))
        global_scores.append(g_score_flip)
        
        averaged_map = np.mean(maps, axis=0)
        return averaged_map, float(np.mean(global_scores))

    def _run_segmentation_at_size(self, pil_image, size):
        """Single forward pass at a given square resolution."""
        resized = pil_image.resize((size, size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_arr = (arr - mean) / std
        
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            tamper_map  = outputs["tamper_map"].squeeze().cpu().numpy()
            global_score = float(outputs["global_score"].squeeze().cpu().item())
            
        return tamper_map, global_score

    def _merge_and_deduplicate_boxes(self, boxes, iou_thresh=0.35):
        """Merges overlapping bounding boxes using Non-Maximum Suppression (NMS)."""
        if not boxes:
            return []
            
        sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.5), reverse=True)
        keep = []
        
        for b in sorted_boxes:
            box_a = b["box"]
            overlap = False
            for k in keep:
                box_b = k["box"]
                # Compute IoU
                x1 = max(box_a[0], box_b[0])
                y1 = max(box_a[1], box_b[1])
                x2 = min(box_a[2], box_b[2])
                y2 = min(box_a[3], box_b[3])
                
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                union_area = area_a + area_b - inter_area
                
                iou = inter_area / float(union_area + 1e-6)
                if iou > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                keep.append(b)
                
        return keep
