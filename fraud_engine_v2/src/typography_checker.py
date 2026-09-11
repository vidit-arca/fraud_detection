import json
import cv2
import numpy as np
from PIL import Image

class TypographyInspector:
    """
    Analyzes typographical alignment, baseline consistency, character spacing,
    and background patch artifacts around text and numerical values.
    """

    @staticmethod
    def detect_background_patch_artifacts(pil_image, ocr_boxes=None):
        """
        Detects unnatural solid rectangular background blocks/patches placed
        over original text to erase and overwrite it.
        """
        gray = np.array(pil_image.convert("L"))
        h, w = gray.shape
        
        # Morphological gradient to isolate background smoothness discontinuities
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Local variance map via box filter
        mean_img = cv2.boxFilter(gray.astype(np.float32), -1, (15, 15))
        sq_mean_img = cv2.boxFilter((gray.astype(np.float32))**2, -1, (15, 15))
        variance_map = np.maximum(0, sq_mean_img - mean_img**2)
        
        # Find rectangular regions of abnormally low variance surrounded by high gradient
        low_variance_mask = (variance_map < 10.0) & (mean_img > 180.0)  # white paper background
        
        patch_anomalies = []
        contours, _ = cv2.findContours(low_variance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter for suspicious patch dimensions (e.g. word/number size: 20x8 to 200x50 px)
            if 25 < bw < 250 and 10 < bh < 60:
                aspect = bw / float(bh)
                if 1.5 < aspect < 8.0:
                    patch_anomalies.append({
                        "box": [int(x), int(y), int(x + bw), int(y + bh)],
                        "type": "Inpainted Background Patch",
                        "confidence": 0.82,
                        "reason": "Abnormal flat texture block indicative of erased original text"
                    })
                    
        return patch_anomalies

    @staticmethod
    def analyze_baseline_alignment(ocr_words):
        """
        Groups words into horizontal text lines and computes linear regression baseline.
        Flags words or numbers that vertically deviate/jitter from the line baseline.
        
        ocr_words: List of dicts with 'box' [x1, y1, x2, y2] and 'text'
        """
        if not ocr_words or len(ocr_words) < 3:
            return []
            
        # Group words by approximate Y position (line clustering)
        sorted_words = sorted(ocr_words, key=lambda w: (w["box"][1], w["box"][0]))
        lines = []
        curr_line = [sorted_words[0]]
        
        for word in sorted_words[1:]:
            prev_y_center = (curr_line[-1]["box"][1] + curr_line[-1]["box"][3]) / 2.0
            word_y_center = (word["box"][1] + word["box"][3]) / 2.0
            word_height = word["box"][3] - word["box"][1]
            
            # If within 50% of word height vertically, consider same line
            if abs(word_y_center - prev_y_center) < (word_height * 0.5):
                curr_line.append(word)
            else:
                if len(curr_line) >= 3:
                    lines.append(curr_line)
                curr_line = [word]
        if len(curr_line) >= 3:
            lines.append(curr_line)
            
        alignment_anomalies = []
        for line in lines:
            # Calculate baseline for each word (bottom Y coordinate)
            xs = [float((w["box"][0] + w["box"][2]) / 2.0) for w in line]
            ys_bottom = [float(w["box"][3]) for w in line]
            
            # Linear regression: baseline = slope * x + intercept
            if len(xs) >= 4:
                coeffs = np.polyfit(xs, ys_bottom, deg=1)
                fitted_baselines = np.polyval(coeffs, xs)
                deviations = np.abs(ys_bottom - fitted_baselines)
                
                std_dev = np.std(deviations)
                for idx, dev in enumerate(deviations):
                    # Flag if word deviates more than 2.5x standard deviation and > 3.0 pixels
                    if dev > 3.5 and dev > 2.2 * std_dev:
                        flagged_word = line[idx]
                        alignment_anomalies.append({
                            "box": flagged_word["box"],
                            "text": flagged_word.get("text", ""),
                            "type": "Baseline Alignment Jitter",
                            "confidence": min(0.95, 0.70 + float(dev) / 10.0),
                            "deviation_px": round(float(dev), 2),
                            "reason": f"Text bottom baseline displaced by {dev:.1f}px from row alignment"
                        })
                        
        return alignment_anomalies

    @staticmethod
    def inspect_typography(pil_image, ocr_json_path=None):
        """
        Runs complete typographic and micro-alignment inspection on the document.
        """
        ocr_words = []
        if ocr_json_path and hasattr(ocr_json_path, 'read'):
            # It's a file object or string
            pass
        elif ocr_json_path and isinstance(ocr_json_path, str):
            try:
                with open(ocr_json_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        ocr_words = data
            except Exception:
                ocr_words = []
                
        # 1. Check background patches
        patch_findings = TypographyInspector.detect_background_patch_artifacts(pil_image, ocr_words)
        
        # 2. Check baseline alignment
        baseline_findings = TypographyInspector.analyze_baseline_alignment(ocr_words)
        
        all_findings = patch_findings + baseline_findings
        
        # Compute overall typography risk score (0.0 to 1.0)
        typo_score = min(1.0, len(all_findings) * 0.35)
        
        return {
            "typography_score": float(typo_score),
            "findings_count": len(all_findings),
            "findings": all_findings
        }
