# 🛡️ Fraud Engine v2

A state-of-the-art **Multi-Tiered Document & Image Fraud Detection Engine** designed to identify and localize micro-edits, altered numbers/text, splicing, and forensic inconsistencies in medical bills, diagnostic reports, and identity cards.

---

## ⚡ Key Highlights

- **Pixel-Level Localization (TruFor / Dual-Stream Segmentor)**: Generates high-resolution heatmaps ($0.0 \to 1.0$) and bounding boxes around manipulated characters and numbers.
- **Micro-Texture Forensics (Spatial Rich Model SRM)**: 30-filter noise residuals isolating scanner/sensor noise mismatches.
- **Multi-Scale Error Level Analysis (ELA)**: Recompression analysis across JPEG qualities 70, 80, 90, and 95.
- **Frequency Domain Double Compression (2D-DCT)**: Identifies periodic grid shifts and double JPEG quantization artifacts.
- **Typographic & Baseline Discrepancy Engine**: Detects baseline vertical jitter and inpainting background patch blocks.
- **Interactive Glassmorphism Dashboard**: Side-by-side original vs. heatmap inspection with bounding box evidence explanations.

---

## 📁 Directory Structure

```
fraud_engine_v2/
├── src/
│   ├── dataset_loader.py       # Scans & pairs 3,288+ image/mask pairs with train/val/test splits
│   ├── signal_forensics.py     # SRM noise, multi-Q ELA, and 2D-DCT grid analysis
│   ├── typography_checker.py   # Baseline jitter & inpainting patch detector
│   ├── tamper_segmentor.py     # Dual-stream TruFor segmentation network & bounding box extractor
│   ├── fraud_engine.py         # Fusion orchestrator & explainability generator
│   └── train_segmentor.py      # Training loop with Focal + Dice Loss
├── static/
│   ├── index.html              # Modern dark-mode UI
│   ├── style.css               # Glassmorphism design system
│   └── app.js                  # Frontend async handler & heatmap toggler
├── models/                     # Checkpoints & weights (e.g. tamper_segmentor.pth)
├── uploads/                    # Temporary storage for uploads & annotated artifacts
├── app.py                      # FastAPI server (REST API on port 8000)
└── requirements.txt            # Python dependencies
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd /Users/apple/Desktop/fraud_detection/fraud_engine_v2
pip install -r requirements.txt
```

### 2. Verify Dataset Splits
```bash
python -m src.dataset_loader --verify
```

### 3. Train the Segmentation Model
```bash
python -m src.train_segmentor --epochs 10 --batch_size 4
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser at: **`http://localhost:8000`**

---

## 🔌 REST API Endpoints

### `POST /api/analyze`
Accepts `multipart/form-data` with a `file` field (`JPG`, `PNG`, or `PDF`).

**Sample Response**:
```json
{
  "status": "TAMPERED",
  "confidence": 98.4,
  "composite_fraud_score": 0.842,
  "tamper_detected": true,
  "tamper_reason": "Localized text/pixel manipulation found (2 suspicious regions); Inconsistent compression artifacts (ELA)",
  "tamper_regions_count": 2,
  "tamper_boxes": [
    {
      "box": [320, 450, 410, 485],
      "confidence": 0.942,
      "area_px": 3150,
      "type": "Localized Text / Numerical Tampering",
      "reason": "Signal noise & pixel texture anomaly detected (Confidence: 94.2%)"
    }
  ],
  "overlay_url": "/uploads/annotated_sample.jpg",
  "processing_time": 0.32,
  "breakdown": {
    "pixel_segmentation_score": 0.942,
    "signal_forensics_score": 0.781,
    "typography_score": 0.700,
    "dct_periodicity": 0.520,
    "noise_disparity": 0.612,
    "ela_std": 14.32
  }
}
```
