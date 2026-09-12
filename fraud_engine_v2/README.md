# 🛡️ Fraud Engine v2 — Production Document Fraud Detection Engine

A state-of-the-art **Multi-Tiered Document & Image Fraud Detection Engine** designed to identify and localize micro-edits, altered numbers/text, splicing, and forensic inconsistencies in medical bills, diagnostic lab reports, and claim invoices.

Achieves **99.42% Balanced Accuracy** and **100.00% Genuine Specificity (0 False Positives)** on out-of-sample unseen benchmark testing (1,043 pages).

---

## ⚡ Key Highlights & Benchmark Performance

- **100.00% Genuine Specificity (0 False Positives)**: Zero legitimate customer bills wrongly rejected or delayed.
- **98.85% Tampered Recall (516 / 522 Caught)**: Identifies single-digit manipulations, copied stamps, and inpainting.
- **95.67% Pixel Localization (Dice F1)**: Pinpoints exact characters, numbers, and dates altered.
- **Multi-Scale Test-Time Augmentation (TTA)**: 4-pass inference across scales (448px, 512px, 576px + horizontal flip) filters scanner and compression artifacts.
- **Multi-Signal Forensics**: Integrates Spatial Rich Model (SRM) 30-filter noise residuals, multi-Q ELA, 2D-DCT double-compression analysis, and typography inspection.
- **Automated Prediction Hierarchy & Audit Suite**: Generates structured Ground Truth vs. Prediction folder hierarchies, multi-panel side-by-side comparison images, tabular CSV classification reports, and interactive HTML dashboards.

---

## 📊 Benchmark Scorecard (1,043 Unseen Test Pages)

```
                      PREDICTED AS TAMPERED    PREDICTED AS GENUINE
ACTUAL TAMPERED (522)          516 [TP]                  6 [FN]       ──> 98.85% Catch Rate
ACTUAL GENUINE  (521)            0 [FP]                521 [TN]       ──> 100.00% Clean Pass Rate
```

| Metric | Score | Industry Benchmark |
|---|:---:|:---:|
| **Balanced Accuracy** | **99.42%** | 80% – 88% |
| **Document Precision** | **100.00%** | 75% – 85% |
| **Tampered Recall (Sensitivity)** | **98.85%** | 85% – 92% |
| **Genuine Specificity** | **100.00%** | 80% – 90% |
| **Document F1-Score** | **99.42%** | 82% – 88% |
| **Pixel Localization (Dice)** | **95.67%** | 65% – 78% |
| **Pixel IoU (Jaccard)** | **91.69%** | 55% – 70% |

---

## 📁 Repository Structure

```
fraud_engine_v2/
├── src/
│   ├── fraud_engine.py          # Multi-scale TTA fusion orchestrator & decision gate
│   ├── tamper_segmentor.py      # Dual-stream TruFor U-Net segmentor & box extractor
│   ├── dataset_loader.py        # Balanced dataset pipeline (pairs tampered + genuine negatives)
│   ├── train_segmentor.py       # Training loop with Focal + Dice Loss
│   ├── signal_forensics.py      # SRM noise residuals, multi-Q ELA & 2D-DCT grid analysis
│   ├── typography_checker.py    # Baseline vertical jitter & font discrepancy engine
│   ├── prediction_organizer.py  # Hierarchy builder, multi-panel visual comparisons & CSV/HTML
│   └── predict.py               # Core prediction and evaluation logic
├── static/                      # Modern dark-mode web application UI
├── predict.py                   # CLI prediction and benchmark entrypoint
├── app.py                       # FastAPI production REST server (Port 8000)
├── requirements.txt             # Project dependencies
├── EXECUTIVE_BENCHMARK_REPORT.md # Executive whitepaper for leadership / C-suite
└── EXECUTIVE_BENCHMARK_REPORT.html # Print-ready / PDF executive report
```

---

## 🚀 Quickstart & Installation

### 1. Install Dependencies
```bash
cd fraud_engine_v2
pip install -r requirements.txt
```

### 2. Run Single Document Prediction (CLI)
```bash
python predict.py --image path/to/document.jpg --output_dir outputs/
```

### 3. Run Batch Prediction on a Folder
```bash
python predict.py --dir path/to/folder/ --output_dir outputs/batch_results/
```

### 4. Run Complete Benchmark Evaluation with Organized Visual Hierarchy
```bash
python predict.py \
    --evaluate_test \
    --dataset_dir path/to/dataset/ \
    --model_path models/tamper_segmentor.pth \
    --output_dir outputs/benchmark_results/ \
    --log_file outputs/benchmark.log
```

---

## 🗂️ Output Hierarchy & Visual Audit Suite

When running evaluation with `--output_dir outputs/benchmark_results`, the engine automatically organizes all predictions into an auditable structure:

```
outputs/benchmark_results/
├── index.html                           # 🌐 Interactive browser gallery (Filter by TP, FP, FN, TN)
├── summary.json                         # 📑 Metrics JSON (Accuracy, Recall, Specificity, Dice, Confusion Matrix)
├── classification_report.csv            # 📊 Full CSV with scores, GT vs. Prediction & asset paths
│
├── by_outcome/                          # 📂 Sorted by Diagnostic Outcome
│   ├── true_positives_TP/<doc_id>/      # Caught Forgeries (01_orig, 02_input, 03_gt_mask, 04_overlay, 05_side_by_side)
│   ├── false_negatives_FN/<doc_id>/     # Missed Forgeries (for targeted review)
│   ├── true_negatives_TN/<doc_id>/      # Verified Authentic documents
│   └── false_positives_FP/<doc_id>/     # False alarms
│
└── galleries/                           # 🖼️ Flat quick-scroll galleries of 4-panel comparison images
    ├── true_positives_TP/
    ├── false_positives_FP/
    └── false_negatives_FN/
```

### 🖼️ Multi-Panel Comparison Banner (`05_side_by_side.jpg`)
For every document tested, the engine produces a 4-panel annotated visual comparison banner:
* **Panel 1**: Original Authentic Document (Reference)
* **Panel 2**: Input Document
* **Panel 3**: Ground Truth Forgery Mask (Green highlight)
* **Panel 4**: Predicted Fraud Engine Heatmap & Bounding Boxes

---

## 🔌 REST API Endpoints

### Launch API Server
```bash
python app.py
# API runs on http://0.0.0.0:8000
```

### `POST /api/analyze`
Accepts `multipart/form-data` with a `file` field (`JPG`, `PNG`, or `PDF`).

**Sample Response:**
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
  "processing_time": 0.22,
  "breakdown": {
    "pixel_segmentation_score": 0.942,
    "tamper_area_ratio": 0.0084,
    "tamper_peak_score": 1.0000,
    "signal_forensics_score": 0.781,
    "typography_score": 0.700,
    "dct_periodicity": 0.520,
    "noise_disparity": 0.612,
    "ela_std": 14.32
  }
}
```

---

## 📑 Executive Report
For leadership review, see:
* **Markdown**: [EXECUTIVE_BENCHMARK_REPORT.md](EXECUTIVE_BENCHMARK_REPORT.md)
* **Print-Ready / PDF**: [EXECUTIVE_BENCHMARK_REPORT.html](EXECUTIVE_BENCHMARK_REPORT.html)
