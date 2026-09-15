# 🛡️ Document Fraud Detection System

A production-grade, multi-tiered document fraud detection framework engineered to detect micro-edits, number alterations, forged stamps, and spliced text across medical bills, diagnostic lab reports, and financial invoices.

---

## ⚡ Flagship Engine: Fraud Engine v2 (Production Milestone)

The latest production system is located in **[`fraud_engine_v2/`](fraud_engine_v2/)**.

### 📊 Benchmark Performance (1,043 Unseen Test Pages)
* **100.00% Genuine Specificity (0 False Positives)**: Zero legitimate customer bills wrongly rejected.
* **98.85% Tampered Recall (516 / 522 Caught)**: Identifies single-digit alterations and localized text manipulations.
* **99.42% Balanced Accuracy**: Evaluated on an unseen 15% out-of-sample test split.
* **95.67% Pixel Localization Dice (F1)**: Pinpoints exact manipulated pixel regions.

```
                      PREDICTED AS TAMPERED    PREDICTED AS GENUINE
ACTUAL TAMPERED (522)          516 [TP]                  6 [FN]       ──> 98.85% Catch Rate
ACTUAL GENUINE  (521)            0 [FP]                521 [TN]       ──> 100.00% Clean Pass Rate
```

---

## 📁 Repository Overview

* **[`fraud_engine_v2/`](fraud_engine_v2/)**: **Current Production Engine (v2.0)** with multi-scale Test-Time Augmentation (TTA), dual-stream pixel segmentation (TruFor), automated prediction hierarchy export, and REST API.
* **[`FRAUD_ENGINE_METHODOLOGY.md`](fraud_engine_v2/FRAUD_ENGINE_METHODOLOGY.md)**: Comprehensive technical and forensic engineering methodology whitepaper.
* **[`EXECUTIVE_BENCHMARK_REPORT.md`](fraud_engine_v2/EXECUTIVE_BENCHMARK_REPORT.md)**: Executive summary for technical leadership and management.
* **[`bill_fraud_system/`](bill_fraud_system/)**: Legacy baseline v1 system.
* **[`index.html`](index.html)**: Live web report for GitHub Pages.

---

## 🚀 Quickstart (`fraud_engine_v2`)

```bash
cd fraud_engine_v2
pip install -r requirements.txt

# Run single document prediction
python predict.py --image path/to/document.jpg --output_dir outputs/

# Run full test benchmark evaluation with structured hierarchy export
python predict.py --evaluate_test --output_dir outputs/benchmark_results/

# Launch REST API Server (Port 8000)
python app.py
```

For full details, API documentation, and benchmark breakdown, see the **[fraud_engine_v2 README](fraud_engine_v2/README.md)**.
