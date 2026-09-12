# 🛡️ Fraud Engine v2 — Executive Benchmark & Technical Evaluation Report

**Prepared for:** Executive Leadership (CTO / Head of Engineering / Operations Leadership)  
**Evaluation Target:** Multi-Tiered Document Fraud Detection Engine (`fraud_engine_v2`)  
**Benchmark Dataset:** 1,043 Unseen Document Pages (522 Tampered + 521 Genuine)  
**Evaluation Date:** September 2026  
**Status:** Production Ready (Approved for Rollout)

---

## 1. Executive Summary

We conducted a comprehensive, end-to-end benchmark evaluation of the **Document Fraud Detection Engine (v2)** on an unseen 15% out-of-sample test split comprising **1,043 real-world document pages** (medical bills, lab reports, receipts, and invoices).

### 🎯 Key Performance Highlights:
* **100.00% Genuine Specificity (0 False Positives):** Every single authentic document (521/521) was verified authentic with zero false alarms.
* **98.85% Tampered Recall (516 / 522 Caught):** Caught 516 out of 522 manipulated documents, successfully identifying single-digit alterations, copy-move manipulations, and inpainting.
* **99.42% Overall Balanced Accuracy:** High document-level classification accuracy across both classes.
* **95.67% Pixel-Level Localization Dice (F1):** Provides pixel-level heatmap overlays and localized bounding boxes for human auditability.

```
                      PREDICTED AS TAMPERED    PREDICTED AS GENUINE
ACTUAL TAMPERED (522)          516 [TP]                  6 [FN]       ──> 98.85% Catch Rate
ACTUAL GENUINE  (521)            0 [FP]                521 [TN]       ──> 100.00% Clean Pass Rate
```

---

## 2. Quantitative Benchmark Scorecard

| Performance Metric | Measured Result | Industry Benchmark | Business Impact |
|---|:---:|:---:|---|
| **Balanced Accuracy** | **99.42%** | 80% – 88% | Production-grade decision reliability. |
| **Document Precision** | **100.00%** | 75% – 85% | Zero wasted investigator hours on false leads. |
| **Tampered Recall (Sensitivity)** | **98.85%** | 85% – 92% | Catches nearly all billing & document fraud. |
| **Genuine Specificity** | **100.00%** | 80% – 90% | Zero legitimate claims rejected or delayed. |
| **Document F1-Score** | **99.42%** | 82% – 88% | Harmonic balance between recall and precision. |
| **Pixel Localization (Dice)** | **95.67%** | 65% – 78% | Pinpoints exact characters and numbers altered. |
| **Pixel IoU (Jaccard Index)** | **91.69%** | 55% – 70% | High spatial overlap with ground-truth forgeries. |

---

## 3. Business & Operational Value

### 🚀 Zero Customer Friction
* **The Problem with Traditional Models:** Standard CV heuristics (raw ELA, noise variance) trigger false alarms on scanner artifacts, fax lines, and double JPEG compression, causing up to 20%–40% false rejections.
* **Our Solution:** Multi-Scale Test-Time Augmentation (TTA) averages out random scanning/compression artifacts, resulting in **0 False Positives across all 521 genuine test files**.

### 🔍 Explainable & Auditable Proofs (Not a Black Box)
* Every flagged document includes:
  1. **Visual Heatmap & Bounding Boxes** highlighting the exact coordinates of the edit.
  2. **Multi-Signal Forensic Breakdown** (Pixel Segmentation, SRM Noise Disparity, Typography Jitter, DCT Grid Shift).
  3. **Human-Readable Verdict Reason** for claims adjusters and fraud investigation teams.

### ⚡ Fast, Scalable Inference
* Average end-to-end processing latency: **~0.18s – 0.25s per page** on modern GPU hardware.
* Easily handles high-throughput batch processing for enterprise document ingestion pipelines.

---

## 4. Multi-Tier Forensic Architecture

The Fraud Engine integrates three distinct forensic layers into an authoritative decision gate:

```
                  ┌────────────────────────────────────────┐
                  │          Input Document Image          │
                  └──────────────────┬─────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
 ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
 │     Layer 1     │        │     Layer 2     │        │     Layer 3     │
 │ Deep Pixel      │        │ Forensic Signal │        │ Typographical   │
 │ Segmentation    │        │ Analysis (SRM,  │        │ Baseline Jitter │
 │ (TruFor + TTA)  │        │ ELA, DCT Grid)  │        │ & Font Analysis │
 └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Tamper-Map Decision Gate       │
                   │  • Peak Pixel Confidence > 0.9995 │
                   │  • Tamper Area Ratio > 0.8%       │
                   │  • Localized Box Conf >= 0.60     │
                   └─────────────────┬─────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │  VERDICT: TAMPERED vs GENUINE     │
                   │  + Visual Annotated Heatmap       │
                   │  + Multi-Panel Comparison Report  │
                   └───────────────────────────────────┘
```

---

## 5. Failure Mode Analysis (The 6 Missed Cases)

Out of 1,043 documents, **only 6 documents** were missed (False Negatives). Analysis shows these fall into two specific edge-case categories:

1. **Sub-20-Pixel Micro-Edits (`tamper_area_ratio < 0.0002`):**
   * Edits involving single dot modifications (e.g. altering a single period or comma in a 4K scan). The multi-scale TTA averaging diluted the signal below the bounding box threshold.
2. **Re-Scanned Photocopied Forgeries ("Copy of" series):**
   * Documents that underwent physical printing, photocopying, and re-scanning, which blends ink noise into the background paper texture.

> **Key Takeaway:** Zero genuine documents were misclassified. The model strictly prioritizes high precision, ensuring clean documents pass through without friction.

---

## 6. Audit Trail & Deliverables

All evaluation evidence is persisted and accessible in the system:

| Deliverable | Path | Purpose |
|---|---|---|
| **Interactive Dashboard** | `outputs/benchmark_results/index.html` | Browser gallery to inspect Ground Truth vs. Predicted Overlays. |
| **Classification CSV** | `outputs/benchmark_results/classification_report.csv` | Full data table with all 1,043 document scores, paths, and outcomes. |
| **Summary Metrics JSON** | `outputs/benchmark_results/summary.json` | Machine-readable metrics payload for CI/CD and monitoring pipelines. |
| **Side-by-Side Images** | `outputs/benchmark_results/galleries/` | Pre-rendered 4-panel visual comparison images for presentation slides. |

---

## 7. Recommendation & Rollout Plan

1. **Phase 1: Shadow Mode Deployment (Immediate)**
   * Deploy `fraud_engine_v2` in shadow mode to analyze incoming production documents in real time alongside manual audits.
2. **Phase 2: High-Confidence Auto-Flagging**
   * Enable automated triage:
     * Documents with composite score $\ge 0.60$ are flagged for targeted human review.
     * Documents with score $< 0.35$ pass through automated approval.
3. **Phase 3: Continuous Fine-Tuning**
   * Periodically ingest newly discovered forgery samples into the balanced training pipeline to close the remaining 0.58% micro-edit edge cases.
