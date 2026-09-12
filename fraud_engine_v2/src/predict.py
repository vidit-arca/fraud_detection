import os
import sys
import glob
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Ensure project root and src are in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.fraud_engine import DocumentFraudEngine
from src.dataset_loader import create_or_load_splits, DocumentTamperDataset, DEFAULT_DATASET_DIR
from src.train_segmentor import evaluate, FocalDiceLoss
from src.prediction_organizer import PredictionHierarchyOrganizer


def format_box(box):
    return f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"


def convert_pdf_to_image_safe(pdf_path):
    """Converts first page of PDF to JPG using sips (macOS) or pdf2image."""
    jpg_path = pdf_path.rsplit(".", 1)[0] + ".jpg"
    try:
        import subprocess
        res = subprocess.run(
            ["sips", "-s", "format", "jpeg", pdf_path, "--out", jpg_path],
            capture_output=True, text=True, timeout=15
        )
        if res.returncode == 0 and os.path.exists(jpg_path):
            return jpg_path
    except Exception:
        pass
        
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
        if images:
            images[0].save(jpg_path, "JPEG", quality=95)
            return jpg_path
    except Exception:
        pass
        
    return None


def predict_single(engine, file_path, output_dir="outputs", ocr_json=None):
    """
    Runs fraud detection on a single document image or PDF.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at '{file_path}'")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"📄 ANALYZING DOCUMENT: {os.path.basename(file_path)}")
    print("=" * 70)
    
    analysis_path = file_path
    if file_path.lower().endswith(".pdf"):
        converted = convert_pdf_to_image_safe(file_path)
        if converted:
            analysis_path = converted
        else:
            print("❌ Failed to convert PDF to image.")
            return None

    res = engine.analyze_document(analysis_path, ocr_json=ocr_json, output_dir=output_dir)
    
    # Output Summary
    status_icon = "⚠️" if res["status"] == "TAMPERED" else "✅"
    print(f"Verdict        : {status_icon} {res['status']}")
    print(f"Confidence     : {res['confidence']}%")
    print(f"Fraud Score    : {res['composite_fraud_score']:.4f} / 1.0000")
    print(f"Regions Found  : {res['tamper_regions_count']} suspicious areas")
    print(f"Evidence       : {res['tamper_reason']}")
    print(f"Processing Time: {res['processing_time']}s")
    
    if res["tamper_boxes"]:
        print("\n🔍 Localized Tamper Bounding Boxes:")
        print(f"  {'#':<3} | {'Coordinates [x1,y1,x2,y2]':<26} | {'Confidence':<10} | {'Type & Details'}")
        print("  " + "-" * 66)
        for i, b in enumerate(res["tamper_boxes"], 1):
            coords = format_box(b["box"])
            conf = f"{b['confidence']*100:.1f}%"
            desc = b.get("reason", b.get("type", ""))
            print(f"  {i:<3} | {coords:<26} | {conf:<10} | {desc}")
            
    print("\n📊 Multi-Layer Forensic Breakdown:")
    b = res["breakdown"]
    print(f"  • Pixel Segmentation (TruFor) : {b['pixel_segmentation_score']:.4f}")
    print(f"  • Signal Noise Forensics (SRM): {b['signal_forensics_score']:.4f}")
    print(f"  • Typographic Baseline Jitter : {b['typography_score']:.4f}")
    print(f"  • DCT Double-Compression Grid : {b['dct_periodicity']:.4f}")
    print(f"  • ELA Compression Variance    : {b['ela_std']:.4f}")
    
    if res.get("overlay_url"):
        annotated_path = os.path.join(output_dir, os.path.basename(res["overlay_url"]))
        print(f"\n🖼️  Annotated Heatmap Overlay saved to:")
        print(f"   file://{os.path.abspath(annotated_path)}")
        
    print("=" * 70 + "\n")
    return res


def predict_batch(engine, folder_path, output_dir="outputs", recursive=True, detailed=True, log_file=None, json_output=None):
    """
    Runs batch inference on an entire folder of images or PDFs, with support
    for recursive directory search and detailed log outputs.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at '{folder_path}'")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    exts = [".jpg", ".jpeg", ".png", ".pdf"]
    files = []
    
    if recursive:
        for root, _, filenames in os.walk(folder_path):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                # Skip visualization images (_vis.jpg) and mask images
                if ext in exts and not f.endswith("_vis.jpg") and not "mask" in f.lower():
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in exts and not f.endswith("_vis.jpg") and not "mask" in f.lower():
                files.append(os.path.join(folder_path, f))
                
    files = sorted(list(set(files)))
    
    if not files:
        print(f"⚠️ No matching images or PDFs found in '{folder_path}'")
        return
        
    log_lines = []
    def log_print(msg):
        print(msg)
        log_lines.append(msg)
        
    log_print(f"\n🚀 Running Batch Fraud Inference on {len(files)} documents in '{folder_path}'...")
    log_print("=" * 80)
    
    counts = {"GENUINE": 0, "TAMPERED": 0, "FAILED": 0}
    batch_results = []
    t0 = time.time()
    
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        try:
            res = engine.analyze_document(fpath, output_dir=output_dir)
            status = res["status"]
            conf = res["confidence"]
            counts[status] += 1
            batch_results.append(res)
            
            icon = "⚠️" if status == "TAMPERED" else "✅"
            log_print(f"[{i:04d}/{len(files):04d}] {icon} {status:<8} | Conf: {conf:.1f}% | File: {fname}")
            
            if detailed and res.get("tamper_boxes"):
                for b_idx, box in enumerate(res["tamper_boxes"], 1):
                    log_print(f"        └─ Box #{b_idx}: {format_box(box['box'])} (Conf: {box['confidence']*100:.1f}%) — {box['reason']}")
            if detailed:
                log_print(f"        └─ Evidence: {res['tamper_reason']}")
                log_print("        " + "-" * 72)
                
        except Exception as e:
            counts["FAILED"] += 1
            log_print(f"[{i:04d}/{len(files):04d}] ❌ FAILED   | File: {fname} | Error: {str(e)}")
            
    total_time = time.time() - t0
    log_print("=" * 80)
    log_print(f"📊 BATCH INFERENCE SUMMARY ({total_time:.2f}s, {total_time/len(files):.2f}s/doc):")
    log_print(f"   • Total Files Processed : {len(files)}")
    log_print(f"   • ✅ Genuine Documents   : {counts['GENUINE']}")
    log_print(f"   • ⚠️ Tampered Documents  : {counts['TAMPERED']}")
    log_print(f"   • ❌ Failed Processing    : {counts['FAILED']}")
    log_print(f"   • Annotated Visuals in   : {os.path.abspath(output_dir)}")
    log_print("=" * 80 + "\n")
    
    # Write to log file if requested
    if log_file:
        with open(log_file, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"📝 Full batch log saved to: {os.path.abspath(log_file)}")
        
    # Write JSON results if requested
    if json_output:
        with open(json_output, "w") as f:
            json.dump({
                "summary": counts,
                "total_files": len(files),
                "total_time_seconds": round(total_time, 2),
                "results": batch_results
            }, f, indent=2)
        print(f"📊 JSON summary exported to: {os.path.abspath(json_output)}")


def evaluate_test_split(
    model_path, 
    dataset_dir=DEFAULT_DATASET_DIR, 
    batch_size=16, 
    img_size=512, 
    log_file=None,
    output_dir=None,
    export_hierarchy=True
):
    """
    Runs comprehensive benchmark evaluation across BOTH unseen genuine and unseen tampered
    documents in the 15% test split (522 genuine + 522 tampered = 1,044 total test documents).
    
    When export_hierarchy is True and output_dir is provided, it organizes all results into:
      output_dir/
        by_outcome/
          true_positives_TP/<doc_id>/ (01_original, 02_input, 03_gt_mask, 04_predicted_overlay, 05_side_by_side, result.json)
          false_negatives_FN/<doc_id>/
          true_negatives_TN/<doc_id>/
          false_positives_FP/<doc_id>/
        galleries/
        classification_report.csv
        summary.json
        index.html (interactive comparison dashboard)
    """
    log_lines = []
    def log_print(msg):
        print(msg)
        log_lines.append(msg)

    log_print(f"\n🧪 Starting Comprehensive Benchmark Evaluation on Unseen Test Split...")
    log_print(f"📁 Dataset Directory: {dataset_dir}")
    if output_dir and export_hierarchy:
        log_print(f"🗂️  Exporting Structured Hierarchy to: {os.path.abspath(output_dir)}")
    
    splits = create_or_load_splits(dataset_dir=dataset_dir)
    test_samples = splits.get("test", [])
    
    if not test_samples:
        log_print("❌ Error: Test split not found in dataset_splits.json")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    engine = DocumentFraudEngine(model_path=model_path, device=device)
    
    # 1. Pixel-Level Evaluation on Tampered Test Set (with Ground Truth Masks)
    log_print(f"\n[1/3] Evaluating Pixel-Level Localization on {len(test_samples)} Unseen Tampered Pages...")
    test_dataset = DocumentTamperDataset(test_samples, target_size=(img_size, img_size), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0)
    criterion = FocalDiceLoss()
    
    pixel_metrics = evaluate(
        engine.model, test_loader, criterion, device, 
        precision="fp32" if device.type != "cuda" else "bf16"
    )
    
    # Temp folder for intermediate overlays if exporting hierarchy
    temp_overlay_dir = os.path.join(output_dir, "_temp_overlays") if output_dir else None
    if temp_overlay_dir:
        os.makedirs(temp_overlay_dir, exist_ok=True)

    evaluated_records = []

    # 2. Document-Level End-to-End Evaluation on TAMPERED Test Pages
    log_print(f"[2/3] Evaluating End-to-End Fraud Pipeline on {len(test_samples)} TAMPERED Pages...")
    tamp_tp = 0
    tamp_fn = 0
    tamp_errors = []
    
    for idx, item in enumerate(test_samples, 1):
        t_path = item["tampered_image"]
        sample_id = item.get("id", f"tampered_doc_{idx}")
        if os.path.exists(t_path):
            res = engine.analyze_document(t_path, output_dir=temp_overlay_dir)
            if res["status"] == "TAMPERED":
                tamp_tp += 1
            else:
                tamp_fn += 1
                bd = res.get("breakdown", {})
                tamp_errors.append((
                    os.path.basename(t_path),
                    res["confidence"],
                    res["composite_fraud_score"],
                    bd.get("tamper_area_ratio", 0),
                    bd.get("tamper_peak_score", 0)
                ))
            
            if output_dir and export_hierarchy:
                rec = PredictionHierarchyOrganizer.organize_evaluation_sample(
                    output_base_dir=output_dir,
                    sample_id=sample_id,
                    input_img_path=t_path,
                    gt_label="TAMPERED",
                    result_dict=res,
                    orig_img_path=item.get("original_image"),
                    gt_mask_path=item.get("mask_image"),
                    ocr_json_path=item.get("ocr_json")
                )
                evaluated_records.append(rec)
                
    # 3. Document-Level End-to-End Evaluation on GENUINE (Original) Test Pages
    log_print(f"[3/3] Evaluating End-to-End Fraud Pipeline on {len(test_samples)} GENUINE Pages...")
    gen_tn = 0
    gen_fp = 0
    gen_errors = []
    gen_count = 0
    
    for idx, item in enumerate(test_samples, 1):
        o_path = item.get("original_image")
        sample_id = f"genuine_{item.get('doc_name', 'doc')}_page_{item.get('page', idx)}"
        if o_path and os.path.exists(o_path):
            gen_count += 1
            res = engine.analyze_document(o_path, output_dir=temp_overlay_dir)
            if res["status"] == "GENUINE":
                gen_tn += 1
            else:
                gen_fp += 1
                bd = res.get("breakdown", {})
                gen_errors.append((
                    os.path.basename(o_path),
                    res["confidence"],
                    res["composite_fraud_score"],
                    bd.get("tamper_area_ratio", 0),
                    bd.get("tamper_peak_score", 0)
                ))
                
            if output_dir and export_hierarchy:
                rec = PredictionHierarchyOrganizer.organize_evaluation_sample(
                    output_base_dir=output_dir,
                    sample_id=sample_id,
                    input_img_path=o_path,
                    gt_label="GENUINE",
                    result_dict=res,
                    orig_img_path=o_path,
                    gt_mask_path=None,
                    ocr_json_path=item.get("ocr_json")
                )
                evaluated_records.append(rec)
                
    # 4. Summary & Metrics Calculation
    total_docs = len(test_samples) + gen_count
    overall_acc = round(((tamp_tp + gen_tn) / max(1, total_docs)) * 100.0, 2)
    tamp_recall = round((tamp_tp / max(1, len(test_samples))) * 100.0, 2)
    gen_specificity = round((gen_tn / max(1, gen_count)) * 100.0 if gen_count > 0 else 100.0, 2)
    doc_precision = round((tamp_tp / max(1, tamp_tp + gen_fp)) * 100.0, 2)
    doc_f1 = round((2.0 * doc_precision * tamp_recall) / max(1e-4, doc_precision + tamp_recall), 2)
    
    summary_data = {
        "total_documents": total_docs,
        "tampered_total": len(test_samples),
        "genuine_total": gen_count,
        "overall_acc": overall_acc,
        "tamp_recall": tamp_recall,
        "gen_specificity": gen_specificity,
        "doc_precision": doc_precision,
        "doc_f1": doc_f1,
        "pixel_f1": round(pixel_metrics.get("pixel_f1", 0) * 100.0, 2),
        "pixel_iou": round(pixel_metrics.get("pixel_iou", 0) * 100.0, 2),
        "pixel_precision": round(pixel_metrics.get("pixel_precision", 0) * 100.0, 2),
        "pixel_recall": round(pixel_metrics.get("pixel_recall", 0) * 100.0, 2),
        "confusion_matrix": {
            "true_positives_TP": tamp_tp,
            "false_negatives_FN": tamp_fn,
            "true_negatives_TN": gen_tn,
            "false_positives_FP": gen_fp
        }
    }

    log_print("\n" + "=" * 80)
    log_print(f"🏆 COMPLETE BENCHMARK RESULTS ON UNSEEN TEST SPLIT ({total_docs} Total Pages)")
    log_print("=" * 80)
    log_print(f"📊 DOCUMENT-LEVEL CLASSIFICATION PERFORMANCE:")
    log_print(f"   • Overall Balanced Accuracy : {overall_acc:.2f}%")
    log_print(f"   • Document F1-Score         : {doc_f1:.2f}%")
    log_print(f"   • Document Precision        : {doc_precision:.2f}% (Resistance to False Positives)")
    log_print(f"   • Document Recall (Sensitivity) : {tamp_recall:.2f}% (Catch Rate on Tampered Bills)")
    log_print("-" * 80)
    log_print(f"📋 CONFUSION MATRIX BREAKDOWN:")
    log_print(f"   • ⚠️  Tampered Pages Caught (True Positives)  : {tamp_tp} / {len(test_samples)} ({tamp_recall:.2f}%)")
    log_print(f"   • ⚠️  Tampered Pages Missed (False Negatives)  : {tamp_fn} / {len(test_samples)}")
    log_print(f"   • ✅ Genuine Pages Verified (True Negatives) : {gen_tn} / {gen_count} ({gen_specificity:.2f}%)")
    log_print(f"   • ❌ Genuine Pages Flagged (False Positives)  : {gen_fp} / {gen_count}")
    log_print("-" * 80)
    log_print(f"🔍 PIXEL-LEVEL TAMPER LOCALIZATION (TruFor Segmentor):")
    log_print(f"   • Pixel F1-Score (Dice)     : {summary_data['pixel_f1']:.2f}%")
    log_print(f"   • Pixel IoU (Jaccard)       : {summary_data['pixel_iou']:.2f}%")
    log_print(f"   • Pixel Precision           : {summary_data['pixel_precision']:.2f}%")
    log_print(f"   • Pixel Recall              : {summary_data['pixel_recall']:.2f}%")
    log_print("=" * 80)
    
    if gen_errors:
        log_print("\n⚠️ False Positive Genuine Documents (map_area > 0.8% OR peak > 0.90 triggered):")
        for row in gen_errors[:10]:
            name, conf, sc, area, peak = row
            log_print(f"   - {name} | conf={conf:.1f}% score={sc:.4f} | area_ratio={area:.5f} peak={peak:.4f}")
    if tamp_errors:
        log_print("\n⚠️ False Negative Tampered Documents (missed by all tiers):")
        for row in tamp_errors[:10]:
            name, conf, sc, area, peak = row
            log_print(f"   - {name} | conf={conf:.1f}% score={sc:.4f} | area_ratio={area:.5f} peak={peak:.4f}")
    log_print("=" * 80 + "\n")
    
    # 5. Export Structured Reports & Clean Up Temp Overlays
    if output_dir and export_hierarchy:
        # Save summary JSON
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)
            
        # Export CSV report
        csv_p = PredictionHierarchyOrganizer.generate_csv_report(output_dir, evaluated_records)
        
        # Export Interactive HTML Dashboard
        html_p = PredictionHierarchyOrganizer.generate_interactive_html_dashboard(output_dir, summary_data, evaluated_records)
        
        # Clean temp directory
        if temp_overlay_dir and os.path.exists(temp_overlay_dir):
            import shutil
            shutil.rmtree(temp_overlay_dir, ignore_errors=True)
            
        log_print(f"✨ Structured evaluation hierarchy successfully generated!")
        log_print(f"   • 📁 Root Output Folder : file://{os.path.abspath(output_dir)}")
        log_print(f"   • 🌐 Visual Dashboard   : file://{os.path.abspath(html_p)}")
        log_print(f"   • 📊 Classification CSV : file://{os.path.abspath(csv_p)}")
        log_print(f"   • 📑 Metrics JSON       : file://{os.path.abspath(os.path.join(output_dir, 'summary.json'))}")
        log_print("=" * 80 + "\n")

    if log_file:
        with open(log_file, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"📝 Full evaluation report saved to: {os.path.abspath(log_file)}")


def main():
    parser = argparse.ArgumentParser(description="Fraud Engine v2 — High-Precision Inference & Prediction CLI")
    parser.add_argument("--image", type=str, help="Path to a single document image or PDF")
    parser.add_argument("--dir", type=str, help="Path to a directory of images/PDFs for batch prediction")
    parser.add_argument("--ocr_json", type=str, help="Optional path to OCR custom JSON metadata")
    parser.add_argument("--model_path", type=str, default=os.path.join(BASE_DIR, "models", "tamper_segmentor.pth"), 
                        help="Path to trained model weights")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "outputs"), 
                        help="Directory to save organized evaluation hierarchy or annotated overlays")
    parser.add_argument("--evaluate_test", action="store_true", help="Run full benchmark evaluation on unseen test split")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset base path for test evaluation")
    parser.add_argument("--json_output", type=str, help="Save result to a JSON file")
    parser.add_argument("--log_file", type=str, help="Save full batch terminal log to a .log text file")
    parser.add_argument("--no_recursive", action="store_true", help="Disable recursive subfolder search in batch mode")
    parser.add_argument("--no_hierarchy", action="store_true", help="Disable structured folder hierarchy creation during evaluation")
    
    args = parser.parse_args()
    
    if args.evaluate_test:
        evaluate_test_split(
            args.model_path, 
            dataset_dir=args.dataset_dir, 
            log_file=args.log_file,
            output_dir=args.output_dir,
            export_hierarchy=(not args.no_hierarchy)
        )
        sys.exit(0)
        
    engine = DocumentFraudEngine(model_path=args.model_path)
    
    if args.image:
        result = predict_single(engine, args.image, output_dir=args.output_dir, ocr_json=args.ocr_json)
        if args.json_output and result:
            with open(args.json_output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"JSON result exported to {args.json_output}")
    elif args.dir:
        predict_batch(
            engine, 
            args.dir, 
            output_dir=args.output_dir, 
            recursive=(not args.no_recursive), 
            log_file=args.log_file,
            json_output=args.json_output
        )
    else:
        print("ℹ️ Please provide an image with --image <path>, a folder with --dir <path>, or use --evaluate_test.")
        print("Run with --help for full usage details.")


if __name__ == "__main__":
    main()

