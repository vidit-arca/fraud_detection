import os
import shutil
import json
import csv
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class PredictionHierarchyOrganizer:
    """
    Organizes model prediction and evaluation results into a clean, 
    scalable folder hierarchy with side-by-side visual comparisons,
    detailed metadata, CSV classification reports, and an interactive HTML gallery.
    """

    @staticmethod
    def create_side_by_side_comparison(
        input_img_path,
        overlay_img_path,
        gt_mask_path=None,
        orig_img_path=None,
        result_dict=None,
        gt_label="TAMPERED",
        target_height=600
    ):
        """
        Builds a multi-panel annotated side-by-side comparison image:
        [Original Doc (if avail)] | [Input Doc] | [Ground Truth Mask] | [Predicted Fraud Overlay]
        with a top banner indicating Ground Truth vs Prediction, Confidence, Score, and Outcome.
        """
        panels = []
        titles = []

        def load_and_resize(path):
            if not path or not os.path.exists(path):
                return None
            img = Image.open(path).convert("RGB")
            # Proportional resize by height
            w, h = img.size
            if h <= 0:
                return None
            new_w = max(10, int(w * (target_height / float(h))))
            return img.resize((new_w, target_height), Image.Resampling.LANCZOS)

        # 1. Original Reference (if available)
        if orig_img_path and os.path.exists(orig_img_path):
            orig_panel = load_and_resize(orig_img_path)
            if orig_panel:
                panels.append(orig_panel)
                titles.append("1. Original Reference (Genuine)")

        # 2. Input Document
        input_panel = load_and_resize(input_img_path)
        if input_panel is None:
            # Create a placeholder if input fails to load
            input_panel = Image.new("RGB", (target_height, target_height), color=(40, 40, 40))
        panels.append(input_panel)
        titles.append(f"2. Input Doc ({gt_label})")

        # 3. Ground Truth Mask Overlay (for tampered ground truth)
        if gt_mask_path and os.path.exists(gt_mask_path):
            mask_raw = Image.open(gt_mask_path).convert("L")
            mask_resized = mask_raw.resize(input_panel.size, Image.Resampling.NEAREST)
            mask_np = np.array(mask_resized) > 128
            
            # Create highlight overlay (Green tint over tampered pixels)
            inp_np = np.array(input_panel).copy()
            overlay_gt = inp_np.copy()
            overlay_gt[mask_np] = [0, 230, 100]  # bright green highlight
            blended_gt = cv2.addWeighted(inp_np, 0.45, overlay_gt, 0.55, 0)
            
            # Find contours and draw box around GT
            contours, _ = cv2.findContours((mask_np.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(blended_gt, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            gt_img = Image.fromarray(blended_gt)
            panels.append(gt_img)
            titles.append("3. Ground Truth Mask (Actual Forgery)")

        # 4. Model Predicted Overlay
        overlay_panel = load_and_resize(overlay_img_path)
        if overlay_panel:
            panels.append(overlay_panel)
            pred_status = result_dict.get("status", "UNKNOWN") if result_dict else "PREDICTION"
            titles.append(f"4. Predicted Heatmap & Boxes ({pred_status})")

        # Combine Panels Horizontally
        spacing = 16
        banner_height = 80
        panel_header_height = 36
        
        total_width = sum(p.width for p in panels) + spacing * (len(panels) + 1)
        total_height = target_height + banner_height + panel_header_height + 20

        comp_img = Image.new("RGB", (total_width, total_height), color=(22, 24, 29))
        draw = ImageDraw.Draw(comp_img)

        # Determine Outcome Category & Colors
        pred_status = result_dict.get("status", "UNKNOWN") if result_dict else "UNKNOWN"
        conf = result_dict.get("confidence", 0.0) if result_dict else 0.0
        score = result_dict.get("composite_fraud_score", 0.0) if result_dict else 0.0

        if gt_label == "TAMPERED" and pred_status == "TAMPERED":
            outcome = "TRUE POSITIVE (CAUGHT FORGERY)"
            banner_bg = (20, 80, 45) # Dark Green
            badge_text = "✅ TP: TAMPER DETECTED"
        elif gt_label == "GENUINE" and pred_status == "GENUINE":
            outcome = "TRUE NEGATIVE (VERIFIED AUTHENTIC)"
            banner_bg = (18, 65, 95) # Dark Blue/Teal
            badge_text = "✅ TN: AUTHENTIC VERIFIED"
        elif gt_label == "TAMPERED" and pred_status == "GENUINE":
            outcome = "FALSE NEGATIVE (MISSED FORGERY)"
            banner_bg = (110, 25, 25) # Red
            badge_text = "⚠️ FN: TAMPER MISSED"
        else: # Genuine flagged as Tampered
            outcome = "FALSE POSITIVE (FALSE ALARM)"
            banner_bg = (115, 60, 15) # Orange/Amber
            badge_text = "❌ FP: FALSE ALARM"

        # Draw Top Banner
        draw.rectangle([(0, 0), (total_width, banner_height)], fill=banner_bg)
        
        # Header Text
        fname = os.path.basename(input_img_path)
        draw.text((20, 12), f"FILE: {fname}", fill=(255, 255, 255))
        draw.text((20, 36), f"Ground Truth: {gt_label}   |   Prediction: {pred_status} (Conf: {conf:.1f}%, Score: {score:.4f})", fill=(220, 230, 242))
        draw.text((20, 56), f"Outcome: {outcome}", fill=(255, 255, 120))
        
        # Right badge on banner
        draw.text((max(20, total_width - 320), 28), badge_text, fill=(255, 255, 255))

        # Paste Panels with individual headers
        curr_x = spacing
        y_panels = banner_height + panel_header_height
        
        for idx, (p, title) in enumerate(zip(panels, titles)):
            # Panel Title Box
            draw.rectangle([(curr_x, banner_height + 6), (curr_x + p.width, banner_height + panel_header_height - 2)], fill=(35, 39, 48))
            draw.text((curr_x + 10, banner_height + 12), title, fill=(200, 215, 230))
            
            # Panel Image
            comp_img.paste(p, (curr_x, y_panels))
            # Border around image
            draw.rectangle([(curr_x - 1, y_panels - 1), (curr_x + p.width, y_panels + target_height)], outline=(60, 68, 80), width=1)
            curr_x += p.width + spacing

        return comp_img

    @staticmethod
    def organize_evaluation_sample(
        output_base_dir,
        sample_id,
        input_img_path,
        gt_label,
        result_dict,
        orig_img_path=None,
        gt_mask_path=None,
        ocr_json_path=None
    ):
        """
        Stores an evaluated document sample into a structured folder hierarchy:
        
        output_base_dir/
          by_outcome/
            {true_positives_TP, false_negatives_FN, true_negatives_TN, false_positives_FP}/
              <sample_id>/
                01_original.jpg
                02_input.jpg
                03_gt_mask.png
                04_predicted_overlay.jpg
                05_side_by_side.jpg
                result.json
        """
        pred_status = result_dict.get("status", "UNKNOWN")
        
        # Determine classification outcome folder
        if gt_label == "TAMPERED":
            if pred_status == "TAMPERED":
                outcome_folder = "true_positives_TP"
                outcome_code = "TP"
            else:
                outcome_folder = "false_negatives_FN"
                outcome_code = "FN"
        else: # GENUINE
            if pred_status == "GENUINE":
                outcome_folder = "true_negatives_TN"
                outcome_code = "TN"
            else:
                outcome_folder = "false_positives_FP"
                outcome_code = "FP"

        # 1. Target Sample Directory in by_outcome
        sample_dir = os.path.join(output_base_dir, "by_outcome", outcome_folder, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # 2. Copy/Save Individual Component Assets
        # 01_original.jpg
        saved_orig = None
        if orig_img_path and os.path.exists(orig_img_path):
            saved_orig = os.path.join(sample_dir, "01_original.jpg")
            shutil.copy2(orig_img_path, saved_orig)

        # 02_input.jpg
        saved_input = os.path.join(sample_dir, "02_input.jpg")
        if os.path.exists(input_img_path):
            shutil.copy2(input_img_path, saved_input)

        # 03_gt_mask.png
        saved_mask = None
        if gt_mask_path and os.path.exists(gt_mask_path):
            saved_mask = os.path.join(sample_dir, "03_gt_mask.png")
            shutil.copy2(gt_mask_path, saved_mask)

        # 04_predicted_overlay.jpg
        saved_overlay = None
        overlay_src = result_dict.get("_overlay_temp_path")
        if not overlay_src and result_dict.get("overlay_url"):
            # Check standard output dir
            c_name = os.path.basename(result_dict["overlay_url"])
            possible = os.path.join(output_base_dir, c_name)
            if os.path.exists(possible):
                overlay_src = possible
                
        if overlay_src and os.path.exists(overlay_src):
            saved_overlay = os.path.join(sample_dir, "04_predicted_overlay.jpg")
            shutil.copy2(overlay_src, saved_overlay)

        # Optional OCR JSON
        if ocr_json_path and os.path.exists(ocr_json_path):
            shutil.copy2(ocr_json_path, os.path.join(sample_dir, "ocr_labels.json"))

        # 05_side_by_side.jpg
        sbs_path = os.path.join(sample_dir, "05_side_by_side.jpg")
        sbs_img = PredictionHierarchyOrganizer.create_side_by_side_comparison(
            input_img_path=input_img_path,
            overlay_img_path=saved_overlay or overlay_src,
            gt_mask_path=gt_mask_path,
            orig_img_path=orig_img_path,
            result_dict=result_dict,
            gt_label=gt_label
        )
        sbs_img.save(sbs_path, "JPEG", quality=90)

        # Also copy side_by_side image into a flat gallery for quick scrolling
        gallery_dir = os.path.join(output_base_dir, "galleries", outcome_folder)
        os.makedirs(gallery_dir, exist_ok=True)
        shutil.copy2(sbs_path, os.path.join(gallery_dir, f"{sample_id}_sbs.jpg"))

        # 3. Save detailed result.json
        meta = {
            "sample_id": sample_id,
            "ground_truth": gt_label,
            "prediction": pred_status,
            "outcome": outcome_code,
            "is_correct": (gt_label == pred_status),
            "confidence": result_dict.get("confidence", 0.0),
            "composite_fraud_score": result_dict.get("composite_fraud_score", 0.0),
            "tamper_detected": result_dict.get("tamper_detected", False),
            "tamper_reason": result_dict.get("tamper_reason", ""),
            "tamper_regions_count": result_dict.get("tamper_regions_count", 0),
            "tamper_boxes": result_dict.get("tamper_boxes", []),
            "breakdown": result_dict.get("breakdown", {}),
            "document_info": result_dict.get("document_info", {}),
            "processing_time": result_dict.get("processing_time", 0.0),
            "assets": {
                "original_image": "01_original.jpg" if saved_orig else None,
                "input_image": "02_input.jpg",
                "ground_truth_mask": "03_gt_mask.png" if saved_mask else None,
                "predicted_overlay": "04_predicted_overlay.jpg" if saved_overlay else None,
                "side_by_side_comparison": "05_side_by_side.jpg"
            }
        }
        with open(os.path.join(sample_dir, "result.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "sample_id": sample_id,
            "sample_dir": os.path.relpath(sample_dir, output_base_dir),
            "ground_truth": gt_label,
            "prediction": pred_status,
            "outcome": outcome_code,
            "is_correct": (gt_label == pred_status),
            "confidence": result_dict.get("confidence", 0.0),
            "score": result_dict.get("composite_fraud_score", 0.0),
            "tamper_area_ratio": result_dict.get("breakdown", {}).get("tamper_area_ratio", 0.0),
            "tamper_peak_score": result_dict.get("breakdown", {}).get("tamper_peak_score", 0.0),
            "tamper_boxes_count": result_dict.get("tamper_regions_count", 0),
            "tamper_reason": result_dict.get("tamper_reason", ""),
            "sbs_rel_path": os.path.relpath(sbs_path, output_base_dir)
        }

    @staticmethod
    def generate_csv_report(output_dir, records):
        """
        Exports tabular CSV report containing all evaluated documents.
        """
        csv_path = os.path.join(output_dir, "classification_report.csv")
        fieldnames = [
            "sample_id", "ground_truth", "prediction", "outcome", "is_correct",
            "confidence", "score", "tamper_area_ratio", "tamper_peak_score",
            "tamper_boxes_count", "tamper_reason", "sample_dir", "sbs_rel_path"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        return csv_path

    @staticmethod
    def generate_interactive_html_dashboard(output_dir, benchmark_metrics, records):
        """
        Builds a self-contained interactive visual comparison dashboard in HTML/JS.
        Allows instant filtering by TP, FP, FN, TN, searching, and viewing side-by-side overlays.
        """
        html_path = os.path.join(output_dir, "index.html")
        
        # Prepare records for embedded JSON
        embedded_json = json.dumps(records)
        metrics_json = json.dumps(benchmark_metrics)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fraud Engine v2 — Ground Truth vs Prediction Visual Benchmark</title>
  <style>
    :root {{
      --bg-primary: #0f1117;
      --bg-card: #181b24;
      --bg-card-hover: #222634;
      --border-color: #2b3042;
      --text-main: #f0f3fa;
      --text-muted: #8d98b0;
      --color-tp: #10b981;
      --color-tn: #3b82f6;
      --color-fp: #f59e0b;
      --color-fn: #ef4444;
      --accent: #6366f1;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg-primary); color: var(--text-main); padding: 24px; }}
    
    .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); }}
    .header h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
    .header p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
    
    /* Metrics Grid */
    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; text-align: center; }}
    .metric-val {{ font-size: 26px; font-weight: 800; margin-top: 6px; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }}
    
    /* Controls Bar */
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin-bottom: 24px; background: var(--bg-card); padding: 14px 18px; border-radius: 12px; border: 1px solid var(--border-color); }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .filter-btn {{ background: transparent; border: 1px solid var(--border-color); color: var(--text-main); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px; transition: all 0.2s; }}
    .filter-btn:hover {{ background: var(--bg-card-hover); }}
    .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .search-input {{ background: #0c0d12; border: 1px solid var(--border-color); color: #fff; padding: 8px 14px; border-radius: 8px; font-size: 13px; width: 260px; outline: none; }}
    .search-input:focus {{ border-color: var(--accent); }}

    /* Gallery Grid */
    .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }}
    .card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; transition: transform 0.15s, border-color 0.15s; }}
    .card:hover {{ transform: translateY(-2px); border-color: #434c68; }}
    
    .card-img-wrap {{ position: relative; width: 100%; height: 260px; background: #000; cursor: pointer; overflow: hidden; }}
    .card-img-wrap img {{ width: 100%; height: 100%; object-fit: contain; }}
    .card-badge {{ position: absolute; top: 10px; right: 10px; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; color: #fff; letter-spacing: 0.5px; }}
    .badge-TP {{ background: var(--color-tp); }}
    .badge-TN {{ background: var(--color-tn); }}
    .badge-FP {{ background: var(--color-fp); }}
    .badge-FN {{ background: var(--color-fn); }}
    
    .card-body {{ padding: 14px 16px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }}
    .card-title {{ font-size: 13px; font-weight: 600; color: #fff; word-break: break-all; margin-bottom: 8px; }}
    .card-meta {{ font-size: 12px; color: var(--text-muted); line-height: 1.6; }}
    .card-footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); }}
    .view-btn {{ font-size: 12px; color: var(--accent); text-decoration: none; font-weight: 600; }}
    .view-btn:hover {{ text-decoration: underline; }}
    
    /* Modal */
    .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); justify-content: center; align-items: center; padding: 20px; }}
    .modal img {{ max-width: 95%; max-height: 90vh; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,0.8); }}
    .modal-close {{ position: absolute; top: 20px; right: 30px; font-size: 32px; color: #fff; cursor: pointer; }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>🛡️ Fraud Engine v2 — Benchmark Prediction Hierarchy</h1>
      <p>Unseen Test Split Comprehensive Comparison (Ground Truth vs Model Output)</p>
    </div>
    <div>
      <a href="classification_report.csv" download class="filter-btn active" style="text-decoration:none;">📥 Export CSV Report</a>
    </div>
  </div>

  <div class="metrics-grid" id="metricsContainer">
    <!-- Populated by JS -->
  </div>

  <div class="controls">
    <div class="filters">
      <button class="filter-btn active" onclick="filterGallery('ALL')">All Documents (<span id="count-ALL">0</span>)</button>
      <button class="filter-btn" onclick="filterGallery('TP')" style="color:var(--color-tp);">✅ True Positives (<span id="count-TP">0</span>)</button>
      <button class="filter-btn" onclick="filterGallery('TN')" style="color:var(--color-tn);">✅ True Negatives (<span id="count-TN">0</span>)</button>
      <button class="filter-btn" onclick="filterGallery('FP')" style="color:var(--color-fp);">❌ False Positives (<span id="count-FP">0</span>)</button>
      <button class="filter-btn" onclick="filterGallery('FN')" style="color:var(--color-fn);">⚠️ False Negatives (<span id="count-FN">0</span>)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="🔍 Search document name..." onkeyup="searchGallery()">
  </div>

  <div class="gallery-grid" id="galleryGrid">
    <!-- Cards populated by JS -->
  </div>

  <div id="imageModal" class="modal" onclick="closeModal()">
    <span class="modal-close">&times;</span>
    <img id="modalImg" src="">
  </div>

  <script>
    const records = {embedded_json};
    const metrics = {metrics_json};
    let currentFilter = 'ALL';

    function renderMetrics() {{
      const container = document.getElementById('metricsContainer');
      container.innerHTML = `
        <div class="metric-card"><div class="metric-label">Balanced Accuracy</div><div class="metric-val" style="color:#10b981">${{metrics.overall_acc || 0}}%</div></div>
        <div class="metric-card"><div class="metric-label">Tampered Recall</div><div class="metric-val" style="color:#34d399">${{metrics.tamp_recall || 0}}%</div></div>
        <div class="metric-card"><div class="metric-label">Genuine Specificity</div><div class="metric-val" style="color:#60a5fa">${{metrics.gen_specificity || 0}}%</div></div>
        <div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val" style="color:#a78bfa">${{metrics.doc_precision || 0}}%</div></div>
        <div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-val" style="color:#f472b6">${{metrics.doc_f1 || 0}}%</div></div>
        <div class="metric-card"><div class="metric-label">Pixel F1 (Dice)</div><div class="metric-val" style="color:#fbbf24">${{metrics.pixel_f1 || 0}}%</div></div>
      `;
      
      // Update counts
      const counts = {{ ALL: records.length, TP: 0, TN: 0, FP: 0, FN: 0 }};
      records.forEach(r => {{ counts[r.outcome] = (counts[r.outcome] || 0) + 1; }});
      for (const k in counts) {{
        const el = document.getElementById('count-' + k);
        if (el) el.innerText = counts[k];
      }}
    }}

    function renderGallery(items) {{
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = items.map(r => `
        <div class="card" data-outcome="${{r.outcome}}" data-name="${{r.sample_id.toLowerCase()}}">
          <div class="card-img-wrap" onclick="openModal('${{r.sbs_rel_path}}')">
            <img src="${{r.sbs_rel_path}}" loading="lazy" alt="${{r.sample_id}}">
            <span class="card-badge badge-${{r.outcome}}">${{r.outcome}}</span>
          </div>
          <div class="card-body">
            <div>
              <div class="card-title">${{r.sample_id}}</div>
              <div class="card-meta">
                <div><strong>GT:</strong> ${{r.ground_truth}} &nbsp;|&nbsp; <strong>Pred:</strong> ${{r.prediction}}</div>
                <div><strong>Conf:</strong> ${{r.confidence}}% &nbsp;|&nbsp; <strong>Score:</strong> ${{r.score}}</div>
                <div><strong>Peak:</strong> ${{r.tamper_peak_score}} &nbsp;|&nbsp; <strong>Area:</strong> ${{r.tamper_area_ratio}}</div>
              </div>
            </div>
            <div class="card-footer">
              <span style="font-size:11px; color:#6b7280;">${{r.tamper_boxes_count}} tamper box(es)</span>
              <a href="${{r.sample_dir}}/result.json" target="_blank" class="view-btn">View JSON ↗</a>
            </div>
          </div>
        </div>
      `).join('');
    }}

    function filterGallery(type) {{
      currentFilter = type;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      applyFilters();
    }}

    function searchGallery() {{
      applyFilters();
    }}

    function applyFilters() {{
      const q = document.getElementById('searchInput').value.toLowerCase();
      const filtered = records.filter(r => {{
        const matchType = (currentFilter === 'ALL' || r.outcome === currentFilter);
        const matchQuery = !q || r.sample_id.toLowerCase().includes(q) || r.tamper_reason.toLowerCase().includes(q);
        return matchType && matchQuery;
      }});
      renderGallery(filtered);
    }}

    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('imageModal').style.display = 'flex';
    }}

    function closeModal() {{
      document.getElementById('imageModal').style.display = 'none';
    }}

    // Initial render
    renderMetrics();
    renderGallery(records);
  </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return html_path
