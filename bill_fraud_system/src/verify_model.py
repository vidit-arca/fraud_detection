import os
import glob
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bill_preprocessing import load_and_preprocess_image, create_multiscale_patches
from src.feature_extractor import FeatureExtractor, ForensicFeatureExtractor
from src.outlier_detector import AnomalyDetector
from src.pipeline import score_image


def verify_folder(folder_path, deep_extractor, detector, expected_label, threshold):
    print(f"\n--- Verifying {os.path.basename(folder_path)} (Expected: {expected_label}) ---")
    
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_paths:
        print("No images found.")
        return 0, 0, [], []

    correct_count = 0
    total_count = len(image_paths)
    errors = []
    all_scores = []

    for path in image_paths:
        filename = os.path.basename(path)
        try:
            combined, deep_score, forensic_score = score_image(path, deep_extractor, detector)
            
            if combined is None:
                print(f"[SKIP] {filename} - Load Failed")
                total_count -= 1
                continue
            
            all_scores.append(combined)
            
            is_anomalous = combined > threshold
            predicted_label = "TAMPERED" if is_anomalous else "GENUINE"
            is_correct = (predicted_label == expected_label)
            
            if is_correct:
                correct_count += 1
                icon = "✅"
            else:
                icon = "❌"
                errors.append((filename, combined))

            print(f"{icon} {filename:<40} Combined: {combined:.4f} "
                  f"(deep={deep_score:.4f}, forensic={forensic_score:.4f}) -> {predicted_label}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

    return correct_count, total_count, errors, all_scores


def threshold_sweep(gen_scores, tamp_scores):
    """Find optimal threshold via sweep."""
    if not gen_scores or not tamp_scores:
        return None
    
    all_vals = sorted(gen_scores + tamp_scores)
    best_t, best_acc = 0, 0
    
    print(f"\n{'Threshold':>10} | {'Gen Acc':>8} | {'Tamp Acc':>8} | {'Overall':>8}")
    print("-" * 48)
    
    for t in np.linspace(min(all_vals) - 0.5, max(all_vals) + 0.5, 50):
        gc = sum(1 for s in gen_scores if s <= t)
        tc = sum(1 for s in tamp_scores if s > t)
        ga = gc / len(gen_scores)
        ta = tc / len(tamp_scores)
        oa = (gc + tc) / (len(gen_scores) + len(tamp_scores))
        if oa > best_acc:
            best_acc = oa
            best_t = t
    
    # Print interesting points
    for t in np.linspace(min(all_vals) - 0.5, max(all_vals) + 0.5, 15):
        gc = sum(1 for s in gen_scores if s <= t)
        tc = sum(1 for s in tamp_scores if s > t)
        ga = gc / len(gen_scores)
        ta = tc / len(tamp_scores)
        oa = (gc + tc) / (len(gen_scores) + len(tamp_scores))
        print(f"{t:>10.4f} | {ga:>7.1%} | {ta:>7.1%} | {oa:>7.1%}")
    
    print(f"\n🏆 Best threshold: {best_t:.4f} (overall: {best_acc:.1%})")
    return best_t


if __name__ == "__main__":
    MODEL_PATH = "models/patch_model.pkl"
    
    # Load model
    detector = AnomalyDetector()
    try:
        detector.load_model(MODEL_PATH)
        THRESHOLD = detector.calibration.get('image_threshold', 2.0)
        print(f"Using calibrated threshold: {THRESHOLD:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    deep_extractor = FeatureExtractor(model_name='efficientnet_b0')
    
    # Verify GENUINE
    g_correct, g_total, g_errors, g_scores = verify_folder(
        "data/train_genuine", deep_extractor, detector, "GENUINE", THRESHOLD)
    
    # Verify TAMPERED
    t_correct, t_total, t_errors, t_scores = verify_folder(
        "Tamp", deep_extractor, detector, "TAMPERED", THRESHOLD)
    
    # Report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if g_total > 0:
        print(f"GENUINE:  {g_correct}/{g_total} ({100*g_correct/g_total:.1f}%)")
        if g_errors:
            print("  False Positives:")
            for name, score in g_errors:
                print(f"    - {name}: {score:.4f}")
    
    print("-" * 40)
    
    if t_total > 0:
        print(f"TAMPERED: {t_correct}/{t_total} ({100*t_correct/t_total:.1f}%)")
        if t_errors:
            print("  False Negatives:")
            for name, score in t_errors:
                print(f"    - {name}: {score:.4f}")
    
    print("=" * 60)
    
    if g_scores and t_scores:
        print("\n--- Threshold Sweep ---")
        threshold_sweep(g_scores, t_scores)
