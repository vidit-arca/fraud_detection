import os
import glob
import numpy as np
import argparse
import sys
from PIL import Image

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bill_preprocessing import load_and_preprocess_image, create_multiscale_patches
from src.feature_extractor import FeatureExtractor, ForensicFeatureExtractor
from src.outlier_detector import AnomalyDetector

N_AUGMENTS = 3  # Augmented copies per training image


def train_pipeline(data_dir, model_save_path):
    print("=" * 60)
    print("BILL FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    
    print("\n[1/4] Initializing Feature Extractors...")
    deep_extractor = FeatureExtractor(model_name='efficientnet_b0')
    
    # Collect all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    
    if not image_paths:
        print("No images found in training directory!")
        return

    # === Extract Deep Features (with augmentation) ===
    print(f"\n[2/4] Extracting deep features from {len(image_paths)} images "
          f"(x{N_AUGMENTS + 1} with augmentation)...")
    
    all_deep_features = []
    for idx, path in enumerate(image_paths):
        # Original
        tensor = load_and_preprocess_image(path, augment=False)
        if tensor is not None:
            patches = create_multiscale_patches(tensor)
            feats = deep_extractor.extract(patches)
            all_deep_features.append(feats)
        
        # Augmented
        for _ in range(N_AUGMENTS):
            tensor_aug = load_and_preprocess_image(path, augment=True)
            if tensor_aug is not None:
                patches_aug = create_multiscale_patches(tensor_aug)
                feats_aug = deep_extractor.extract(patches_aug)
                all_deep_features.append(feats_aug)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images...")
    
    # === Extract Forensic Features (no augmentation — need original artifacts) ===
    print(f"\n[3/4] Extracting forensic features from {len(image_paths)} images...")
    all_forensic_features = []
    for path in image_paths:
        try:
            pil_image = Image.open(path).convert("RGB")
            forensic = ForensicFeatureExtractor.extract_image_forensics(pil_image)
            all_forensic_features.append(forensic)
        except Exception as e:
            print(f"  Warning: forensic extraction failed for {os.path.basename(path)}: {e}")
    
    if not all_deep_features or not all_forensic_features:
        print("Failed to extract features.")
        return

    X_deep = np.vstack(all_deep_features)
    X_forensic = np.array(all_forensic_features)
    print(f"  Deep patches: {X_deep.shape}")
    print(f"  Forensic vectors: {X_forensic.shape}")
    
    # === Train Detector ===
    print(f"\n[4/4] Training Anomaly Detector...")
    detector = AnomalyDetector(contamination='auto', n_estimators=300, use_pca=True)
    detector.train(X_deep, X_forensic)
    
    # === Calibrate image-level thresholds ===
    print("\nCalibrating image-level thresholds on training data...")
    deep_image_scores = []
    forensic_image_scores = []
    
    for i, path in enumerate(image_paths):
        # Deep score
        tensor = load_and_preprocess_image(path, augment=False)
        if tensor is None:
            continue
        patches = create_multiscale_patches(tensor)
        feats = deep_extractor.extract(patches)
        deep_patch_scores = detector.predict_deep(feats)
        
        # Top-k aggregation for deep scores
        k = min(5, len(deep_patch_scores))
        img_deep_score = np.mean(np.sort(deep_patch_scores)[-k:])
        deep_image_scores.append(img_deep_score)
        
        # Forensic score
        if i < len(all_forensic_features):
            forensic_score = detector.predict_forensic(all_forensic_features[i])
            forensic_image_scores.append(float(forensic_score[0]))
    
    deep_image_scores = np.array(deep_image_scores)
    forensic_image_scores = np.array(forensic_image_scores)
    
    # Store calibration stats
    detector.calibration['deep_image_mean'] = float(np.mean(deep_image_scores))
    detector.calibration['deep_image_std'] = float(np.std(deep_image_scores)) + 1e-8
    detector.calibration['forensic_image_mean'] = float(np.mean(forensic_image_scores))
    detector.calibration['forensic_image_std'] = float(np.std(forensic_image_scores)) + 1e-8
    
    # Combined score for training images
    deep_norm = (deep_image_scores - np.mean(deep_image_scores)) / (np.std(deep_image_scores) + 1e-8)
    forensic_norm = (forensic_image_scores - np.mean(forensic_image_scores)) / (np.std(forensic_image_scores) + 1e-8)
    combined_scores = 0.35 * deep_norm + 0.65 * forensic_norm
    
    # Threshold: mean + 2σ of combined scores
    threshold = float(np.mean(combined_scores) + 2.0 * np.std(combined_scores))
    detector.calibration['image_threshold'] = threshold
    
    print(f"\nCalibration Results:")
    print(f"  Deep scores: mean={np.mean(deep_image_scores):.4f}, std={np.std(deep_image_scores):.4f}")
    print(f"  Forensic scores: mean={np.mean(forensic_image_scores):.4f}, std={np.std(forensic_image_scores):.4f}")
    print(f"  Combined threshold: {threshold:.4f}")
    
    detector.save_model(model_save_path)
    print("\n✅ Training complete!")


def score_image(image_path, deep_extractor, detector):
    """Score a single image using both deep and forensic features."""
    # Deep score
    tensor = load_and_preprocess_image(image_path, augment=False)
    if tensor is None:
        return None, None, None
    
    patches = create_multiscale_patches(tensor)
    feats = deep_extractor.extract(patches)
    deep_patch_scores = detector.predict_deep(feats)
    
    k = min(5, len(deep_patch_scores))
    deep_score = np.mean(np.sort(deep_patch_scores)[-k:])
    
    # Forensic score
    pil_image = Image.open(image_path).convert("RGB")
    forensic_feats = ForensicFeatureExtractor.extract_image_forensics(pil_image)
    forensic_score = float(detector.predict_forensic(forensic_feats)[0])
    
    # Normalize and combine
    deep_norm = (deep_score - detector.calibration['deep_image_mean']) / detector.calibration['deep_image_std']
    forensic_norm = (forensic_score - detector.calibration['forensic_image_mean']) / detector.calibration['forensic_image_std']
    
    combined = 0.35 * deep_norm + 0.65 * forensic_norm
    
    return combined, deep_score, forensic_score


def inference_pipeline(image_path, model_path):
    print(f"Processing {image_path}...")
    
    detector = AnomalyDetector()
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    deep_extractor = FeatureExtractor(model_name='efficientnet_b0')
    
    combined, deep_score, forensic_score = score_image(image_path, deep_extractor, detector)
    
    if combined is None:
        print("Could not load image.")
        return
    
    threshold = detector.calibration.get('image_threshold', 2.0)
    is_anomalous = combined > threshold
    
    print(f"\n--- Result ---")
    print(f"Combined Score: {combined:.4f} (threshold: {threshold:.4f})")
    print(f"  Deep Feature Score: {deep_score:.4f}")
    print(f"  Forensic Score: {forensic_score:.4f}")
    
    if is_anomalous:
        print(f"\n>> ⚠️  FLAGGED AS TAMPERED / ANOMALOUS <<")
    else:
        print(f"\n>> ✅ CLASSIFIED AS GENUINE <<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bill Fraud Detection Pipeline")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train or predict")
    parser.add_argument('--data_dir', type=str, help="Directory containing genuine bills for training")
    parser.add_argument('--model_path', type=str, default='models/patch_model.pkl', help="Path to save/load model")
    parser.add_argument('--image_path', type=str, help="Path to image for prediction")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_dir:
            print("Error: --data_dir is required for training.")
        else:
            train_pipeline(args.data_dir, args.model_path)
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image_path is required for prediction.")
        else:
            inference_pipeline(args.image_path, args.model_path)
