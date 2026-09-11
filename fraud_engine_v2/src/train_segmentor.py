import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import create_or_load_splits, DocumentTamperDataset, DEFAULT_DATASET_DIR
from src.tamper_segmentor import DocumentTamperSegmentor

class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss for extreme class imbalance in document tampering.
    """
    def __init__(self, alpha=0.85, gamma=2.0, dice_weight=1.0, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # 1. Focal Loss
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_factor * focal_factor * bce).mean()
        
        # 2. Dice Loss
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return focal_loss + self.dice_weight * dice_loss


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, 
                    precision="bf16", grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    use_autocast = device.type == "cuda" and precision in ["bf16", "fp16"]
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    for idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
                logits = outputs["tamper_logits"]
                loss = criterion(logits, masks) / grad_accum_steps
                
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(images)
            logits = outputs["tamper_logits"]
            loss = criterion(logits, masks) / grad_accum_steps
            loss.backward()

        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * grad_accum_steps
        if (idx + 1) % 25 == 0 or (idx + 1) == len(dataloader):
            print(f"  Step [{idx+1}/{len(dataloader)}] — Loss: {(loss.item() * grad_accum_steps):.4f}")
            
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, precision="bf16", threshold=0.50):
    """
    Computes comprehensive evaluation metrics:
    - Pixel IoU (Jaccard Index)
    - Pixel F1-Score (Dice Score)
    - Pixel Precision & Pixel Recall
    - Document-Level Tampering Detection Accuracy
    """
    model.eval()
    val_loss = 0.0
    
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    
    doc_tp = 0.0
    doc_fp = 0.0
    doc_fn = 0.0
    doc_tn = 0.0
    doc_total = 0
    
    use_autocast = device.type == "cuda" and precision in ["bf16", "fp16"]
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
                logits = outputs["tamper_logits"]
                loss = criterion(logits, masks)
        else:
            outputs = model(images)
            logits = outputs["tamper_logits"]
            loss = criterion(logits, masks)
            
        val_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Pixel-level True Positives, False Positives, False Negatives
        tp = (preds * masks).sum().item()
        fp = (preds * (1.0 - masks)).sum().item()
        fn = ((1.0 - preds) * masks).sum().item()
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        # Document-level classification check (document flagged if >= 20 tampered pixels found)
        for i in range(images.size(0)):
            has_gt_tamper = (masks[i].sum().item() > 10)
            has_pred_tamper = (preds[i].sum().item() > 20)
            if has_gt_tamper and has_pred_tamper:
                doc_tp += 1
            elif not has_gt_tamper and has_pred_tamper:
                doc_fp += 1
            elif has_gt_tamper and not has_pred_tamper:
                doc_fn += 1
            else:
                doc_tn += 1
            doc_total += 1
            
    avg_loss = val_loss / len(dataloader)
    
    # Calculate Pixel-Level Metrics
    pixel_precision = tp_total / max(1.0, tp_total + fp_total)
    pixel_recall = tp_total / max(1.0, tp_total + fn_total)
    pixel_f1 = (2.0 * pixel_precision * pixel_recall) / max(1e-6, pixel_precision + pixel_recall)
    pixel_iou = tp_total / max(1.0, tp_total + fp_total + fn_total)
    
    # Calculate Document-Level Metrics
    doc_precision = doc_tp / max(1.0, doc_tp + doc_fp)
    doc_recall = doc_tp / max(1.0, doc_tp + doc_fn)
    doc_f1 = (2.0 * doc_precision * doc_recall) / max(1e-6, doc_precision + doc_recall)
    doc_accuracy = ((doc_tp + doc_tn) / max(1, doc_total)) * 100.0
    
    metrics = {
        "val_loss": avg_loss,
        "pixel_iou": pixel_iou,
        "pixel_f1": pixel_f1,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "doc_precision": doc_precision,
        "doc_recall": doc_recall,
        "doc_f1": doc_f1,
        "doc_accuracy": doc_accuracy
    }
    return metrics


def setup_ada_optimizations():
    """
    Enables NVIDIA Ada Lovelace / Ampere hardware-specific performance optimizations:
    - TensorFloat-32 (TF32) for Matrix Multiplications and Convolutions
    - cuDNN benchmark auto-tuner
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Detected NVIDIA GPU: {gpu_name}")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("⚡ Enabled TensorFloat-32 (TF32) & cuDNN Auto-Tuning for Ada Tensor Cores")


def run_training(epochs=20, batch_size=16, lr=2e-4, dataset_dir=DEFAULT_DATASET_DIR, 
                 model_save_path="models/tamper_segmentor.pth", num_workers=8,
                 precision="bf16", grad_accum_steps=1, img_size=512, compile_model=False,
                 patience=5):
    
    # 1. Device Setup & Ada Optimizations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        setup_ada_optimizations()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
        precision = "fp32"
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU for training")
        precision = "fp32"

    print(f"Loading dataset splits from {dataset_dir}...")
    splits = create_or_load_splits(dataset_dir=dataset_dir)
    
    train_dataset = DocumentTamperDataset(splits["train"], target_size=(img_size, img_size), is_train=True)
    val_dataset = DocumentTamperDataset(splits["val"], target_size=(img_size, img_size), is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0 and device.type == "cuda"),
        prefetch_factor=2 if (num_workers > 0 and device.type == "cuda") else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0 and device.type == "cuda"),
        prefetch_factor=2 if (num_workers > 0 and device.type == "cuda") else None
    )
    
    train_genuine = sum(1 for it in train_dataset.items if it["is_genuine"])
    val_genuine   = sum(1 for it in val_dataset.items   if it["is_genuine"])
    print(f"Dataset: {len(train_dataset)} Train ({len(train_dataset)-train_genuine} tampered + {train_genuine} genuine) "
          f"| {len(val_dataset)} Val ({len(val_dataset)-val_genuine} tampered + {val_genuine} genuine) "
          f"| Resolution: {img_size}x{img_size}")
    print(f"Batch Size: {batch_size} (Effective: {batch_size * grad_accum_steps} with Grad Accum: {grad_accum_steps})")
    print(f"Precision: {precision.upper()} | Workers: {num_workers} | Early Stopping Patience: {patience} epochs")
    
    # 2. Model Initialization
    print("Initializing Document Tamper Segmentor...")
    model = DocumentTamperSegmentor(pretrained=True).to(device)
    
    if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        print("⚡ Compiling model with torch.compile for maximum Ada throughput...")
        model = torch.compile(model)
        
    criterion = FocalDiceLoss(alpha=0.85, gamma=2.0, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and precision == "fp16") else None
    
    save_dir = os.path.dirname(model_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    best_f1 = -1.0
    patience_counter = 0
    
    print("=" * 80)
    print("STARTING HIGH-PERFORMANCE TRAINING & EVALUATION LOOP")
    print("=" * 80)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        curr_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch}/{epochs}] — Learning Rate: {curr_lr:.6f}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, 
            precision=precision, grad_accum_steps=grad_accum_steps
        )
        
        # Evaluate with comprehensive metrics
        m = evaluate(model, val_loader, criterion, device, precision=precision)
        scheduler.step()
        
        elapsed = time.time() - t0
        print("-" * 80)
        print(f"📊 Epoch [{epoch}/{epochs}] Results ({elapsed:.1f}s):")
        print(f"   Train Loss      : {train_loss:.4f}  |  Val Loss: {m['val_loss']:.4f}")
        print(f"   [Pixel-Level]   F1 (Dice): {m['pixel_f1']*100:.2f}%  |  IoU: {m['pixel_iou']*100:.2f}%")
        print(f"                   Precision: {m['pixel_precision']*100:.2f}%  |  Recall: {m['pixel_recall']*100:.2f}%")
        print(f"   [Doc-Level]     Accuracy:  {m['doc_accuracy']:.2f}%  |  F1: {m['doc_f1']*100:.2f}%")
        print(f"                   Precision: {m['doc_precision']*100:.2f}%  |  Recall: {m['doc_recall']*100:.2f}%")
        print("-" * 80)
        
        # Checkpoint Best Model based on Pixel F1-score
        if m["pixel_f1"] > best_f1:
            best_f1 = m["pixel_f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": m,
                "img_size": img_size
            }, model_save_path)
            print(f"  ⭐ [BEST CHECKPOINT] Saved to {model_save_path} (F1: {m['pixel_f1']*100:.2f}%, IoU: {m['pixel_iou']*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  ℹ️ No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered! Model converged to optimal F1 score without overtraining.")
                break
            
    print("\n" + "=" * 80)
    print(f"🎉 Training Complete! Best Validation Pixel F1: {best_f1*100:.2f}%")
    print(f"Optimized Checkpoint saved to: {model_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance Document Tamper Segmentor Training & Evaluation")
    parser.add_argument("--epochs", type=int, default=20, help="Ideal number of epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model_path", type=str, default="models/tamper_segmentor.pth")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", 
                        help="Precision mode (bf16 recommended for Ada Lovelace)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--img_size", type=int, default=512, help="Image resolution for training")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0 torch.compile")
    args = parser.parse_args()
    
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dataset_dir=args.dataset_dir,
        model_save_path=args.model_path,
        precision=args.precision,
        grad_accum_steps=args.grad_accum,
        num_workers=args.num_workers,
        img_size=args.img_size,
        compile_model=args.compile
    )
