import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "dataset_64bffd17")
)

def scan_and_pair_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    """
    Scans the dataset directory, pairing each tampered page with its ground-truth
    binary mask, corresponding original image, and OCR metadata.
    """
    forgeries_dir = os.path.join(dataset_dir, "forgeries")
    originals_dir = os.path.join(dataset_dir, "originals")
    
    if not os.path.exists(forgeries_dir) or not os.path.exists(originals_dir):
        raise FileNotFoundError(f"Dataset folders not found at {dataset_dir}")
        
    forgery_subdirs = sorted([
        d for d in os.listdir(forgeries_dir) 
        if os.path.isdir(os.path.join(forgeries_dir, d))
    ])
    
    paired_samples = []
    
    for sub in forgery_subdirs:
        forg_path = os.path.join(forgeries_dir, sub)
        img_dir = os.path.join(forg_path, "images")
        mask_dir = os.path.join(forg_path, "masks")
        label_dir = os.path.join(forg_path, "labels")
        
        # Locate raw tampered image (exclude _vis.jpg)
        tampered_imgs = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(".jpg") and not f.endswith("_vis.jpg")
        ] if os.path.exists(img_dir) else []
        
        # Locate binary mask
        masks = [
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.endswith(".png") and "mask" in f
        ] if os.path.exists(mask_dir) else []
        
        # Locate OCR custom JSON
        ocr_jsons = [
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith("_custom.json")
        ] if os.path.exists(label_dir) else []
        
        if not tampered_imgs or not masks:
            continue
            
        t_img = tampered_imgs[0]
        m_img = masks[0]
        ocr_file = ocr_jsons[0] if ocr_jsons else None
        
        # Determine original doc and page
        # Example sub: '30061101_S_LAB_REPORTS_DS_page_3_tampered'
        # Base name: '30061101_S_LAB_REPORTS_DS'
        raw_name = sub.replace("_tampered", "")
        parts = raw_name.rsplit("_page_", 1)
        doc_name = parts[0]
        page_num = parts[1] if len(parts) > 1 else "1"
        
        # Find corresponding original file in originals
        orig_img_path = None
        orig_doc_candidates = [
            os.path.join(originals_dir, doc_name),
            os.path.join(originals_dir, f"Copy of {doc_name}")
        ]
        
        for cand in orig_doc_candidates:
            if os.path.exists(cand):
                cand_img_dir = os.path.join(cand, "images")
                cand_img = os.path.join(cand_img_dir, f"{doc_name}_page_{page_num}.jpg")
                if os.path.exists(cand_img):
                    orig_img_path = cand_img
                    break
                cand_copy_img = os.path.join(cand_img_dir, f"Copy of {doc_name}_page_{page_num}.jpg")
                if os.path.exists(cand_copy_img):
                    orig_img_path = cand_copy_img
                    break
        
        paired_samples.append({
            "id": sub,
            "tampered_image": t_img,
            "mask_image": m_img,
            "original_image": orig_img_path,
            "ocr_json": ocr_file,
            "doc_name": doc_name,
            "page": page_num
        })
        
    return paired_samples


def create_or_load_splits(dataset_dir=DEFAULT_DATASET_DIR, split_file="dataset_splits.json", 
                          train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, force_rescan=False):
    """
    Creates deterministic document-level splits (train/val/test) so pages from
    the same document do not leak across sets.
    """
    split_path = os.path.join(os.path.dirname(__file__), "..", split_file)
    if os.path.exists(split_path) and not force_rescan:
        try:
            with open(split_path, "r") as f:
                cached_data = json.load(f)
            # Verify that the paths in the cache actually exist on this system
            if cached_data.get("train") and len(cached_data["train"]) > 0:
                first_img = cached_data["train"][0].get("tampered_image", "")
                if os.path.exists(first_img):
                    return cached_data
                else:
                    print(f"ℹ️ Cached split paths from another machine ({first_img}). Auto re-scanning on current machine...")
        except Exception:
            pass
            
    pairs = scan_and_pair_dataset(dataset_dir)
    
    # Group by base document name to prevent data leakage
    doc_groups = {}
    for p in pairs:
        doc = p["doc_name"]
        if doc not in doc_groups:
            doc_groups[doc] = []
        doc_groups[doc].append(p)
        
    docs = sorted(list(doc_groups.keys()))
    random.seed(seed)
    random.shuffle(docs)
    
    n_docs = len(docs)
    n_train = int(n_docs * train_ratio)
    n_val = int(n_docs * val_ratio)
    
    train_docs = set(docs[:n_train])
    val_docs = set(docs[n_train:n_train + n_val])
    test_docs = set(docs[n_train + n_val:])
    
    splits = {
        "train": [p for p in pairs if p["doc_name"] in train_docs],
        "val": [p for p in pairs if p["doc_name"] in val_docs],
        "test": [p for p in pairs if p["doc_name"] in test_docs],
        "stats": {
            "total_pairs": len(pairs),
            "train_pairs": sum(1 for p in pairs if p["doc_name"] in train_docs),
            "val_pairs": sum(1 for p in pairs if p["doc_name"] in val_docs),
            "test_pairs": sum(1 for p in pairs if p["doc_name"] in test_docs),
            "unique_docs": n_docs
        }
    }
    
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
        
    return splits


class DocumentTamperDataset(Dataset):
    """
    PyTorch Dataset for paired Document Tamper Segmentation.
    Loads RGB image and Ground-Truth Binary Mask with synchronized data augmentations.
    
    Includes genuine (original) images as negative examples with all-zero masks
    so the model learns what clean, unmanipulated documents look like.
    """
    def __init__(self, samples, target_size=(512, 512), is_train=False,
                 include_genuine=True, genuine_ratio=1.0):
        self.target_size = target_size
        self.is_train = is_train
        
        # Build list of (image_path, mask_path_or_None, is_genuine)
        self.items = []
        
        for item in samples:
            # Tampered sample (positive example)
            self.items.append({
                "image_path": item["tampered_image"],
                "mask_path": item["mask_image"],
                "is_genuine": False,
                "id": item["id"]
            })
            
            # Genuine sample (negative example — all-zero mask)
            if include_genuine and item.get("original_image") and os.path.exists(item["original_image"]):
                if not is_train or random.random() < genuine_ratio:
                    self.items.append({
                        "image_path": item["original_image"],
                        "mask_path": None,   # zero mask — this is a clean document
                        "is_genuine": True,
                        "id": item["id"] + "_genuine"
                    })
        
        if is_train:
            random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        # Load mask or create all-zero mask for genuine images
        if item["mask_path"] is not None:
            mask = Image.open(item["mask_path"]).convert("L")
        else:
            # Genuine image: ground truth is all zeros (no tampering)
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8), mode="L")
        
        # Resize both to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # Synchronized Data Augmentations during Training
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # Random slight rotation (-3 to 3 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-3, 3)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
                
            # Color jitter on RGB image only
            if random.random() > 0.5:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                
        # Convert to Tensor
        img_tensor = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        
        # Normalize image using standard ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Convert binary mask to [1, H, W] where tampered=1.0, background=0.0
        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > 128).astype(np.float32)).unsqueeze(0)
        
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "id": item["id"],
            "image_path": item["image_path"],
            "is_genuine": item["is_genuine"]
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Verification & Splitter")
    parser.add_argument("--verify", action="store_true", help="Scan dataset and print statistics")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    
    print(f"Scanning dataset at {args.dataset_dir}...")
    splits = create_or_load_splits(dataset_dir=args.dataset_dir)
    stats = splits["stats"]
    print("=" * 60)
    print("DATASET PREPARATION & SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Paired Tampered Samples : {stats['total_pairs']}")
    print(f"Unique Base Documents         : {stats['unique_docs']}")
    print(f"Training Set Pairs            : {stats['train_pairs']} (70%)")
    print(f"Validation Set Pairs          : {stats['val_pairs']} (15%)")
    print(f"Test Set Pairs                : {stats['test_pairs']} (15%)")
    print("=" * 60)
    print("✅ Dataset verified and splits successfully initialized!")
