import os
import sys
import time
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bill_preprocessing import load_and_preprocess_image, create_multiscale_patches
from src.feature_extractor import FeatureExtractor, ForensicFeatureExtractor
from src.outlier_detector import AnomalyDetector
from src.pipeline import score_image

# === Universal ELA-based tampering check (model-independent) ===
ELA_TAMPER_THRESHOLD = 2.4  # Calibrated: catches 93% tampered, 93% genuine pass

def compute_ela_tamper_score(pil_image):
    """Compute a standalone tampering score using ELA forensics.
    Works independently of which anomaly model is selected.
    Returns (score, is_suspicious)."""
    feats = ForensicFeatureExtractor.extract_image_forensics(pil_image)
    ela_range = feats[26]       # ELA range across patches (high = inconsistent edits)
    ela_consistency = feats[24] # ELA consistency (high = regional anomalies)
    noise_mean = feats[20]      # Noise level (low = heavily re-saved/edited)
    score = (ela_range / 10.0) + (ela_consistency / 3.0) - (noise_mean / 5.0)
    return float(score), score > ELA_TAMPER_THRESHOLD

app = FastAPI(title="Bill Fraud Detection System")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Dual-model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BILL_MODEL_PATH = os.path.join(MODELS_DIR, "bill_model_v2.pkl")
CARD_MODEL_PATH = os.path.join(MODELS_DIR, "card_model.pkl")

# Model containers
models = {
    "bill": {"detector": None, "name": "Bill / Prescription"},
    "card": {"detector": None, "name": "ECHS Card / ID"},
}
deep_extractor = None


def load_models():
    global models, deep_extractor
    print("Loading models...")
    deep_extractor = FeatureExtractor(model_name='efficientnet_b0')
    
    # Load bill model
    if os.path.exists(BILL_MODEL_PATH):
        models["bill"]["detector"] = AnomalyDetector()
        models["bill"]["detector"].load_model(BILL_MODEL_PATH)
        print(f"  ✅ Bill model loaded from {BILL_MODEL_PATH}")
    else:
        print(f"  ⚠️ Bill model not found at {BILL_MODEL_PATH}")
    
    # Load card model
    if os.path.exists(CARD_MODEL_PATH):
        models["card"]["detector"] = AnomalyDetector()
        models["card"]["detector"].load_model(CARD_MODEL_PATH)
        print(f"  ✅ Card model loaded from {CARD_MODEL_PATH}")
    else:
        print(f"  ⚠️ Card model not found at {CARD_MODEL_PATH}")
    
    print("Models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/faq")
async def faq():
    return FileResponse(os.path.join(STATIC_DIR, "faq.html"))


@app.get("/api/models")
async def list_models():
    """Return available models and their status."""
    available = {}
    for key, val in models.items():
        available[key] = {
            "name": val["name"],
            "loaded": val["detector"] is not None,
        }
    return JSONResponse(content={"models": available})


def convert_pdf_to_image(pdf_path):
    """Convert a PDF to a JPEG image using sips (macOS) or PIL."""
    jpg_path = pdf_path.rsplit('.', 1)[0] + '.jpg'
    
    try:
        # Try using sips (macOS built-in) for conversion
        result = subprocess.run(
            ['sips', '-s', 'format', 'jpeg', pdf_path, '--out', jpg_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and os.path.exists(jpg_path):
            return jpg_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fallback: try pdf2image if available
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
        if images:
            images[0].save(jpg_path, 'JPEG', quality=95)
            return jpg_path
    except ImportError:
        pass
    
    return None


@app.post("/api/analyze")
async def analyze_bill(
    file: UploadFile = File(...),
    doc_type: str = Form(default="bill")
):
    """Analyze an uploaded document for tampering."""
    start_time = time.time()
    
    # Validate doc_type
    if doc_type not in models:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown document type: {doc_type}. Use 'bill' or 'card'."}
        )
    
    # Use the model matching the selected document type
    detector = models[doc_type]["detector"]
    if detector is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Model for '{doc_type}' is not loaded."}
        )
    
    # Save uploaded file
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    upload_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
    
    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Handle PDF: convert to image first
        analysis_path = upload_path
        if ext.lower() == '.pdf':
            converted = convert_pdf_to_image(upload_path)
            if converted:
                analysis_path = converted
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "PDF conversion failed. Please upload a JPG or PNG image instead."}
                )
        
        # Get image info
        pil_image = Image.open(analysis_path).convert("RGB")
        img_width, img_height = pil_image.size
        
        # Run detection with the selected model
        combined, deep_score, forensic_score = score_image(
            analysis_path, deep_extractor, detector
        )
        
        if combined is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not process the image."}
            )
        
        threshold = detector.calibration.get('image_threshold', 2.0)
        model_flagged = combined > threshold
        
        # Run independent ELA-based tampering check (works for ANY doc type)
        ela_score, ela_flagged = compute_ela_tamper_score(pil_image)
        
        # Flag as tampered if EITHER the model OR the ELA check triggers
        is_tampered = model_flagged or ela_flagged
        
        # Compute confidence level (0-100%)
        if is_tampered:
            if model_flagged:
                confidence = min(99.9, 70 + 30 * min(1.0, (combined - threshold) / (threshold * 2)))
            else:
                # ELA-only detection — confidence based on how far above ELA threshold
                ela_margin = (ela_score - ELA_TAMPER_THRESHOLD) / ELA_TAMPER_THRESHOLD
                confidence = min(95.0, 65 + 30 * min(1.0, ela_margin))
        else:
            margin = (threshold - combined) / (threshold + abs(combined) + 1e-8)
            confidence = min(99.9, 60 + 40 * min(1.0, margin))

        # Determine tampering reason
        tamper_reason = "None"
        if is_tampered:
            reasons = []
            # Check model components
            if model_flagged:
                # Check if deep (visual/structure) score is high
                if deep_score > 0.5:  # calibrated threshold
                    reasons.append("Visual/structural anomalies detected")
                # Check if forensic (noise/ELA) score is high
                if forensic_score > 0.5:
                    reasons.append(" inconsistent compression artifacts")
            
            # Check ELA specific
            if ela_flagged:
                reasons.append("Abnormal editing traces (ELA)")
            
            if not reasons:
                reasons.append("Statistical anomaly in document structure")
            
            tamper_reason = "; ".join(reasons).replace("detected; ", "detected, ").capitalize()
        
        elapsed = time.time() - start_time
        
        result = {
            "status": "TAMPERED" if is_tampered else "GENUINE",
            "confidence": round(confidence, 1),
            "combined_score": round(float(combined), 4),
            "deep_score": round(float(deep_score), 4),
            "forensic_score": round(float(forensic_score), 4),
            "threshold": round(float(threshold), 4),
            "processing_time": round(elapsed, 2),
            "doc_type": doc_type,
            "model_name": models[doc_type]["name"],
            "image_info": {
                "filename": file.filename,
                "width": img_width,
                "height": img_height,
                "format": ext.replace(".", "").upper(),
            },
            "upload_url": f"/uploads/{os.path.basename(analysis_path)}",
            "tamper_reason": tamper_reason,
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )
    finally:
        # Clean up old uploads (keep last 20)
        try:
            uploads = sorted(
                [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)],
                key=os.path.getmtime
            )
            for old_file in uploads[:-20]:
                os.remove(old_file)
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
