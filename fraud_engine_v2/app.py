import os
import sys
import time
import subprocess
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fraud_engine import DocumentFraudEngine

app = FastAPI(
    title="Fraud Engine v2 API",
    description="Multi-Tiered Document & Image Fraud Detection Engine with Pixel-Level Localization",
    version="2.0.0"
)

# Enable CORS for external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Mount static folders
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize Engine
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, "tamper_segmentor.pth")
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    print("🚀 Initializing Fraud Engine v2...")
    engine = DocumentFraudEngine(model_path=MODEL_WEIGHTS_PATH)
    print("✅ Fraud Engine v2 ready for requests!")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
async def health_check():
    return JSONResponse(content={
        "status": "online",
        "engine_version": "2.0.0",
        "device": str(engine.device) if engine else "uninitialized",
        "model_loaded": engine.model_loaded if engine else False
    })


def convert_pdf_to_image(pdf_path):
    """Converts first page of PDF to JPG using sips (macOS) or pdf2image."""
    jpg_path = pdf_path.rsplit(".", 1)[0] + ".jpg"
    try:
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


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...)
):
    """
    Analyzes an uploaded document image or PDF for fraudulent tampering.
    Returns pixel-level localization bounding boxes, confidence score, and forensic breakdowns.
    """
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Fraud engine is not initialized."})
        
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    timestamp = int(time.time() * 1000)
    upload_filename = f"upload_{timestamp}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    
    try:
        # Save file to uploads
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
            
        analysis_path = upload_path
        if ext == ".pdf":
            converted = convert_pdf_to_image(upload_path)
            if converted:
                analysis_path = converted
            else:
                return JSONResponse(status_code=400, content={"error": "Failed to convert PDF document to image."})
                
        # Run Fraud Engine Analysis
        results = engine.analyze_document(analysis_path, output_dir=UPLOAD_DIR)
        
        # Attach upload URL
        results["original_url"] = f"/uploads/{os.path.basename(analysis_path)}"
        results["filename"] = file.filename
        
        return JSONResponse(content=results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Document analysis failed: {str(e)}"})
    finally:
        # Cleanup old uploads (keep latest 30)
        try:
            files = sorted(
                [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)],
                key=os.path.getmtime
            )
            for old_file in files[:-30]:
                os.remove(old_file)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
