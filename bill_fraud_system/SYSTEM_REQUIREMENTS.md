# System Requirements: Bill & Card Fraud Detection System

This document outlines the hardware, operating system, and software requirements to build, run, and deploy the **Bill & Card Fraud Detection System**.

The application operates a dual-path detection pipeline using **EfficientNet-B0** for deep feature extraction, **Isolation Forest** & **Mahalanobis Distance** for anomaly scoring, **Error Level Analysis (ELA)** & **Laplacian Noise analysis** for forensic validation, and a **FastAPI backend** for API routing.

---

## 1. Hardware Requirements

Since the system leverages a highly lightweight convolutional neural network (**EfficientNet-B0** with ~5.3M parameters) and offline forensic algorithms, it has a very low hardware footprint and does **not** require a GPU.

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Dual-core 2.0 GHz (Intel, AMD, or Apple Silicon M-Series) | Quad-core+ (Intel Core i5/i7, AMD Ryzen 5, or Apple M-Series) |
| **Memory (RAM)** | **2 GB** (for running the web API and single-image scoring) | **4 GB or more** (allows concurrent requests and model training) |
| **Graphics (GPU)** | Not required (uses CPU for inference) | Optional (only needed if processing extreme batch inference) |
| **Disk Space** | **3 GB** of free storage | **5 GB or more** (to store cached model weights, uploads, and logs) |

---

## 2. Operating System Compatibility

The application is built on cross-platform frameworks and can run seamlessly on:
* **macOS**: 11.0 (Big Sur) or newer (native support for fast `sips` PDF conversion).
* **Linux**: Ubuntu 20.04+, Debian, CentOS, or any Docker-compatible distribution.
* **Windows**: Windows 10/11 (with Python 3.8+ or running via WSL2/Docker).

---

## 3. Software & System Dependencies

If running directly on your host system (without Docker), you need the following prerequisites installed:

### Core Language
* **Python 3.8 to 3.11** (Python 3.10 is recommended and tested in the Docker setup).

### System Utilities (Required for PDF processing)
* **For PDF-to-Image conversion** (`pdf2image` library):
  * **macOS**: Installed automatically via native `sips`, or optionally via Homebrew: `brew install poppler`
  * **Linux (Ubuntu/Debian)**: `sudo apt-get install poppler-utils`
  * **Windows**: Requires downloading Poppler for Windows and adding the `bin/` directory to the system PATH.

---

## 4. Python Package Dependencies

The main dependencies specified in the project's `requirements.txt` include:

* **Deep Learning**: `torch`, `torchvision` *(CPU-only versions are highly recommended to save ~1.5 GB of disk space)*.
* **Forensics & Image Processing**: `opencv-python-headless` (or standard `opencv-python`), `pillow`, `numpy`.
* **Machine Learning & Abstraction**: `scikit-learn`, `pandas`, `joblib`.
* **Web Server & API**: `fastapi`, `uvicorn`, `python-multipart`.
* **Testing & PDF Conversion**: `pdf2image`, `requests`.

---

## 5. Docker / Containerization Requirements (Alternative & Recommended Setup)

If you prefer running the system inside the pre-configured container using the provided `Dockerfile` and `docker-compose.yml`, the environment requires:

* **Docker Engine** v20.10 or newer.
* **Docker Compose** v2.0 or newer.
* **Internet Connection** (only during the first build to download the base image and cache the pre-trained EfficientNet-B0 weights, which are 20 MB).

---

## 6. Performance & Footprint Breakdown

* **Model Size on Disk**: ~20 MB (EfficientNet-B0) + under 2 MB per custom trained anomaly model (`.pkl` files).
* **Inference Speed**: ~2.5 to 3.0 seconds per document on a modern quad-core CPU.
* **RAM usage under load**: ~800 MB to 1.2 GB active RAM during heavy ELA/Deep Feature extraction.
