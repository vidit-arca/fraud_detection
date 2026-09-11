#!/usr/bin/env python3
"""
Fraud Engine v2 — CLI Prediction & Evaluation Entrypoint
Delegates execution to src/predict.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import main

if __name__ == "__main__":
    main()
