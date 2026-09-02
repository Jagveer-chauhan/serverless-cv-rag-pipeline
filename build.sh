#!/usr/bin/env bash
# Render Linux Build Script for Serverless CV Parsing and RAG Pipeline
# Adheres to warm-path SLA <= 5.0s by pre-installing native OCR and Poppler binaries

set -o errexit

echo "=========================================================="
echo "==> [Phase 1] Starting Render Linux Environment Setup..."
echo "=========================================================="

echo "==> Updating apt-get package lists..."
apt-get update -y

echo "==> Installing system dependencies: tesseract-ocr, poppler-utils, libgl1-mesa-glx..."
apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing backend Python dependencies from backend/requirements.txt..."
pip install -r backend/requirements.txt

echo "=========================================================="
echo "==> Render Linux Environment build completed successfully."
echo "=========================================================="
