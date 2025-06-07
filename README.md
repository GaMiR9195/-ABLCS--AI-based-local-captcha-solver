# Qwen2.5-VL-7B Vision (Captcha Solver) API without enchanting images

A lightweight implementation of Qwen2.5-VL-7B-Instruct multimodal model for simple captcha solving tasks (EDUCATION ONLY).

## Overview

This project provides a simple HTTP API wrapper around the Qwen2.5-VL-7B-Instruct vision-language model, allowing you to make image -> text requests.

> **⚠️ Educational Purpose Only** - This implementation is intended for learning and experimentation. Not recommended for production use.

## Requirements

- Git (https://git-scm.com/downloads/win)
- CUDA TOOKIT (https://developer.nvidia.com/cuda-downloads)
- Python 3.10.11 (optional)
- CUDA-capable GPU (10GB+ VRAM recommended)
- PyTorch 2.0.0+

## Installation

```bash
# Clone repository
git clone https://github.com/GaMiR9195/AI-based-local-captcha-solver/tree/main

# Open console
python -m venv env
# Go to env\Scripts\activate.bat and activate venv
```
### Edit your requirements.txt (extra index url)
```bash
--extra-index-url SCROLL DOWN TO SET YOUR LINK CORRECTLY
torch>=2.3.0
torchvision>=0.18.0
transformers @ git+https://github.com/huggingface/transformers.git@f3f6c86582611976e72be054675e2bf0abb5f775
accelerate>=0.33.0
bitsandbytes>=0.43.0
huggingface-hub>=0.24.0
fastapi>=0.111.0
uvicorn>=0.30.0
pillow>=10.4.0
python-multipart>=0.0.9
pydantic>=2.8.0
qwen-vl-utils>=0.0.8
psutil>=7.0.0
opencv-python>=4.11.0
scikit-image>=0.25.2
PyWavelets>=1.8.0
matplotlib>=3.10.3
opencv-contrib-python>=4.11.0.86
opencv-python-headless>=4.11.0.86
```
### Select your extra index url and put in top of requirements.txt
```bash
--extra-index-url https://download.pytorch.org/whl/cu128/ # CUDA 12.8 NVIDIA 30+ SERIES
--extra-index-url https://download.pytorch.org/whl/cu126/ # CUDA 12.1 NVIDIA 20 / 16 SERIES
--extra-index-url https://download.pytorch.org/whl/cu118/ # CUDA 11.8 NVIDIA 10 SERIES
--extra-index-url https://download.pytorch.org/whl/rocm6.2.4 # AMD GPU ROCm 6.2
--extra-index-url https://download.pytorch.org/whl/cpu # CPU ONLY TORCH
```
### Installation continue...
```bash
pip install -r requirements.txt

# Download model (and put in Qwen folder)
# https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main
```

## Quick Start

```bash
# Start API server
# OPEN env\Scripts\activate.bat 
python AI.py 

# Test with sample request
curl -X POST "http://localhost:8000/analyze" -F "file=@test.png"
```
