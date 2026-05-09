<div align="center">

# Emontic AI

**AI-Powered Facial Emotion Recognition Platform**

_Real-time face detection, alignment, and 7-class emotion inference with a production-oriented training and serving pipeline._

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-FF6F00.svg)](https://tensorflow.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg)](https://vitejs.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-system-architecture) · [🧠 Training Strategy](#-training-strategy) · [🔌 API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem We Solve](#-problem-we-solve)
- [Core Strategy](#-core-strategy)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Training Strategy](#-training-strategy)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Development Notes](#-development-notes)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Emontic AI is an end-to-end emotion AI system built for practical use, not just notebook demos.

It combines:

- A **React + Vite frontend** for image upload and visual prediction results
- A **FastAPI backend** for robust image validation, face detection, and emotion inference
- A **two-stage training pipeline** using AffectNet and RAF-DB for improved generalization

Predicted classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## ❗ Problem We Solve

Most emotion-recognition prototypes fail during real usage because they are trained once and served without production safeguards.

Common issues:

- Poor face localization under varied lighting and pose
- Weak cross-dataset transfer
- No confidence distribution or observability for downstream apps
- Fragile integration between frontend and inference service

Emontic AI addresses these through a modular architecture and staged model development strategy.

---

## 🧭 Core Strategy

The project uses curriculum-style optimization across datasets:

1. **Stage A (AffectNet)**: Learn broad emotional representations.
2. **Stage B (RAF-DB)**: Fine-tune for sharper class boundaries.
3. **Serving pipeline**: Detection, preprocessing, and classification are separated for maintainability.

Why this works:

- AffectNet contributes data diversity and scale.
- RAF-DB improves class discrimination during late-stage adaptation.
- Decoupled services simplify debugging and future upgrades.

---

## ✨ Key Features

### 🧠 ML Pipeline

- EfficientNetB0-based classifier
- Progressive layer unfreezing
- AdamW + weight decay
- Label smoothing
- Class weighting
- Cosine LR schedule in fine-tuning
- Early stopping + checkpoints + TensorBoard + CSV logs
- Reproducibility controls via deterministic seeds

### 🖼️ Inference Pipeline

- RetinaFace for robust face detection
- MediaPipe-assisted face alignment and preprocessing
- EXIF-safe image loading and validation
- Confidence output with full probability distribution
- Optional TTA and configurable confidence threshold

### ⚙️ Product Engineering

- FastAPI lifecycle hooks with startup model checks
- Health and metrics endpoints
- Latency capture per inference request
- Frontend-ready API contract for direct integration

---

## 🏗️ System Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                    │
│                Upload image + visualize results               │
└───────────────────────────────┬───────────────────────────────┘
                                │ HTTP (multipart)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  Backend API (FastAPI, :8000)                 │
│                                                               │
│  /api/predict                                                 │
│  ├─ validate_and_load_image()                                 │
│  ├─ detect_and_crop_face()  [RetinaFace]                      │
│  └─ predict_emotion()       [EfficientNetB0 model]            │
│                                                               │
│  /health    /metrics                                          │
└───────────────────────────────┬───────────────────────────────┘
                                │ model artifacts
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               SavedModel / Keras checkpoints                  │
│      Produced by staged training (AffectNet -> RAF-DB)        │
└───────────────────────────────────────────────────────────────┘
```

### Data Flow

```text
User uploads image
   │
   ├─► Backend validates type/size and decodes image
   ├─► RetinaFace detects and crops primary face
   ├─► Preprocessing aligns and normalizes input
   ├─► Model returns class probabilities
   └─► API responds with emotion + confidence + bbox + latency
```

---

## 🛠️ Tech Stack

### Frontend

- React 18
- Vite 5
- Tailwind CSS
- Framer Motion
- Axios

### Backend

- FastAPI + Uvicorn
- TensorFlow / Keras
- RetinaFace
- MediaPipe
- OpenCV + Pillow

### Training

- TensorFlow (GPU-capable setup via WSL/Linux script)
- NumPy + scikit-learn
- TensorBoard

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm 9+

### 1) Start Backend

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies and run API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:5173 by default.

Optional frontend env override:

```env
VITE_API_URL=http://localhost:8000/api
```

---

## 🧠 Training Strategy

Training entry point: training/train.py

### Stages

- **Stage A (AffectNet Pretraining)**
  - Phase 1: frozen backbone
  - Phase 2: unfreeze top layers for adaptation
- **Stage B (RAF-DB Fine-tuning)**
  - initialize from Stage A checkpoint
  - unfreeze more layers with cosine decay learning rate

### Run Commands

```bash
cd training
python train.py --stage a
python train.py --stage b
python train.py --stage both
python train.py --stage b --checkpoint checkpoints/stageA_final.keras
```

Windows launcher for WSL workflow:

```powershell
cd training
.\run_training.ps1 -Stage both
```

Training outputs:

- training/checkpoints/
- training/logs/

---

## 🔌 API Reference

### Base URL

```text
http://localhost:8000
```

### Endpoints

- GET /health
- GET /metrics
- POST /api/predict

### Predict Endpoint

Request:

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/face.jpg"
```

Example response:

```json
{
  "emotion": "Happy",
  "confidence": 0.9412,
  "all_probs": {
    "Angry": 0.004,
    "Disgust": 0.003,
    "Fear": 0.007,
    "Happy": 0.941,
    "Neutral": 0.022,
    "Sad": 0.009,
    "Surprise": 0.014
  },
  "bbox": [120, 80, 340, 300],
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "latency_ms": 86.3
}
```

Swagger docs when backend is running:

- http://localhost:8000/docs

---

## 🐳 Deployment

Backend Docker build:

```bash
cd backend
docker build -t emontic-backend .
docker run --rm -p 8000:8000 emontic-backend
```

---

## 🧪 Development Notes

### Repository Layout

```text
emontic-ai/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routers/predict.py
│   ├── services/
│   └── models/emontic_ai/
├── frontend/
│   ├── src/components/
│   ├── src/hooks/
│   └── src/utils/api.js
├── training/
│   ├── train.py
│   ├── dataset.py
│   ├── model.py
│   ├── evaluate.py
│   ├── export.py
│   ├── run_training.ps1
│   └── train_wsl.sh
├── data/
└── saved_model/
```

---

## 🐛 Troubleshooting

| Issue                                   | Likely Cause                         | Fix                                            |
| --------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| Model unavailable at startup            | MODEL_PATH is wrong or files missing | Verify backend/config.py and saved model files |
| 422 No face detected                    | Image quality/angle too poor         | Use clear frontal face with better lighting    |
| Frontend cannot call API                | CORS or API URL mismatch             | Check CORS_ORIGINS and VITE_API_URL            |
| First request is slow                   | Model cold start                     | Warm up once after service start               |
| Training fails on Windows CPU/GPU setup | Local env mismatch                   | Use training/run_training.ps1 with WSL         |

---

## 🗺️ Roadmap

- [ ] Video stream inference endpoint
- [ ] Benchmark suite for per-device latency and throughput
- [ ] Model card and evaluation protocol documentation
- [ ] CI for lint, tests, and API smoke checks
- [ ] Full-stack container orchestration

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch.
3. Keep changes focused and well documented.
4. Validate backend and frontend locally.
5. Open a pull request with summary and test evidence.

---

## 📄 License

No LICENSE file is currently present in this repository.
Add one before public distribution to clarify usage rights.

---

<div align="center">

Built for practical, explainable, and scalable emotion AI workflows.

</div>

</div>
