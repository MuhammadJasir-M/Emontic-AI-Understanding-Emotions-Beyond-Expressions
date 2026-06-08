# backend/config.py
# Central configuration for the Emontic AI backend.
# All tunable constants live here — never scatter magic numbers across services.

import os

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/emontic_ai.onnx")
INPUT_SIZE = (224, 224)  # Matches deployed ONNX serving signature (EfficientNetV2S)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"

# ── Emotion Classes ──────────────────────────────────────────────────────────
# Order MUST match training label order exactly.
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── Database Configuration ───────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "emontic_ai_db")

# ── Upload / Storage ────────────────────────────────────────────────────────
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# ── Upload Constraints ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MIN_FACE_BOX_PX = int(os.getenv("MIN_FACE_BOX_PX", "30"))  # Minimum face bounding box side in px
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

# ── CORS Origins ─────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")
