# backend/config.py
# Central configuration for the Emontic AI backend.
# All tunable constants live here — never scatter magic numbers across services.

import os

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/emontic_ai")
INPUT_SIZE = (224, 224)  # Must match training INPUT_SHAPE (EfficientNetB0)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"

# ── Emotion Classes ──────────────────────────────────────────────────────────
# Order MUST match training label order exactly.
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── Upload Constraints ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MIN_FACE_BOX_PX = int(os.getenv("MIN_FACE_BOX_PX", "30"))  # Minimum face bounding box side in px
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

# ── CORS Origins ─────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")
