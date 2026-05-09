# backend/main.py
# FastAPI entry point for Emontic AI.

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# TensorFlow 2.21 requires the legacy Keras implementation for tf.keras APIs.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from config import CORS_ORIGINS, MODEL_PATH

logger = logging.getLogger("emontic_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info("Emontic AI starting up...")
    logger.info(f"Model path: {MODEL_PATH}")
    # Eagerly import predictor to load model at startup (fail-fast)
    try:
        from services.emotion_predictor import get_model_status
        status = get_model_status()
        logger.info(f"Model status: {status}")
    except Exception as e:
        logger.warning(f"Model not loaded — inference will fail until model is available: {e}")
    yield
    logger.info("Emontic AI shutting down.")


app = FastAPI(
    title="Emontic AI",
    description="Production 7-class facial emotion recognition API powered by EfficientNetB0, RetinaFace, and MediaPipe.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
from routers import predict  # noqa: E402

app.include_router(predict.router, prefix="/api")


# ── Health & Metrics ─────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    """Liveness probe — always returns ok if the process is running."""
    try:
        from services.emotion_predictor import get_model_status
        model_status = get_model_status()
    except Exception:
        model_status = "unavailable"
    return {"status": "ok", "model": model_status}


@app.get("/metrics", tags=["system"])
def metrics():
    """
    Basic observability endpoint.
    In production, replace with Prometheus counters / histograms.
    """
    from services.emotion_predictor import inference_metrics
    return inference_metrics()
