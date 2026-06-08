# backend/main.py
import os
import sys
import site

# ==============================================================================
# PERMANENT HERMETIC CUDA FIX (TensorFlow 2.21.0+)
# Must execute BEFORE any keras, tensorflow, or fastapi imports!
# ==============================================================================
def _inject_cuda_paths():
    paths_to_add = ["/usr/lib/wsl/lib"]
    try:
        site_packages = site.getsitepackages()[0]
        nvidia_libs = [
            'cudnn', 'cublas', 'cuda_runtime', 'cusolver', 
            'cusparse', 'curand', 'nvjitlink', 'cufft'
        ]
        for lib in nvidia_libs:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_path):
                paths_to_add.append(lib_path)
    except Exception:
        pass

    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = ":".join(paths_to_add)
    
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

_inject_cuda_paths()
# ==============================================================================

# CRITICAL: Must be set before ANY TensorFlow/Keras imports.
# RetinaFace uses tf.shape() on KerasTensors which is incompatible with Keras 3.
# This forces tf.keras to use the legacy tf_keras (Keras 2) shim for compatibility.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from config import CORS_ORIGINS, MODEL_PATH, UPLOAD_DIR

logger = logging.getLogger("emontic_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info("Emontic AI starting up...")
    
    # Ensure uploads directory exists
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # TRUE WARMUP: Force load model graph fully into GPU VRAM at boot time
    try:
        from services.emotion_predictor import _load_model, get_model_status
        logger.info(f"Pre-loading position-aware weights from: {MODEL_PATH}")
        _load_model() # This loads the network layers completely into RAM
        logger.info(f"Initialization complete. Model status: {get_model_status()}")
    except Exception as e:
        logger.critical(f"FATAL: Model failed to load at startup sequence: {e}", exc_info=True)

    # Initialize MySQL database
    try:
        from database import init_db
        init_db()
        logger.info("Database connection successfully verified.")
    except Exception as e:
        logger.warning(f"Database unavailable — history logs will be cached locally: {e}")

    yield
    logger.info("Emontic AI shutting down.")


app = FastAPI(
    title="Emontic AI",
    description="Production 7-class facial emotion recognition API powered by EfficientNetV2-S with position-aware multi-head self-attention.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Deferred imports to ensure environment properties initialize first
from routers import predict, history, live_predict  # noqa: E402

app.include_router(predict.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(live_predict.router, prefix="/api")


@app.get("/health", tags=["system"])
def health():
    try:
        from services.emotion_predictor import get_model_status
        model_status = get_model_status()
    except Exception:
        model_status = "unavailable"
    return {"status": "ok", "model": model_status}


@app.get("/metrics", tags=["system"])
def metrics():
    from services.emotion_predictor import inference_metrics
    return inference_metrics()