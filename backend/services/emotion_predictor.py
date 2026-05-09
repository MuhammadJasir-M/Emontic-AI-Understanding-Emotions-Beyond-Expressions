# backend/services/emotion_predictor.py
# EfficientNetB0 inference with Test-Time Augmentation (TTA).
# The model is loaded once at import time — not per-request.

import os
import logging
import threading
import numpy as np
from PIL import Image

from config import MODEL_PATH, INPUT_SIZE, CONFIDENCE_THRESHOLD, EMOTION_LABELS, TTA_ENABLED

logger = logging.getLogger("emontic_ai")

# ── In-memory metrics (upgradeable to Prometheus later) ──────────────────────
_metrics_lock = threading.Lock()
_metrics = {
    "requests_total": 0,
    "no_face_total": 0,
    "uncertain_total": 0,
    "confidence_sum": 0.0,
    "latency_sum_ms": 0.0,
    "emotion_counts": {label: 0 for label in EMOTION_LABELS},
}


def record_request(
    emotion: str = None,
    confidence: float = None,
    latency_ms: float = None,
    no_face: bool = False,
):
    """Thread-safe metric recording."""
    with _metrics_lock:
        _metrics["requests_total"] += 1
        if no_face:
            _metrics["no_face_total"] += 1
            return
        if emotion == "Uncertain":
            _metrics["uncertain_total"] += 1
        if confidence is not None:
            _metrics["confidence_sum"] += confidence
        if latency_ms is not None:
            _metrics["latency_sum_ms"] += latency_ms
        if emotion and emotion in _metrics["emotion_counts"]:
            _metrics["emotion_counts"][emotion] += 1


def inference_metrics() -> dict:
    """Return current metrics snapshot."""
    with _metrics_lock:
        total = _metrics["requests_total"] or 1  # avoid division by zero
        successful = total - _metrics["no_face_total"]
        return {
            "requests_total": _metrics["requests_total"],
            "no_face_rate": round(_metrics["no_face_total"] / total, 4),
            "uncertain_rate": round(_metrics["uncertain_total"] / total, 4),
            "avg_confidence": round(
                _metrics["confidence_sum"] / max(successful, 1), 4
            ),
            "avg_latency_ms": round(
                _metrics["latency_sum_ms"] / max(successful, 1), 1
            ),
            "emotion_counts": dict(_metrics["emotion_counts"]),
        }


# ── Model Loading ────────────────────────────────────────────────────────────
_model = None
_infer_fn = None


def _run_inference(batch):
    """Execute one forward pass and always return a numpy array."""
    if _infer_fn is None:
        raise RuntimeError("Model is not initialized. Call _load_model() first.")
    pred = _infer_fn(batch)
    return pred.numpy()


def _load_model():
    """Lazy-load the TensorFlow SavedModel."""
    global _model, _infer_fn
    if _model is not None:
        return

    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        logger.error(
            f"Model not found at '{MODEL_PATH}'. "
            "Run the training pipeline and export the SavedModel first."
        )
        raise FileNotFoundError(
            f"Emotion model not found at '{MODEL_PATH}'. "
            "Please train and export the model before running inference."
        )

    logger.info(f"Loading SavedModel from '{MODEL_PATH}'...")
    loaded = tf.keras.models.load_model(MODEL_PATH)

    if callable(loaded):
        _model = loaded
        _infer_fn = lambda batch: _model(batch, training=False)
        logger.info("Model loaded as callable Keras model.")
        return

    # Keras 3 exports can deserialize to a SavedModel user object.
    # In that case, run inference through the serving signature.
    if hasattr(loaded, "signatures") and "serving_default" in loaded.signatures:
        _model = loaded
        serving_fn = loaded.signatures["serving_default"]

        def _savedmodel_infer(batch):
            outputs = serving_fn(tf.convert_to_tensor(batch))
            if isinstance(outputs, dict):
                return next(iter(outputs.values()))
            return outputs

        _infer_fn = _savedmodel_infer
        logger.info("Model loaded via SavedModel serving_default signature.")
        return

    raise TypeError(
        "Loaded model is neither callable nor exposes a serving_default signature."
    )


def get_model_status() -> str:
    """Check whether the model is available without loading it."""
    if _model is not None:
        return "loaded"
    if os.path.exists(MODEL_PATH):
        return "available (not yet loaded)"
    return "not found"


# ── EfficientNet Preprocessing ──────────────────────────────────────────────

def _efficientnet_preprocess(image_array):
    """Apply EfficientNet-specific preprocessing to a single image."""
    import tensorflow as tf
    return tf.keras.applications.efficientnet.preprocess_input(
        image_array.astype("float32")
    )


# ── Test-Time Augmentation (TTA) ────────────────────────────────────────────

def _predict_with_tta(face_rgb):
    """
    Predict with TTA: average predictions over original, flipped, and brightened.

    Args:
        face_rgb: uint8 RGB numpy array, 224×224, NOT yet preprocessed.
    Returns:
        Averaged probability array of shape (7,)
    """
    original = face_rgb.astype("float32")
    flipped = original[:, ::-1, :]  # horizontal flip
    brightened = np.clip(original * 1.15, 0, 255)  # +15% brightness

    raw_versions = [original, flipped, brightened]

    preds = []
    for img in raw_versions:
        preprocessed = _efficientnet_preprocess(img.copy())
        batch = np.expand_dims(preprocessed, axis=0)
        preds.append(_run_inference(batch))

    return np.mean(preds, axis=0)[0]


def _predict_single(face_rgb):
    """
    Predict without TTA (single forward pass).

    Args:
        face_rgb: uint8 RGB numpy array, 224×224, NOT yet preprocessed.
    Returns:
        Probability array of shape (7,)
    """
    preprocessed = _efficientnet_preprocess(face_rgb.copy())
    batch = np.expand_dims(preprocessed, axis=0)
    return _run_inference(batch)[0]


# ── Public API ───────────────────────────────────────────────────────────────

def predict_emotion(face_image) -> dict:
    """
    Takes a cropped face (numpy array RGB 224×224 or PIL Image),
    runs EfficientNetB0 inference with optional TTA,
    and returns the predicted emotion label, confidence, and all probabilities.

    Args:
        face_image: numpy array (uint8 RGB, 224×224) or PIL Image
    Returns:
        dict with keys: emotion, confidence, all_probs
    """
    # Ensure model is loaded
    _load_model()

    # Convert PIL Image to numpy if needed
    if isinstance(face_image, Image.Image):
        face_image = face_image.resize(INPUT_SIZE, Image.BILINEAR)
        face_rgb = np.array(face_image, dtype=np.uint8)
    else:
        face_rgb = face_image  # Already numpy uint8 RGB 224×224

    # ── Inference ────────────────────────────────────────────────────────
    if TTA_ENABLED:
        probs = _predict_with_tta(face_rgb)
    else:
        probs = _predict_single(face_rgb)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    # ── Build response ───────────────────────────────────────────────────
    all_probs = {
        label: round(float(p), 4) for label, p in zip(EMOTION_LABELS, probs)
    }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "emotion": "Uncertain",
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
        }

    return {
        "emotion": EMOTION_LABELS[top_idx],
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
    }
