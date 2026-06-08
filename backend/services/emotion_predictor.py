# backend/services/emotion_predictor.py
# Position-Aware Inference engine featuring Vectorized TTA execution blocks.
# Uses tf.saved_model.load() for Keras 3 compatibility with SavedModel format.

import os
import logging
import threading
import numpy as np
from PIL import Image
import cv2

from config import MODEL_PATH, INPUT_SIZE, CONFIDENCE_THRESHOLD, EMOTION_LABELS, TTA_ENABLED

logger = logging.getLogger("emontic_ai")

# ── In-memory metrics ────────────────────────────────────────────────────────
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
        total = _metrics["requests_total"] or 1
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
_session = None


def _run_inference(batch):
    """Execute forward passes on stacked matrices simultaneously using ONNX."""
    if _session is None:
        raise RuntimeError("Model is not initialized. Call _load_model() first.")
    
    input_name = _session.get_inputs()[0].name
    tensor_input = batch.astype(np.float32)
    outputs = _session.run(None, {input_name: tensor_input})
    return outputs[0]


def _load_model():
    """
    Lazy-load the ONNX model using onnxruntime.
    Prioritizes CUDAExecutionProvider for hardware acceleration.
    """
    global _session
    if _session is not None:
        return

    import onnxruntime as ort

    if not os.path.exists(MODEL_PATH):
        logger.error(f"ONNX Model not found at '{MODEL_PATH}'.")
        raise FileNotFoundError(f"Emotion model not found at '{MODEL_PATH}'.")

    logger.info(f"Loading ONNX Model from '{MODEL_PATH}'...")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        _session = ort.InferenceSession(MODEL_PATH, providers=providers)
        logger.info(f"ONNX Model freshly loaded into RAM. Active providers: {_session.get_providers()}")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise


def get_model_status() -> str:
    """Check whether the model is available without loading it."""
    if _session is not None:
        return "loaded"
    if os.path.exists(MODEL_PATH):
        return "available (not yet loaded)"
    return "not found"


# ── EfficientNetV2S Preprocessing ───────────────────────────────────────────

def _v2s_preprocess(image_array):
    """
    Preprocess for EfficientNetV2S inference.

    The model was trained with include_preprocessing=True inside the EfficientNetV2S
    backbone, which means the model itself handles internal rescaling. The input
    contract is: float32 pixels in [0.0, 255.0] range, resized to INPUT_SIZE.
    No external normalization is needed.
    """
    img = image_array.astype("float32")
    # Resize to model input dimensions (112×112) using bicubic interpolation
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0.0, 255.0)


# ── Vectorized Test-Time Augmentation (TTA) ─────────────────────────────────

def _predict_with_tta(face_rgb):
    """
    Predict with Vectorized TTA: Stack transformations into a single batch
    to trigger massive GPU parallelization.

    Args:
        face_rgb: uint8 RGB numpy array, NOT yet preprocessed.
    Returns:
        Averaged probability array of shape (7,)
    """
    original = _v2s_preprocess(face_rgb)
    flipped = original[:, ::-1, :]  # Horizontal axis mirror
    brightened = np.clip(original * 1.15, 0, 255)  # Scalar luminosity addition

    # Combine into a single 4D batch tensor: Shape (3, 112, 112, 3)
    tensor_batch = np.stack([original, flipped, brightened], axis=0)

    # Execute single unified evaluation traversal over device architecture
    batch_predictions = _run_inference(tensor_batch)

    # Collapse batch space by calculating means across the batch axis
    return np.mean(batch_predictions, axis=0)


def _predict_single(face_rgb):
    """Predict without TTA utilizing a single unified matrix pass."""
    preprocessed = _v2s_preprocess(face_rgb)
    batch = np.expand_dims(preprocessed, axis=0)
    return _run_inference(batch)[0]


# ── Public API ───────────────────────────────────────────────────────────────

def predict_emotion(face_image, use_tta: bool = True, confidence_threshold: float = None) -> dict:
    """
    Takes a cropped face matrix, evaluates emotional classification probabilities,
    and returns verified scoring dictionaries. Bypasses mutable global race conditions.

    Args:
        face_image: numpy array (uint8 RGB) or PIL Image — any resolution,
                    will be resized to INPUT_SIZE internally.
        use_tta: Thread-safe explicit execution control flag for TTA pipeline bypass
        confidence_threshold: Override for the global CONFIDENCE_THRESHOLD.
                              Use a lower value for live webcam predictions.
    Returns:
        dict with keys: emotion, confidence, all_probs
    """
    _load_model()

    if isinstance(face_image, Image.Image):
        face_rgb = np.array(face_image.convert("RGB"), dtype=np.uint8)
    else:
        face_rgb = face_image

    # Execution paths parsed via non-mutable function state parameters
    if TTA_ENABLED and use_tta:
        probs = _predict_with_tta(face_rgb)
    else:
        probs = _predict_single(face_rgb)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    all_probs = {
        label: round(float(p), 4) for label, p in zip(EMOTION_LABELS, probs)
    }

    threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    selected_emotion = "Uncertain" if confidence < threshold else EMOTION_LABELS[top_idx]

    return {
        "emotion": selected_emotion,
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
    }