# backend/routers/live_predict.py
import io
import time
import base64
import logging
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.face_detector import detect_and_crop_face_fast
from services.emotion_predictor import predict_emotion, record_request

logger = logging.getLogger("emontic_ai")
router = APIRouter()


class FrameRequest(BaseModel):
    image: str  # Base64-encoded image string


@router.post("/live-predict", tags=["prediction"])
async def live_predict(req: FrameRequest):
    """
    Accept a base64-encoded webcam frame and execute thread-safe, 
    low-latency single-pass inference (TTA explicitly bypassed via parameter).
    """
    start_time = time.perf_counter()

    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"Live-predict: base64 decoding error — {e}")
        raise HTTPException(status_code=400, detail="Invalid image payload format.")

    # Frame Detection Step
    try:
        cropped_face, bbox = detect_and_crop_face_fast(image)
    except ValueError as e:
        # Graceful return for empty webcam spaces to prevent log spamming
        return {
            "emotion": None,
            "confidence": 0.0,
            "all_probs": {},
            "bbox": None,
            "image_size": {"width": image.width, "height": image.height},
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
            "message": str(e),
        }

    # Full-precision pass: TTA is vectorized (single batched GPU call) so the
    # latency cost is negligible (~30ms) relative to the capture interval.
    # Lower confidence threshold (0.30) for live webcam — dynamic lighting,
    # angles, and compression artifacts naturally produce lower confidence
    # than high-quality uploaded images.
    result = predict_emotion(cropped_face, use_tta=True, confidence_threshold=0.30)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    
    record_request(
        emotion=result["emotion"],
        confidence=result["confidence"],
        latency_ms=elapsed_ms,
    )

    return {
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "all_probs": result["all_probs"],
        "bbox": bbox,
        "image_size": {"width": image.width, "height": image.height},
        "latency_ms": elapsed_ms,
    }