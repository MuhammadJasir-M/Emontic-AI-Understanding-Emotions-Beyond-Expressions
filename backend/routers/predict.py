# backend/routers/predict.py
# POST /api/predict — the core emotion recognition endpoint.

import time
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from services.image_utils import validate_and_load_image
from services.face_detector import detect_and_crop_face
from services.emotion_predictor import predict_emotion, record_request

logger = logging.getLogger("emontic_ai")
router = APIRouter()


@router.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image, detect the largest face,
    predict the emotion, and return structured results.
    """
    start_time = time.perf_counter()
    file_bytes = await file.read()

    # Step 1: Validate and load image (with EXIF correction)
    image = validate_and_load_image(file_bytes, file.content_type)

    # Step 2: Detect and crop face
    try:
        cropped_face, bbox = detect_and_crop_face(image)
    except ValueError as e:
        record_request(no_face=True)
        raise HTTPException(status_code=422, detail=str(e))

    # Step 3: Predict emotion
    result = predict_emotion(cropped_face)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    logger.info(
        f"Prediction: {result['emotion']} ({result['confidence']:.2%}) "
        f"| bbox={bbox} | latency={elapsed_ms}ms"
    )

    # Record metrics
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
