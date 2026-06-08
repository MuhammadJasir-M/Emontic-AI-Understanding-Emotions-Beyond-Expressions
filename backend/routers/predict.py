# backend/routers/predict.py
import os
import uuid
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import UPLOAD_DIR
from services.image_utils import validate_and_load_image
from services.face_detector import detect_and_crop_face_fast
from services.emotion_predictor import predict_emotion, record_request
from database import save_prediction

logger = logging.getLogger("emontic_ai")
router = APIRouter()


@router.post("/predict", tags=["prediction"])
async def predict(
    file: UploadFile = File(...),
    person_name: str = Form(...),
):
    """High-precision prediction path utilizing full Test-Time Augmentation."""
    start_time = time.perf_counter()

    person_name = person_name.strip()
    if not person_name:
        raise HTTPException(status_code=422, detail="Person name identifier is required.")

    file_bytes = await file.read()
    image = validate_and_load_image(file_bytes, file.content_type)

    try:
        cropped_face, bbox = detect_and_crop_face_fast(image)
    except ValueError as e:
        record_request(no_face=True)
        raise HTTPException(status_code=422, detail=str(e))

    # Explicitly utilize TTA processing for the deep file analysis channel
    result = predict_emotion(cropped_face, use_tta=True)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    
    record_request(
        emotion=result["emotion"],
        confidence=result["confidence"],
        latency_ms=elapsed_ms,
    )

    # File storage sequence
    image_filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename or '.jpg')[1]}"
    try:
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        with open(image_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.warning(f"File IO persistence failure: {e}")
        image_filename = ""

    # Database logging task
    try:
        save_prediction(
            person_name=person_name,
            image_path=image_filename,
            emotion=result["emotion"],
            confidence=result["confidence"],
        )
    except Exception as e:
        logger.warning(f"Database transaction failure: {e}")

    return {
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "all_probs": result["all_probs"],
        "bbox": bbox,
        "image_size": {"width": image.width, "height": image.height},
        "latency_ms": elapsed_ms,
        "person_name": person_name,
    }