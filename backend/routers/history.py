# backend/routers/history.py
# GET /api/history/names — list unique person names.
# GET /api/history/{name} — list all predictions for a person.
# DELETE /api/history/record/{record_id} — delete a single prediction.

import os
import logging
from fastapi import APIRouter, HTTPException

from database import get_unique_names, get_history_by_name, delete_prediction
from config import UPLOAD_DIR

logger = logging.getLogger("emontic_ai")
router = APIRouter()


@router.get("/history/names", tags=["history"])
def list_names():
    """Return all unique person names from emotion_history."""
    try:
        names = get_unique_names()
        return {"names": names}
    except Exception as e:
        logger.error(f"Failed to fetch names: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prediction history names.")


@router.get("/history/{name}", tags=["history"])
def person_history(name: str):
    """
    Return all prediction records for a specific person,
    sorted with the latest predictions first.
    """
    try:
        records = get_history_by_name(name)
        return {"person_name": name, "records": records}
    except Exception as e:
        logger.error(f"Failed to fetch history for '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history for '{name}'.")


@router.delete("/history/record/{record_id}", tags=["history"])
def delete_record(record_id: int):
    """Delete a single prediction record and its associated uploaded image."""
    try:
        image_path = delete_prediction(record_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="Prediction record not found.")

        # Clean up the uploaded image file
        if image_path:
            full_path = os.path.join(UPLOAD_DIR, image_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    logger.info(f"Deleted image file: {full_path}")
                except OSError as e:
                    logger.warning(f"Failed to delete image file '{full_path}': {e}")

        return {"detail": "Prediction deleted successfully.", "id": record_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prediction #{record_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete the prediction record.")
