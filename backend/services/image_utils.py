# backend/services/image_utils.py
# Image validation, loading, and EXIF orientation correction.

from PIL import Image, ImageOps
import io
from fastapi import HTTPException

from config import MAX_FILE_SIZE_MB, ALLOWED_CONTENT_TYPES


def validate_and_load_image(file_bytes: bytes, content_type: str) -> Image.Image:
    """
    Validates upload constraints (file type, size) and returns a PIL Image.
    Applies EXIF orientation correction — critical for mobile camera uploads
    where the image may appear rotated without this fix.
    """
    # ── File size check ──────────────────────────────────────────────────
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    # ── Content type check ───────────────────────────────────────────────
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Accepted: JPEG, PNG, or WebP.",
        )

    # ── Load and decode ──────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or unreadable image file. Please try a different image.",
        )

    # ── EXIF orientation correction ──────────────────────────────────────
    # Mobile cameras embed rotation metadata in EXIF.  Without this,
    # a portrait photo may be fed to the model sideways.
    image = ImageOps.exif_transpose(image)

    return image
