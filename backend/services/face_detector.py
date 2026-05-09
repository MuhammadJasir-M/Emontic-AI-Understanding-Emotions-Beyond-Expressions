# backend/services/face_detector.py
# Face detection via RetinaFace + alignment via MediaPipe Face Mesh.
# Replaces the MediaPipe BlazeFace detector from v1.

import logging
import numpy as np
import cv2
from PIL import Image

from config import MIN_FACE_BOX_PX, INPUT_SIZE

logger = logging.getLogger("emontic_ai")


# ═══════════════════════════════════════════════════════════════════════════════
#  Face Detection — RetinaFace
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_faces_retinaface(image_array):
    """
    Detect faces using RetinaFace.
    Returns the largest face's bounding box and facial area.

    Args:
        image_array: numpy array in BGR format (OpenCV convention)
    Returns:
        (x1, y1, x2, y2) bounding box of the largest face, or None
    """
    from retinaface import RetinaFace

    # RetinaFace accepts numpy array (BGR) or file path
    results = RetinaFace.detect_faces(image_array)

    if not results or not isinstance(results, dict):
        return None

    # Select the largest face by bounding box area
    best_key = max(
        results.keys(),
        key=lambda k: (
            (results[k]["facial_area"][2] - results[k]["facial_area"][0]) *
            (results[k]["facial_area"][3] - results[k]["facial_area"][1])
        ),
    )
    face = results[best_key]
    x1, y1, x2, y2 = face["facial_area"]

    return x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════════════════════
#  Face Alignment — MediaPipe Face Mesh
# ═══════════════════════════════════════════════════════════════════════════════

def _align_face(image_array):
    """
    Align face using MediaPipe Face Mesh eye landmarks.
    Computes rotation angle from left eye (landmark 33) and right eye (landmark 263),
    then warps the image to horizontal alignment.

    Args:
        image_array: numpy array in RGB format
    Returns:
        Aligned image as numpy array (RGB), or original if alignment fails
    """
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        h, w = image_array.shape[:2]

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image_array)

            if not results.multi_face_landmarks:
                logger.debug("Face Mesh: no landmarks found, skipping alignment")
                return image_array

            landmarks = results.multi_face_landmarks[0].landmark

            # Left eye: landmark 33, Right eye: landmark 263
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)

            # Compute rotation angle
            angle = np.degrees(np.arctan2(ry - ly, rx - lx))

            # Only align if rotation is significant but not extreme
            if abs(angle) < 0.5:
                return image_array  # Already aligned
            if abs(angle) > 45:
                logger.debug(f"Extreme angle ({angle:.1f}°), skipping alignment")
                return image_array

            center = ((lx + rx) // 2, (ly + ry) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(
                image_array, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            logger.debug(f"Face aligned by {angle:.1f}°")
            return aligned

    except ImportError:
        logger.warning("MediaPipe not available, skipping face alignment")
        return image_array
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        return image_array


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_and_crop_face(image: Image.Image):
    """
    Full face processing pipeline:
      1. Detect face (RetinaFace)
      2. Align face (MediaPipe Face Mesh)
      3. Crop face bounding box
      4. Resize to INPUT_SIZE (224×224)

    Args:
        image: PIL Image in RGB mode.
    Returns:
        (cropped_face: numpy array RGB 224×224, bbox: dict)
    Raises:
        ValueError: If no face is found or face is too small.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Convert RGB to BGR for RetinaFace (OpenCV convention)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Step 1: Detect face
    detection = _detect_faces_retinaface(img_bgr)
    if detection is None:
        raise ValueError("No face detected in the image.")

    x1, y1, x2, y2 = detection

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Reject faces that are too small
    if (x2 - x1) < MIN_FACE_BOX_PX or (y2 - y1) < MIN_FACE_BOX_PX:
        raise ValueError(
            "Detected face is too small to analyze. "
            "Please upload a higher-resolution image with a clearly visible face."
        )

    # Step 2: Align face (on full image, before cropping)
    aligned_rgb = _align_face(img_array)

    # Step 3: Crop face from aligned image
    face_crop = aligned_rgb[y1:y2, x1:x2]

    # Step 4: Resize to model input size
    face_resized = cv2.resize(face_crop, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    bbox = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}

    return face_resized, bbox
