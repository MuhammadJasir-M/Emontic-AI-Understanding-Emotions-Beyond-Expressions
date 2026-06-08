# backend/services/face_detector.py
# Pure RetinaFace face detection and alignment pipeline.
# Bypasses MediaPipe completely to stabilize execution inside virtualized/WSL2 stacks.

import logging
import numpy as np
import cv2
from PIL import Image

from config import MIN_FACE_BOX_PX, INPUT_SIZE

logger = logging.getLogger("emontic_ai")


def _detect_faces_retinaface(image_array):
    """
    Detect faces using RetinaFace.
    Returns the largest face's bounding box and facial landmarks natively.

    Args:
        image_array: numpy array in BGR format (OpenCV convention)
    Returns:
        (facial_area, landmarks) of the largest face, or (None, None)
    """
    from retinaface import RetinaFace

    results = RetinaFace.detect_faces(image_array)

    if not results or not isinstance(results, dict):
        return None, None

    # Select the largest face by bounding box area
    best_key = max(
        results.keys(),
        key=lambda k: (
            (results[k]["facial_area"][2] - results[k]["facial_area"][0]) *
            (results[k]["facial_area"][3] - results[k]["facial_area"][1])
        ),
    )
    face = results[best_key]
    return face["facial_area"], face.get("landmarks")


def _align_face_native(image_array, landmarks):
    """
    Align face horizontally using the native eye landmarks from RetinaFace.
    Bypasses external EGL graphics binding contexts entirely.

    Args:
        image_array: numpy array in RGB format
        landmarks: dictionary containing facial keypoint arrays
    Returns:
        Aligned image as numpy array (RGB), or original if alignment fails
    """
    if not landmarks:
        return image_array

    try:
        h, w = image_array.shape[:2]

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return image_array

        # Extract eye center coordinates (x, y)
        lx, ly = int(left_eye[0]), int(left_eye[1])
        rx, ry = int(right_eye[0]), int(right_eye[1])

        # Compute horizontal rotation angle
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))

        # Only align if rotation is significant but not extreme
        if abs(angle) < 0.5:
            return image_array  
        if abs(angle) > 45:
            logger.debug(f"Extreme angle ({angle:.1f}°), skipping alignment")
            return image_array

        center = (float((lx + rx) / 2.0), float((ly + ry) / 2.0))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        aligned = cv2.warpAffine(
            image_array, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug(f"Face aligned natively by {angle:.1f}°")
        return aligned

    except Exception as e:
        logger.warning(f"Native face alignment failed: {e}")
        return image_array


def detect_and_crop_face(image: Image.Image):
    """
    Full face processing pipeline powered entirely by RetinaFace:
      1. Detect face & landmarks (RetinaFace)
      2. Align face natively using eye tracking coordinates
      3. Crop face bounding box
      4. Resize to INPUT_SIZE (112×112)

    Args:
        image: PIL Image in RGB mode.
    Returns:
        (cropped_face: numpy array RGB 112×112, bbox: dict)
    Raises:
        ValueError: If no face is found or face is too small.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Convert RGB to BGR for RetinaFace (OpenCV convention)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Step 1: Detect face and pull coordinates alongside tracking keypoints
    detection, landmarks = _detect_faces_retinaface(img_bgr)
    if detection is None:
        raise ValueError("No face detected in the image.")

    x1, y1, x2, y2 = detection

    # Clamp bounds safely
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

    # Step 2: Align face natively on full image context using extracted landmarks
    aligned_rgb = _align_face_native(img_array, landmarks)

    # Step 3: Crop face from the horizontally aligned matrix context
    face_crop = aligned_rgb[y1:y2, x1:x2]

    # Step 4: Resize to model input size using bicubic interpolation
    face_resized = cv2.resize(face_crop, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    bbox = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}

    return face_resized, bbox


# ── Lazy-loaded cascades for fast live detection ─────────────────────────────
_haar_face = None
_haar_eyes = None
_haar_profile = None


def _get_cascades():
    """Load OpenCV's Haar cascades once and cache them."""
    global _haar_face, _haar_eyes, _haar_profile
    if _haar_face is None:
        _haar_face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        _haar_eyes = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        _haar_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        logger.info("Haar cascade face, eye, and profile detectors loaded.")
    return _haar_face, _haar_eyes, _haar_profile


def _align_face_eyes(img_array, gray, fx, fy, fw, fh):
    """
    Lightweight face alignment using Haar eye cascade.
    Detects eyes within the face region and rotates to level them.
    Falls back to the unaligned image if eyes aren't found.

    Args:
        img_array: Full RGB image as numpy array.
        gray: Grayscale version of the full image.
        fx, fy, fw, fh: Face bounding box from Haar cascade.
    Returns:
        Aligned RGB image (or original if alignment not possible).
    """
    _, eye_cascade, _ = _get_cascades()

    # Search for eyes only within the upper half of the face region
    face_roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(fw * 0.08), int(fh * 0.08)),
        maxSize=(int(fw * 0.45), int(fh * 0.45)),
    )

    if len(eyes) < 2:
        return img_array  # Can't align without both eyes

    # Sort eyes by x-coordinate: leftmost first
    eyes_sorted = sorted(eyes, key=lambda e: e[0])
    # Take the two most separated eyes (first and last)
    left_eye = eyes_sorted[0]
    right_eye = eyes_sorted[-1]

    # Compute eye centers (relative to full image)
    lx = fx + left_eye[0] + left_eye[2] // 2
    ly = fy + left_eye[1] + left_eye[3] // 2
    rx = fx + right_eye[0] + right_eye[2] // 2
    ry = fy + right_eye[1] + right_eye[3] // 2

    angle = np.degrees(np.arctan2(ry - ly, rx - lx))

    # Only align if rotation is significant but not extreme
    if abs(angle) < 0.5 or abs(angle) > 30:
        return img_array

    h_img, w_img = img_array.shape[:2]
    center = (float((lx + rx) / 2.0), float((ly + ry) / 2.0))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        img_array, M, (w_img, h_img),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def detect_and_crop_face_fast(image: Image.Image):
    """
    Fast face detection for live webcam with eye-based alignment.
    Uses Haar cascades (~30-60ms total) for near real-time performance
    while maintaining accuracy comparable to RetinaFace.

    Pipeline:
      1. Detect face (Haar frontalface_alt2)
      2. Align face using eye landmarks (Haar eye)
      3. Crop with appropriate padding
      4. Enhance contrast with CLAHE
      5. Resize to INPUT_SIZE

    Args:
        image: PIL Image in RGB mode.
    Returns:
        (cropped_face: numpy array RGB 112×112, bbox: dict)
    Raises:
        ValueError: If no face is found.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    face_cascade, _, profile_cascade = _get_cascades()
    
    # 1. Try frontal face detection
    detections = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(MIN_FACE_BOX_PX, MIN_FACE_BOX_PX),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(detections) == 0:
        # 2. Try profile face detection (looking right)
        detections = profile_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(MIN_FACE_BOX_PX, MIN_FACE_BOX_PX),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(detections) == 0:
            # 3. Try profile face detection (looking left) by flipping image
            gray_eq_flipped = cv2.flip(gray_eq, 1)
            detections_flipped = profile_cascade.detectMultiScale(
                gray_eq_flipped,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(MIN_FACE_BOX_PX, MIN_FACE_BOX_PX),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            # Re-map bounding box coordinates to original un-flipped image
            if len(detections_flipped) > 0:
                detections = []
                for (x_flip, y_flip, fw_flip, fh_flip) in detections_flipped:
                    x_original = w - (x_flip + fw_flip)
                    detections.append([x_original, y_flip, fw_flip, fh_flip])

    if len(detections) == 0:
        raise ValueError("No face detected in the frame.")

    # Select the largest face by area
    fx, fy, fw, fh = max(detections, key=lambda d: d[2] * d[3])

    # Step 1: Align face using eye positions
    aligned = _align_face_eyes(img_array, gray, fx, fy, fw, fh)

    # Step 2: Crop with padding matched to RetinaFace-style output
    pad_x = int(0.12 * fw)
    pad_top = int(0.22 * fh)
    pad_bot = int(0.12 * fh)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_top)
    x2 = min(w, fx + fw + pad_x)
    y2 = min(h, fy + fh + pad_bot)

    face_crop = aligned[y1:y2, x1:x2]

    # Removed CLAHE contrast enhancement as it alters the input domain 
    # away from the training distribution (ImageNet/RAF-DB).

    # Step 4: Resize to model input
    face_resized = cv2.resize(face_crop, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    bbox = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
    return face_resized, bbox