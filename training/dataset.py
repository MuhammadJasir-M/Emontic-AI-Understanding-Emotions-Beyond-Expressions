# training/dataset.py
# Unified data loaders for AffectNet and RAF-DB datasets.
# Returns tf.data.Dataset pipelines ready for training.

import os
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from augmentation import build_augmentation_layer

from config import (
    AFFECTNET_DIR, RAFDB_DIR, INPUT_SIZE, BATCH_SIZE, SEED,
    NUM_CLASSES, AFFECTNET_LABEL_MAP, AFFECTNET_EXCLUDE_CLASSES,
    RAFDB_LABEL_MAP, EMOTION_LABELS,
)

logger = logging.getLogger("emontic_ai.dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── EfficientNet Preprocessing ───────────────────────────────────────────────
def efficientnet_preprocess(image, label):
    """Apply EfficientNet-specific preprocessing (NOT simple /255.0)."""
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def resize_image(image, label):
    """Resize image to INPUT_SIZE."""
    image = tf.image.resize(image, INPUT_SIZE)
    return image, label


# ═══════════════════════════════════════════════════════════════════════════════
#  AffectNet Loader
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_affectnet_split(split_dir):
    """
    Discover images from AffectNet folder structure.
    Expected: split_dir/{class_index_or_name}/image.jpg

    Returns list of (filepath, target_label) tuples.
    """
    samples = []
    split_path = Path(split_dir)

    if not split_path.exists():
        logger.warning(f"AffectNet split directory not found: {split_dir}")
        return samples

    # Try numeric subfolder names first (0, 1, 2, ..., 7)
    for class_dir in sorted(split_path.iterdir()):
        if not class_dir.is_dir():
            continue

        # Parse class index from folder name
        try:
            affectnet_class = int(class_dir.name)
        except ValueError:
            # Try mapping folder names like "anger", "happy", etc.
            name_map = {
                "anger": 0, "angry": 0,
                "contempt": 1,
                "disgust": 2,
                "fear": 3,
                "happy": 4, "happiness": 4,
                "neutral": 5,
                "sad": 6, "sadness": 6,
                "surprise": 7, "surprised": 7,
            }
            affectnet_class = name_map.get(class_dir.name.lower())
            if affectnet_class is None:
                logger.warning(f"Unknown AffectNet class folder: {class_dir.name}")
                continue

        # Skip excluded classes (Contempt)
        if affectnet_class in AFFECTNET_EXCLUDE_CLASSES:
            continue

        target_label = AFFECTNET_LABEL_MAP.get(affectnet_class)
        if target_label is None:
            continue

        # Collect all image files
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((str(img_file), target_label))

    return samples


def load_affectnet(split="train", batch_size=BATCH_SIZE):
    """
    Load AffectNet dataset as a tf.data.Dataset.

    Args:
        split: "train" or "test" (maps to Train/ or Test/ folder)
        batch_size: Batch size for the dataset
    Returns:
        tf.data.Dataset of (image, one_hot_label) pairs
    """
    # Try common folder name variants
    split_names = {
        "train": ["Train", "train", "training"],
        "test": ["Test", "test", "validation", "val", "Val"],
    }

    split_dir = None
    for name in split_names.get(split, [split]):
        candidate = os.path.join(AFFECTNET_DIR, name)
        if os.path.isdir(candidate):
            split_dir = candidate
            break

    if split_dir is None:
        raise FileNotFoundError(
            f"AffectNet {split} directory not found in {AFFECTNET_DIR}. "
            f"Expected one of: {split_names.get(split, [split])}"
        )

    samples = _discover_affectnet_split(split_dir)
    if not samples:
        raise ValueError(f"No valid images found in {split_dir}")

    # Log class distribution
    label_counts = {}
    for _, label in samples:
        label_counts[EMOTION_LABELS[label]] = label_counts.get(EMOTION_LABELS[label], 0) + 1
    logger.info(f"AffectNet {split}: {len(samples)} images")
    for emotion, count in sorted(label_counts.items()):
        logger.info(f"  {emotion}: {count}")

    # Build tf.data.Dataset
    filepaths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(labels, depth=NUM_CLASSES)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_decode(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, INPUT_SIZE)
        return img, label
    
    if split == "train":
        ds = ds.shuffle(buffer_size=min(len(samples), 50000), seed=SEED)

    ds = ds.map(load_and_decode, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    # ADD THIS: Apply GPU augmentations to the training set!
    if split == "train":
        data_augmentation = build_augmentation_layer()  # <--- Moved it here!
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(efficientnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(samples), labels


# ═══════════════════════════════════════════════════════════════════════════════
#  RAF-DB Loader (supports both folder-based and annotation-file formats)
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_rafdb_folders(rafdb_dir, split):
    """
    Discover RAF-DB images from folder structure.
    Handles two formats:
      - Numbered subfolders: train/{1,2,3,4,5,6,7}/image.jpg (Kaggle mirror)
      - Named subfolders:   train/{angry,disgust,...}/image.jpg

    Returns list of (filepath, target_label) tuples.
    """
    split_dir = os.path.join(rafdb_dir, split)
    if not os.path.isdir(split_dir):
        # Try capitalized
        split_dir = os.path.join(rafdb_dir, split.capitalize())
        if not os.path.isdir(split_dir):
            return []

    samples = []
    split_path = Path(split_dir)

    # Name-based mapping for RAF-DB class folders
    name_map = {
        "surprise": 1, "surprised": 1,
        "fear": 2, "fearful": 2,
        "disgust": 3, "disgusted": 3,
        "happy": 4, "happiness": 4,
        "sad": 5, "sadness": 5,
        "angry": 6, "anger": 6,
        "neutral": 7,
    }

    for class_dir in sorted(split_path.iterdir()):
        if not class_dir.is_dir():
            continue

        # Try numeric class index first (1-7)
        try:
            rafdb_class = int(class_dir.name)
        except ValueError:
            # Try name-based mapping
            rafdb_class = name_map.get(class_dir.name.lower())
            if rafdb_class is None:
                logger.warning(f"Unknown RAF-DB class folder: {class_dir.name}")
                continue

        target_label = RAFDB_LABEL_MAP.get(rafdb_class)
        if target_label is None:
            logger.warning(f"RAF-DB class {rafdb_class} not in label map, skipping")
            continue

        # Collect image files
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((str(img_file), target_label))

    return samples


def _parse_rafdb_annotations(rafdb_dir):
    """
    Parse RAF-DB list_patition_label.txt annotation file (official format).
    Returns dict: {split: [(filepath, target_label), ...]}
    """
    annotation_candidates = [
        os.path.join(rafdb_dir, "EmoLabel", "list_patition_label.txt"),
        os.path.join(rafdb_dir, "list_patition_label.txt"),
        os.path.join(rafdb_dir, "basic", "EmoLabel", "list_patition_label.txt"),
    ]

    annotation_file = None
    for candidate in annotation_candidates:
        if os.path.isfile(candidate):
            annotation_file = candidate
            break

    if annotation_file is None:
        return None  # No annotation file found

    image_dir_candidates = [
        os.path.join(rafdb_dir, "Image", "aligned"),
        os.path.join(rafdb_dir, "basic", "Image", "aligned"),
        os.path.join(rafdb_dir, "aligned"),
        os.path.join(rafdb_dir, "Image", "original"),
        os.path.join(rafdb_dir, "Image"),
    ]

    image_dir = None
    for candidate in image_dir_candidates:
        if os.path.isdir(candidate):
            image_dir = candidate
            break

    if image_dir is None:
        return None

    logger.info(f"RAF-DB annotations: {annotation_file}")
    logger.info(f"RAF-DB images: {image_dir}")

    splits = {"train": [], "test": []}
    with open(annotation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            filename, rafdb_label = parts[0], int(parts[1])
            split = "train" if filename.startswith("train") else "test" if filename.startswith("test") else None
            if split is None:
                continue
            target_label = RAFDB_LABEL_MAP.get(rafdb_label)
            if target_label is None:
                continue
            base_name = filename.replace(".jpg", "").replace(".png", "")
            for ext in [base_name + "_aligned.jpg", filename, base_name + ".png"]:
                filepath = os.path.join(image_dir, ext)
                if os.path.isfile(filepath):
                    splits[split].append((filepath, target_label))
                    break

    return splits


def load_rafdb(split="train", batch_size=BATCH_SIZE):
    """
    Load RAF-DB dataset as a tf.data.Dataset.
    Auto-detects format: folder-based (Kaggle) or annotation-file (official).
    """
    logger.info(f"RAF-DB directory: {RAFDB_DIR}")

    # Strategy 1: Try folder-based loading (Kaggle mirror with numbered subfolders)
    samples = _discover_rafdb_folders(RAFDB_DIR, split)

    # Strategy 2: Try annotation-file based loading (official RAF-DB)
    if not samples:
        logger.info("No folder-based RAF-DB found, trying annotation file...")
        parsed = _parse_rafdb_annotations(RAFDB_DIR)
        if parsed:
            samples = parsed.get(split, [])

    if not samples:
        raise ValueError(
            f"No RAF-DB {split} images found in {RAFDB_DIR}. "
            f"Expected either numbered subfolders (1-7) or list_patition_label.txt"
        )

    # Log class distribution
    label_counts = {}
    for _, label in samples:
        label_counts[EMOTION_LABELS[label]] = label_counts.get(EMOTION_LABELS[label], 0) + 1
    logger.info(f"RAF-DB {split}: {len(samples)} images")
    for emotion, count in sorted(label_counts.items()):
        logger.info(f"  {emotion}: {count}")

    # Build tf.data.Dataset
    filepaths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(labels, depth=NUM_CLASSES)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_decode(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, INPUT_SIZE)
        return img, label

    # <--- Notice how there is NO ds.map() here anymore! --->

    # 1. SHUFFLE THE LIGHTWEIGHT TEXT PATHS FIRST
    if split == "train":
        ds = ds.shuffle(buffer_size=min(len(samples), 20000), seed=SEED)

    # 2. THEN MAP THE DECODER (using only 2 workers to save RAM)
    ds = ds.map(load_and_decode, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. BATCH THE DATA
    ds = ds.batch(batch_size)

    # 4. Apply GPU augmentations to the training set
    if split == "train":
        data_augmentation = build_augmentation_layer()  
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(efficientnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(samples), labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: Extract labels for class weight computation
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_weights(labels):
    """
    Compute balanced class weights from a list of integer labels.
    """
    from sklearn.utils import class_weight
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )
    weight_dict = dict(zip(classes.astype(int), weights))

    logger.info("Class weights:")
    for idx, weight in sorted(weight_dict.items()):
        logger.info(f"  {EMOTION_LABELS[idx]}: {weight:.4f}")

    return weight_dict
