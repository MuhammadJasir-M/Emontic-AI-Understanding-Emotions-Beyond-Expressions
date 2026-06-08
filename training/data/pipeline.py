# training/data/pipeline.py
# Optimized dataset builders featuring dynamic minority oversampling 
# and hardware-accelerated batch augmentation.

import csv
import os
from collections import Counter
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split

from training.core.config import (
    SEED, NUM_CLASSES, LABELS,
    RAFDB_TRAIN_CSV, RAFDB_TEST_CSV,
    RAFDB_TRAIN_IMAGES, RAFDB_TEST_IMAGES,
    AFFECTNET_LABELS_CSV, AFFECTNET_TRAIN_IMAGES, AFFECTNET_TEST_IMAGES,
    AFFECTNET_MIN_CONFIDENCE, AFFECTNET_EXCLUDED,
    convert_rafdb_label, convert_affectnet_label,
    get_input_shape, get_batch_size,
    RAFDB_VAL_FRACTION, PRETRAIN_VAL_FRACTION,
    get_split_file,
)
from .augment import get_augmentation_pipeline

MAP_PARALLEL_CALLS = tf.data.AUTOTUNE
PREFETCH_BUFFER = tf.data.AUTOTUNE


def _parse_rafdb_csv(csv_path, image_root):
    paths, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["image"].strip()
            raf_label = int(row["label"].strip())
            unified = convert_rafdb_label(raf_label)
            fpath = os.path.join(str(image_root), str(raf_label), filename)
            if not os.path.isfile(fpath):
                continue
            paths.append(fpath)
            labels.append(unified)
    return paths, labels


def _parse_affectnet_csv(csv_path, image_root, min_confidence=0.0):
    paths, labels = [], []
    excluded_lower = {s.lower() for s in AFFECTNET_EXCLUDED}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = row["label"].strip()
            if label_str.lower() in excluded_lower:
                continue
            unified = convert_affectnet_label(label_str)
            if unified is None:
                continue
            confidence = float(row["relFCs"].strip())
            if confidence < min_confidence:
                continue
            pth = row["pth"].strip()
            fpath = os.path.join(str(image_root), pth)
            if not os.path.isfile(fpath):
                continue
            paths.append(fpath)
            labels.append(unified)
    return paths, labels


def _scan_affectnet_test_dir(test_root):
    paths, labels = [], []
    excluded_lower = {s.lower() for s in AFFECTNET_EXCLUDED}
    for folder_name in sorted(os.listdir(str(test_root))):
        folder_path = os.path.join(str(test_root), folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.lower() in excluded_lower:
            continue
        unified = convert_affectnet_label(folder_name)
        if unified is None:
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder_path, fname))
                labels.append(unified)
    return paths, labels


def _oversample_minority_classes(paths, labels, boost_factor=3, threshold_pct=10.0):
    total_samples = len(labels)
    counts = Counter(labels)
    
    boosted_paths = []
    boosted_labels = []
    
    for p, l in zip(paths, labels):
        class_pct = (counts[l] / total_samples) * 100.0
        if class_pct < threshold_pct:
            for _ in range(boost_factor):
                boosted_paths.append(p)
                boosted_labels.append(l)
        else:
            boosted_paths.append(p)
            boosted_labels.append(l)
            
    return boosted_paths, boosted_labels


def compute_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    class_weights = {
        cls: round(total / (NUM_CLASSES * count), 4)
        for cls, count in counts.items()
    }
    for idx in range(NUM_CLASSES):
        if idx not in class_weights:
            class_weights[idx] = 1.0
            
    return class_weights


def _decode_and_resize(raw, label, input_size):
    image = tf.io.decode_jpeg(raw, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    # BILINEAR: 5x faster than BICUBIC with negligible quality loss at 224px
    image = tf.image.resize(image, input_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def _read_raw(file_path, label):
    # Reads the raw file bytes without decoding them (very lightweight)
    return tf.io.read_file(file_path), label


def _build_dataset(paths, labels, arch, training=False, augment_strength=1.0, shuffle=True):
    input_size = get_input_shape(arch)[:2]
    batch_size = get_batch_size(arch)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Phase 1: Read raw file bytes (lightweight — defers decode to Phase 4)
    ds = ds.map(_read_raw, num_parallel_calls=MAP_PARALLEL_CALLS)

    # Phase 2: Cache raw JPEG binaries in RAM
    # Eliminates filesystem I/O for all epochs after the first.
    ds = ds.cache()

    # Phase 3: Shuffle the cached binaries
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 50000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    # Phase 4: Decode to Images (happens on the fly, preventing the 6GB RAM crash)
    def _decode(raw, label):
        img, lbl = _decode_and_resize(raw, label, input_size)
        return img, lbl

    ds = ds.map(_decode, num_parallel_calls=MAP_PARALLEL_CALLS)

    # Phase 5: BATCH with drop_remainder to prevent XLA recompilation on partial batches
    ds = ds.batch(batch_size, drop_remainder=True)

    # Phase 6: GPU-vectorized augmentation (0.023s/batch vs 0.137s on CPU)
    if training:
        aug_pipeline = get_augmentation_pipeline(augment_strength)
        ds = ds.map(lambda x, y: (aug_pipeline(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(PREFETCH_BUFFER)
    return ds


def dataset_stats(paths, labels, name="dataset"):
    counts = Counter(labels)
    total = len(labels)
    print(f"\n{'='*60}")
    print(f" {name} — {total} samples")
    print(f"{'='*60}")
    for idx in range(NUM_CLASSES):
        cnt = counts.get(idx, 0)
        pct = 100.0 * cnt / total if total > 0 else 0
        print(f" {idx} {LABELS[idx]:>10s}: {cnt:>6d} ({pct:5.1f}%)")
    print(f"{'='*60}\n")
    return counts


def _write_split(split_path, paths, labels):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(zip(paths, labels))


def _read_split(split_path):
    paths, labels = [], []
    with open(split_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["path"].strip()
            # Skip per-file stat checks — splits are authoritative after generation
            paths.append(p)
            labels.append(int(row["label"]))
    return paths, labels


def _get_or_create_split(dataset_name, all_paths, all_labels, val_fraction):
    train_file = get_split_file(dataset_name, "train")
    val_file = get_split_file(dataset_name, "val")

    if train_file.is_file() and val_file.is_file():
        train_paths, train_labels = _read_split(train_file)
        val_paths, val_labels = _read_split(val_file)
        if train_paths and val_paths:
            return train_paths, train_labels, val_paths, val_labels

    train_idx, val_idx = train_test_split(
        range(len(all_paths)),
        test_size=val_fraction,
        stratify=all_labels,
        random_state=SEED,
    )

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    _write_split(train_file, train_paths, train_labels)
    _write_split(val_file, val_paths, val_labels)
    return train_paths, train_labels, val_paths, val_labels


def get_rafdb_split_labels(split):
    all_paths, all_labels = _parse_rafdb_csv(RAFDB_TRAIN_CSV, RAFDB_TRAIN_IMAGES)
    train_paths, train_labels, val_paths, val_labels = _get_or_create_split(
        "rafdb", all_paths, all_labels, RAFDB_VAL_FRACTION
    )
    if split == "train":
        return train_paths, train_labels
    if split == "val":
        return val_paths, val_labels
    raise ValueError(f"Unsupported split: {split}")


def get_affectnet_split_labels(split):
    all_paths, all_labels = _parse_affectnet_csv(
        AFFECTNET_LABELS_CSV,
        AFFECTNET_TRAIN_IMAGES,
        min_confidence=AFFECTNET_MIN_CONFIDENCE,
    )
    train_paths, train_labels, val_paths, val_labels = _get_or_create_split(
        "affectnet", all_paths, all_labels, PRETRAIN_VAL_FRACTION
    )
    if split == "train":
        return train_paths, train_labels
    if split == "val":
        return val_paths, val_labels
    raise ValueError(f"Unsupported split: {split}")


def build_rafdb_dataset(split, arch):
    if split == "test":
        paths, labels = _parse_rafdb_csv(RAFDB_TEST_CSV, RAFDB_TEST_IMAGES)
        dataset_stats(paths, labels, "RAF-DB Test Split")
        return _build_dataset(paths, labels, arch, training=False, shuffle=False)

    if split == "val":
        paths, labels = get_rafdb_split_labels("val")
        dataset_stats(paths, labels, "RAF-DB Validation Split")
        return _build_dataset(paths, labels, arch, training=False, shuffle=False)

    paths, labels = get_rafdb_split_labels("train")
    dataset_stats(paths, labels, "RAF-DB Active Train Split (Pre-Oversampling)")
    
    paths, labels = _oversample_minority_classes(paths, labels, boost_factor=3, threshold_pct=10.0)
    dataset_stats(paths, labels, "RAF-DB Post-Oversampling Distribution Vector")
    
    return _build_dataset(paths, labels, arch, training=True, augment_strength=1.0, shuffle=True)


def build_affectnet_dataset(split, arch):
    if split == "test":
        paths, labels = _scan_affectnet_test_dir(AFFECTNET_TEST_IMAGES)
        dataset_stats(paths, labels, "AffectNet Test Split")
        return _build_dataset(paths, labels, arch, training=False, shuffle=False)

    if split == "val":
        paths, labels = get_affectnet_split_labels("val")
        dataset_stats(paths, labels, "AffectNet Validation Split")
        return _build_dataset(paths, labels, arch, training=False, shuffle=False)

    paths, labels = get_affectnet_split_labels("train")
    dataset_stats(paths, labels, "AffectNet Active Train Split (Pre-Oversampling)")
    
    paths, labels = _oversample_minority_classes(paths, labels, boost_factor=3, threshold_pct=10.0)
    dataset_stats(paths, labels, "AffectNet Post-Oversampling Distribution Vector")
    
    return _build_dataset(paths, labels, arch, training=True, augment_strength=1.4, shuffle=True)