# training/augmentation.py
# GPU-accelerated data augmentation for facial emotion recognition.
# Applied ONLY to training data, never to validation or test.

import tensorflow as tf


def build_augmentation_layer():
    """
    Build a Keras Sequential augmentation layer for on-GPU augmentation.
    Applied BEFORE EfficientNet preprocessing (on raw [0, 255] pixel values).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.042, fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 255)),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
            fill_mode="nearest",
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="nearest",
        ),
    ], name="train_augmentation")
