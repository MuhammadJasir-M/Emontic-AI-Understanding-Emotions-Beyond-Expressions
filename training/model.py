# training/model.py
# EfficientNetB0 model builder for Emontic AI v2.

import tensorflow as tf
from tensorflow.keras import layers, models

from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_DENSE_512, DROPOUT_DENSE_256


def build_model(num_classes=NUM_CLASSES, trainable_base=False):
    """
    Build EfficientNetB0-based emotion classification model.

    Architecture:
        EfficientNetB0 (ImageNet pretrained) → GAP → BN → Dense(512) →
        Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(7, softmax)

    Args:
        num_classes: Number of output classes (default: 7)
        trainable_base: Whether the EfficientNetB0 backbone is trainable
    """
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = base(inputs, training=trainable_base)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(DROPOUT_DENSE_512, name="dropout_512")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(DROPOUT_DENSE_256, name="dropout_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="emontic_ai_v2")
    return model


def unfreeze_top_layers(model, num_layers=60):
    """
    Unfreeze the top N layers of the EfficientNetB0 backbone
    for fine-tuning (Phase 2 / Stage B).

    Args:
        model: The compiled Keras model
        num_layers: Number of top layers to unfreeze
    """
    # The backbone is the first layer (functional model wrapping EfficientNetB0)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        # Fallback: try the first sub-model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break

    if backbone is None:
        raise ValueError("Could not find EfficientNetB0 backbone in model.")

    # Unfreeze top N layers
    backbone.trainable = True
    total = len(backbone.layers)
    freeze_until = max(0, total - num_layers)

    for i, layer in enumerate(backbone.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"Backbone: {total} total layers, {trainable_count} trainable "
          f"(unfroze top {num_layers})")


def get_model_summary(model):
    """Print model summary with trainable parameter counts."""
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    print(f"\nTrainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    print(f"Total params:         {trainable + non_trainable:,}")
    model.summary(show_trainable=True, expand_nested=False)
