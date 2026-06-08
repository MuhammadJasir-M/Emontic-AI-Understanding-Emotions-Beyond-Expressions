# training/data/augment.py
# Hardware-accelerated data augmentation layer for facial expressions.
# Shifted from CPU tf.image ops to GPU-native Keras Preprocessing Layers.

import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable(package="Custom")
class RandomCutout(layers.Layer):
    """
    Randomly masks out a rectangular region of the image to simulate occlusions 
    like hands or glasses. This forces the model to rely on global features.
    """
    def __init__(self, mask_size=0.2, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size

    def call(self, inputs, training=None):
        if not training:
            return inputs
            
        # Get shape
        batch_size = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        c = tf.shape(inputs)[3]

        # Calculate mask dimensions (percentage of image)
        mask_h = tf.cast(tf.cast(h, tf.float32) * self.mask_size, tf.int32)
        mask_w = tf.cast(tf.cast(w, tf.float32) * self.mask_size, tf.int32)
        
        # Random top-left corner
        y = tf.random.uniform([batch_size], minval=0, maxval=h - mask_h, dtype=tf.int32)
        x = tf.random.uniform([batch_size], minval=0, maxval=w - mask_w, dtype=tf.int32)

        # Build mask using tf.map_fn or bounding boxes
        def _apply_mask(args):
            img, y_pos, x_pos = args
            # Create a 2D mask of 1s
            mask = tf.ones((h, w), dtype=img.dtype)
            
            # Use pad to create the black box
            box = tf.zeros((mask_h, mask_w), dtype=img.dtype)
            pad_top = y_pos
            pad_bottom = h - y_pos - mask_h
            pad_left = x_pos
            pad_right = w - x_pos - mask_w
            
            box_padded = tf.pad(box, [[pad_top, pad_bottom], [pad_left, pad_right]], constant_values=1.0)
            box_padded = tf.expand_dims(box_padded, -1)
            
            # Average pixel value or 0
            return img * box_padded

        # Vectorize across batch
        output = tf.map_fn(_apply_mask, (inputs, y, x), fn_output_signature=inputs.dtype)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"mask_size": self.mask_size})
        return config

def get_augmentation_pipeline(strength=1.0):
    """
    Returns a compiled Keras Sequential pipeline.
    TensorFlow will automatically execute these transformations directly 
    on the RTX 4060's Tensor Cores in parallel batches.
    """
    return tf.keras.Sequential([
        # Spatial Geometry
        layers.RandomFlip("horizontal"),
        # 0.02 factor roughly equals ~7 degrees of rotation
        layers.RandomRotation(factor=0.02 * strength),     
        layers.RandomZoom(height_factor=0.08 * strength), 
        layers.RandomTranslation(
            height_factor=0.04 * strength, 
            width_factor=0.04 * strength
        ),
        
        # Color & Lighting (maintaining bounds to avoid washing out features)
        layers.RandomBrightness(factor=0.08 * strength),
        layers.RandomContrast(factor=0.08 * strength),
        
        # Random Erasing (Occlusion simulation)
        RandomCutout(mask_size=0.20 * strength)
    ])