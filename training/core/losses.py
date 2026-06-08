# training/core/losses.py
# Production loss math and macro metrics tracking for Emontic AI.
# Fixed to return unreduced per-sample losses for full Keras class_weight support.

import math
import keras
import tensorflow as tf

from .config import NUM_CLASSES, WARMUP_RATIO


@keras.saving.register_keras_serializable(package="emontic_ai")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule combining linear warmup with cosine decay.
    """
    def __init__(self, base_lr, total_steps, warmup_ratio=WARMUP_RATIO, initial_lr=1e-6):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.initial_lr = float(initial_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warmup from initial_lr to base_lr
        warmup_lr = self.initial_lr + (self.base_lr - self.initial_lr) * (step / tf.maximum(warmup, 1.0))
        
        decay_steps = tf.maximum(total - warmup, 1.0)
        progress = (step - warmup) / decay_steps
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_ratio": float(self.warmup_steps) / float(self.total_steps),
            "initial_lr": self.initial_lr,
        }


@keras.saving.register_keras_serializable(package="emontic_ai")
class SparseFocalLoss(keras.losses.Loss):
    """
    Sparse Focal Loss function to balance minority gradients.
    Returns unreduced per-sample arrays to support Keras sample and class weighting.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, name="sparse_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        # 1. Isolate target maps before label smoothing to ensure accurate true-class probability
        raw_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=tf.float32)
        p_t = tf.reduce_sum(raw_oh * y_pred, axis=-1)
        focal = tf.pow(1.0 - p_t, self.gamma)

        # 2. Smooth the targets separately to regularize the cross-entropy calculation
        y_true_oh = raw_oh
        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / num_classes
            y_true_oh = y_true_oh * smooth_pos + smooth_neg

        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        
        # Unreduced per-sample element-wise multiplication allows class-weighting to function
        return focal * ce

    def get_config(self):
        return {
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
            "name": self.name,
        }


@keras.saving.register_keras_serializable(package="emontic_ai")
class MacroF1(keras.metrics.Metric):
    """
    State-tracking metric to evaluate Macro F1 stability across imbalanced fields.
    """
    def __init__(self, num_classes=NUM_CLASSES, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        y_true_oh = tf.one_hot(y_true, self.num_classes)
        y_pred_oh = tf.one_hot(y_pred_labels, self.num_classes)

        self.tp.assign_add(tf.reduce_sum(y_true_oh * y_pred_oh, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - y_true_oh) * y_pred_oh, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true_oh * (1 - y_pred_oh), axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros(self.num_classes))
        self.fp.assign(tf.zeros(self.num_classes))
        self.fn.assign(tf.zeros(self.num_classes))

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        return config


@keras.saving.register_keras_serializable(package="emontic_ai")
class ArcFaceLoss(keras.losses.Loss):
    """
    Additive Angular Margin Loss (ArcFace).
    Enforces a strict angular margin on the hypersphere to maximize inter-class 
    separability and intra-class compactness.
    """
    def __init__(self, margin=0.3, scale=30.0, name="arcface_loss"):
        super().__init__(name=name)
        self.margin = float(margin)
        self.scale = float(scale)

    def call(self, y_true, y_pred):
        # y_pred represents scaled logits: cos_theta * scale
        cos_theta = y_pred / self.scale
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y_true, depth=tf.shape(cos_theta)[-1], dtype=tf.float32)
        
        theta = tf.acos(cos_theta)
        target_logits = tf.cos(theta + self.margin)
        
        # Apply margin to the target class
        logits = tf.where(tf.cast(one_hot, tf.bool), target_logits, cos_theta)
        logits = logits * self.scale
        
        # Use Sparse Categorical Crossentropy on the modified logits
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True)

    def get_config(self):
        return {"margin": self.margin, "scale": self.scale, "name": self.name}