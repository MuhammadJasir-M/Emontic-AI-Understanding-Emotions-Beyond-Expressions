# training/training/schedule.py
# Learning rate schedules and dataset processing utility calculations.

import math
from typing import Dict, Sequence
from sklearn.utils.class_weight import compute_class_weight
from training.core.config import get_batch_size
from training.core.losses import WarmupCosineDecay

def compute_class_weights(labels: Sequence[int]) -> Dict[int, float]:
    """
    Compute balanced class weights based on the active dataset distribution.
    Acts as a secure fallback interface for the training stages.
    """
    import numpy as np
    classes = np.array(sorted(set(labels)))
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(c): float(w) for c, w in zip(classes, cw)}

def compute_steps(n_samples: int, arch: str, epochs: int):
    """
    Calculate steps per epoch and total execution steps for the training run.
    Uses ceiling boundaries to protect trailing fractional batch paths.
    """
    batch_size = get_batch_size(arch)
    steps_per_epoch = math.ceil(n_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    return steps_per_epoch, total_steps

def make_lr_schedule(base_lr: float, total_steps: int):
    """
    Factory builder to bind values cleanly to the WarmupCosineDecay footprint.
    """
    return WarmupCosineDecay(base_lr=base_lr, total_steps=total_steps)