# training/eval/calibrate.py
import json
import os
import numpy as np
import keras

from scipy.optimize import minimize_scalar

from training.core.config import get_calibration_json
from training.data.pipeline import build_rafdb_dataset, get_rafdb_split_labels
from training.core.losses import MacroF1

# Import the new architectural elements
from training.core.model import SpatialPositionalEmbedding, AttentionPooling


def _softmax_with_temperature(logits, temperature):
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask  = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = accuracies[mask].mean()
        ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


def calibrate_checkpoint(arch: str, checkpoint: str):
    """Compute temperature scaling on RAF-DB validation set and save JSON."""
    
    # Safely inject the new spatial layers into the deserializer context
    model = keras.models.load_model(
        checkpoint,
        custom_objects={
            "MacroF1": MacroF1,
            "SpatialPositionalEmbedding": SpatialPositionalEmbedding,
            "AttentionPooling": AttentionPooling
        },
        compile=False
    )

    val_ds = build_rafdb_dataset("val", arch)

    all_probs  = model.predict(val_ds, verbose=1).astype(np.float32)
    
    _, val_labels = get_rafdb_split_labels("val")
    all_labels = np.array(val_labels, dtype=np.int64)

    eps    = 1e-7
    logits = np.log(np.clip(all_probs, eps, 1.0 - eps))

    ece_before = compute_ece(all_probs, all_labels)

    def _nll_loss(temperature):
        probs_t = _softmax_with_temperature(logits, temperature)
        lp  = np.log(np.clip(probs_t, eps, 1.0))
        nll = -lp[np.arange(len(all_labels)), all_labels].mean()
        return nll

    res   = minimize_scalar(_nll_loss, bounds=(0.1, 10.0), method="bounded")
    opt_T = float(res.x)

    probs_after = _softmax_with_temperature(logits, opt_T)
    ece_after   = compute_ece(probs_after, all_labels)

    calibration = {
        "architecture":       arch,
        "temperature":        round(opt_T, 6),
        "ece_before":         round(ece_before, 6),
        "ece_after":          round(ece_after, 6),
        "validation_samples": len(all_labels),
    }

    cal_path = get_calibration_json(arch)
    os.makedirs(os.path.dirname(cal_path), exist_ok=True)
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"[OK] Calibration: T={opt_T:.4f} | ECE {ece_before:.4f} → {ece_after:.4f}")
    
    keras.backend.clear_session()
    
    return calibration