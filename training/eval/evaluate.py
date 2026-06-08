# training/eval/evaluate.py
import json
import numpy as np
import os
import keras

from training.core.config import (
    NUM_CLASSES, LABELS, get_eval_report_path, 
    get_confusion_matrix_path, RAFDB_TEST_CSV, RAFDB_TEST_IMAGES
)
from training.data.pipeline import build_rafdb_dataset, _parse_rafdb_csv
from training.core.losses import MacroF1

# Import the new architectural elements
from training.core.model import SpatialPositionalEmbedding, AttentionPooling, ArcFaceDense
from training.core.losses import ArcFaceLoss


def _save_confusion_matrix(cm, cm_path: str, labels, arch: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix — {arch.upper()}")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"[OK] Confusion matrix saved: {cm_path}")
    except Exception:
        np.save(f"{cm_path}.npy", cm)


from training.training.stages import FastTrainModel

def evaluate_checkpoint(arch: str, checkpoint: str):
    """Evaluate model on RAF-DB test split and save report JSON."""
    
    # Safely inject the new spatial layers into the deserializer context
    model = keras.models.load_model(
        checkpoint,
        custom_objects={
            "FastTrainModel": FastTrainModel,
            "MacroF1": MacroF1,
            "SpatialPositionalEmbedding": SpatialPositionalEmbedding,
            "AttentionPooling": AttentionPooling,
            "ArcFaceDense": ArcFaceDense,
            "ArcFaceLoss": ArcFaceLoss,
        },
        compile=False
    )

    test_ds = build_rafdb_dataset("test", arch)

    print("Evaluating best model to generate Confusion Matrix...\n")
    
    probs_list = []
    for x_batch, _ in test_ds:
        preds = model(x_batch, training=False)
        probs_list.append(preds.numpy())
    probs_all = np.concatenate(probs_list, axis=0).astype(np.float32)
    
    _, test_labels = _parse_rafdb_csv(RAFDB_TEST_CSV, RAFDB_TEST_IMAGES)
    y_true = np.array(test_labels, dtype=np.int32)
    
    # Ensure lengths match in case dataset pipeline has issues
    if len(probs_all) != len(y_true):
        print(f"WARNING: Length mismatch! Preds: {len(probs_all)}, True: {len(y_true)}")
        # Truncate to shortest length just in case
        min_len = min(len(probs_all), len(y_true))
        probs_all = probs_all[:min_len]
        y_true = y_true[:min_len]
        
    y_pred = np.argmax(probs_all, axis=-1)

    overall_acc = float((y_true == y_pred).mean())

    from sklearn.metrics import confusion_matrix, classification_report
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    clf = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report = {
        "architecture": arch,
        "checkpoint": checkpoint,
        "overall_accuracy": round(overall_acc, 6),
        "per_class": clf,
    }

    rp = get_eval_report_path(arch)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    cm_path = get_confusion_matrix_path(arch)
    _save_confusion_matrix(cm, cm_path, LABELS, arch)

    keras.backend.clear_session()

    return report