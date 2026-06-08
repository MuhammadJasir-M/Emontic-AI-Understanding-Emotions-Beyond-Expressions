# training/eval/bias_audit.py
import json
import os
import numpy as np
import keras

from sklearn.metrics import f1_score, precision_recall_fscore_support

from training.core.config import (
    NUM_CLASSES, LABELS, get_bias_report_path, get_checkpoint_dir,
    RAFDB_TEST_CSV, RAFDB_TEST_IMAGES
)
from training.data.pipeline import build_rafdb_dataset, _parse_rafdb_csv
from training.core.losses import MacroF1

# Import the new architectural elements
from training.core.model import SpatialPositionalEmbedding, AttentionPooling


def bias_audit(arch: str, checkpoint: str = None):
    """Audit model for per-class bias thresholds on the RAF-DB test subset."""
    if checkpoint is None:
        checkpoint = str(get_checkpoint_dir(arch) / "finetune_best.keras")

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

    test_ds = build_rafdb_dataset("test", arch)

    probs_all = model.predict(test_ds, verbose=1).astype(np.float32)
    
    _, test_labels = _parse_rafdb_csv(RAFDB_TEST_CSV, RAFDB_TEST_IMAGES)
    y_true = np.array(test_labels, dtype=np.int32)
    y_pred = np.argmax(probs_all, axis=-1)

    overall_acc = float((y_true == y_pred).mean())
    overall_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    flag_threshold = overall_acc - 0.10 

    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=list(range(NUM_CLASSES)), 
        average=None, 
        zero_division=0
    )

    per_class = {}
    flagged   = []

    for c in range(NUM_CLASSES):
        mask = y_true == c
        class_count = int(mask.sum())

        if class_count == 0:
            per_class[LABELS[c]] = {
                "count": 0, "accuracy": None,
                "f1": None, "precision": None, "recall": None,
                "gap": None, "flagged": False,
            }
            continue

        correct = int((y_true[mask] == y_pred[mask]).sum())
        class_acc = correct / class_count
        gap = class_acc - overall_acc
        is_flagged = class_acc < flag_threshold

        per_class[LABELS[c]] = {
            "count":     class_count,
            "correct":   correct,
            "accuracy":  round(class_acc, 4),
            "f1":        round(float(f1_scores[c]), 4),
            "precision": round(float(precisions[c]), 4),
            "recall":    round(float(recalls[c]), 4),
            "gap":       round(gap, 4),
            "flagged":   is_flagged,
        }
        if is_flagged:
            flagged.append(LABELS[c])

    report = {
        "architecture":    arch,
        "dataset":         "rafdb_test",
        "overall_accuracy":  round(overall_acc, 6),
        "overall_macro_f1":  round(overall_macro_f1, 6),
        "per_class":         per_class,
        "flagged_classes":   flagged,
        "flag_threshold":    round(flag_threshold, 4),
    }

    rp = get_bias_report_path(arch)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  BIAS AUDIT — {arch.upper()} | Overall Acc: {overall_acc:.4f} | Macro F1: {overall_macro_f1:.4f}")
    print(f"{'='*65}")
    print(f"  {'Emotion':<12} {'Count':>6} {'Acc':>7} {'F1':>7} {'Precision':>10} {'Recall':>8} {'Flag':>6}")
    print(f"  {'-'*63}")
    for label, m in per_class.items():
        if m["accuracy"] is None:
            continue
        flag_str = "⚠️" if m["flagged"] else "✅"
        print(f"  {label:<12} {m['count']:>6} {m['accuracy']:>7.4f} {m['f1']:>7.4f}"
              f" {m['precision']:>10.4f} {m['recall']:>8.4f} {flag_str:>6}")
    print(f"{'='*65}")
    if flagged:
        print(f"  ⚠️  Flagged classes (>10% below overall): {', '.join(flagged)}")
    else:
        print("  ✅ All classes within 10% of overall accuracy")
    print()

    keras.backend.clear_session()
    return report