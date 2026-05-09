# training/evaluate.py
# Comprehensive evaluation for Emontic AI v2.
# Produces accuracy, F1, precision, recall, classification report, confusion matrix.

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR, NUM_CLASSES, EMOTION_LABELS, BATCH_SIZE, SEED,
)
from dataset import load_rafdb, load_affectnet

logger = logging.getLogger("emontic_ai.evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def evaluate_model(model, dataset, dataset_name="test"):
    """Run full evaluation on a dataset and print metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix, classification_report,
    )

    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"Evaluation Results — {dataset_name}")
    print(f"{'='*60}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}  ← PRIMARY METRIC")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Save confusion matrix as image
    try:
        save_confusion_matrix(cm, dataset_name)
    except Exception as e:
        logger.warning(f"Could not save confusion matrix image: {e}")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def save_confusion_matrix(cm, dataset_name):
    """Save confusion matrix as a PNG image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        title=f"Confusion Matrix — {dataset_name}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    save_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix_{dataset_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to .keras model file (defaults to best_stageB.keras)",
    )
    parser.add_argument(
        "--dataset", choices=["rafdb", "affectnet", "both"], default="rafdb",
        help="Which dataset to evaluate on",
    )
    args = parser.parse_args()

    # Find model
    if args.model and os.path.exists(args.model):
        model_path = args.model
    else:
        candidates = [
            os.path.join(CHECKPOINT_DIR, "stageB_final.keras"),
            os.path.join(CHECKPOINT_DIR, "best_stageB.keras"),
            os.path.join(CHECKPOINT_DIR, "stageA_final.keras"),
            os.path.join(CHECKPOINT_DIR, "best_stageA_phase2.keras"),
        ]
        model_path = None
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            print("No trained model found. Run train.py first.")
            sys.exit(1)

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if args.dataset in ("rafdb", "both"):
        try:
            test_ds, count, _ = load_rafdb("test", BATCH_SIZE)
            evaluate_model(model, test_ds, "RAF-DB Test")
        except Exception as e:
            logger.error(f"RAF-DB evaluation failed: {e}")

    if args.dataset in ("affectnet", "both"):
        try:
            test_ds, count, _ = load_affectnet("test", BATCH_SIZE)
            evaluate_model(model, test_ds, "AffectNet Test")
        except Exception as e:
            logger.error(f"AffectNet evaluation failed: {e}")


if __name__ == "__main__":
    main()
