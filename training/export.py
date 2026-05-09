# training/export.py
# Export trained model to SavedModel format for deployment.

import os
import sys
import shutil
import argparse
import logging
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHECKPOINT_DIR, EXPORT_DIR, BACKEND_MODEL_DIR, INPUT_SHAPE

logger = logging.getLogger("emontic_ai.export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def export_savedmodel(model, export_path):
    """Export model as TensorFlow SavedModel."""
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    model.export(export_path)
    
    logger.info(f"SavedModel exported to: {export_path}")

    # Verify the exported SavedModel with a SavedModel-native loader.
    loaded = tf.keras.layers.TFSMLayer(export_path, call_endpoint="serving_default")
    dummy = tf.random.normal((1, *INPUT_SHAPE))
    output = loaded(dummy)
    if isinstance(output, dict):
        output = next(iter(output.values()))
    logger.info(f"Verification — output shape: {output.shape} (expected: (1, 7))")

def export_tflite(model, output_path):
    """Export model as TFLite with default optimizations."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        logger.info(f"TFLite model exported to: {output_path} ({size_mb:.1f} MB)")
    except Exception as exc:
        logger.warning(f"TFLite export skipped: {exc}")


def copy_to_backend(export_path, backend_path):
    """Copy exported model to backend's models directory."""
    if os.path.exists(backend_path):
        shutil.rmtree(backend_path)
    shutil.copytree(export_path, backend_path)
    logger.info(f"Model copied to backend: {backend_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to .keras model file",
    )
    parser.add_argument(
        "--tflite", action="store_true",
        help="Also export TFLite model",
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Don't copy to backend directory",
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

    # Export SavedModel
    export_savedmodel(model, EXPORT_DIR)

    # Export TFLite (optional)
    if args.tflite:
        tflite_path = os.path.join(
            os.path.dirname(EXPORT_DIR), "emontic_ai.tflite"
        )
        export_tflite(model, tflite_path)

    # Copy to backend
    if not args.no_copy:
        copy_to_backend(EXPORT_DIR, BACKEND_MODEL_DIR)

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
