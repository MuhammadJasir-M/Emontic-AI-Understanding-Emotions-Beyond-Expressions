# training/train.py
# Two-stage cross-dataset training orchestrator for Emontic AI v2.
#
# Stage A: AffectNet pretraining (broad emotional representation)
#   Phase 1: Frozen backbone, LR=3e-4
#   Phase 2: Unfreeze top 60 layers, LR=1e-5
#
# Stage B: RAF-DB fine-tuning (precision boundaries)
#   Unfreeze top 80 layers, LR=1e-5 with cosine decay
#
# Usage:
#   python train.py                          # Full pipeline (Stage A + B)
#   python train.py --stage a                # AffectNet only
#   python train.py --stage b                # RAF-DB only (requires Stage A checkpoint)
#   python train.py --stage b --checkpoint path/to/model.keras

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf

# Add training dir to path for config imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, BATCH_SIZE, NUM_CLASSES, CHECKPOINT_DIR, LOG_DIR,
    STAGE_A_PHASE1_LR, STAGE_A_PHASE1_EPOCHS,
    STAGE_A_PHASE2_LR, STAGE_A_PHASE2_EPOCHS,
    STAGE_A_UNFREEZE_LAYERS,
    STAGE_B_LR, STAGE_B_EPOCHS, STAGE_B_UNFREEZE_LAYERS, STAGE_B_PATIENCE,
    WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOP_PATIENCE,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, MIN_LR,
    EMOTION_LABELS,
)
from model import build_model, unfreeze_top_layers, get_model_summary
from dataset import load_affectnet, load_rafdb, get_class_weights

logger = logging.getLogger("emontic_ai.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logger.info(f"Global seed set to {seed}")


def setup_gpu():
    """Configure GPU if available, otherwise prepare for CPU training."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU(s) available: {[g.name for g in gpus]}")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (float16) enabled")
    else:
        logger.info("No GPU detected — running on CPU.")
        logger.info("Dataset is small (~27K images) — CPU training is feasible.")
        logger.info("Estimated time: Stage A ~1-2 hrs, Stage B ~20-30 min on modern CPU.")


def get_callbacks(stage, phase=""):
    """Build callback list for a training phase."""
    tag = f"{stage}_{phase}" if phase else stage
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{tag}.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, tag),
            histogram_freq=0,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"{tag}_history.csv"),
        ),
    ]
    return callbacks, checkpoint_path


def compile_model(model, learning_rate, weight_decay=WEIGHT_DECAY):
    """Compile model with AdamW + label-smoothed categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage A — AffectNet Pretraining
# ═══════════════════════════════════════════════════════════════════════════════

def train_stage_a():
    """
    Stage A: Learn generalized emotional representations from AffectNet.
    Phase 1: Frozen backbone (fast head training)
    Phase 2: Unfreeze top layers (fine-tune features)
    """
    logger.info("=" * 70)
    logger.info("STAGE A — AffectNet Pretraining")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading AffectNet dataset...")
    train_ds, train_count, train_labels = load_affectnet("train", BATCH_SIZE)
    val_ds, val_count, _ = load_affectnet("test", BATCH_SIZE)
    logger.info(f"Train: {train_count} | Val: {val_count}")

    # Compute class weights
    class_weights = get_class_weights(train_labels) if train_labels else None

    # ── Phase 1: Frozen backbone ─────────────────────────────────────────
    logger.info("\n--- Phase 1: Frozen Backbone ---")
    model = build_model(num_classes=NUM_CLASSES, trainable_base=False)
    model = compile_model(model, STAGE_A_PHASE1_LR)
    get_model_summary(model)

    callbacks_p1, ckpt_p1 = get_callbacks("stageA", "phase1")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE_A_PHASE1_EPOCHS,
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1,
    )

    # ── Phase 2: Partial unfreeze ────────────────────────────────────────
    logger.info("\n--- Phase 2: Unfreeze Top Layers ---")
    unfreeze_top_layers(model, STAGE_A_UNFREEZE_LAYERS)
    model = compile_model(model, STAGE_A_PHASE2_LR)
    get_model_summary(model)

    callbacks_p2, ckpt_p2 = get_callbacks("stageA", "phase2")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE_A_PHASE2_EPOCHS,
        callbacks=callbacks_p2,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final Stage A model
    final_path = os.path.join(CHECKPOINT_DIR, "stageA_final.keras")
    model.save(final_path)
    logger.info(f"Stage A model saved to {final_path}")

    return model, final_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage B — RAF-DB Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

def train_stage_b(checkpoint_path=None):
    """
    Stage B: Sharpen emotional discrimination on RAF-DB.
    Starts from Stage A checkpoint, unfreezes more layers, lower LR.
    """
    logger.info("=" * 70)
    logger.info("STAGE B — RAF-DB Fine-Tuning")
    logger.info("=" * 70)

    # Load model from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        # Try default Stage A checkpoint
        default_ckpt = os.path.join(CHECKPOINT_DIR, "stageA_final.keras")
        if os.path.exists(default_ckpt):
            logger.info(f"Loading Stage A model from: {default_ckpt}")
            model = tf.keras.models.load_model(default_ckpt)
        else:
            logger.warning("No Stage A checkpoint found. Building fresh model.")
            model = build_model(num_classes=NUM_CLASSES, trainable_base=False)

    # Load data
    logger.info("Loading RAF-DB dataset...")
    train_ds, train_count, train_labels = load_rafdb("train", BATCH_SIZE)
    val_ds, val_count, _ = load_rafdb("test", BATCH_SIZE)
    logger.info(f"Train: {train_count} | Val: {val_count}")

    # Compute class weights
    class_weights = get_class_weights(train_labels) if train_labels else None

    # Unfreeze more layers for fine-tuning
    unfreeze_top_layers(model, STAGE_B_UNFREEZE_LAYERS)

    # Cosine decay LR schedule
    steps_per_epoch = train_count // BATCH_SIZE
    total_steps = steps_per_epoch * STAGE_B_EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=STAGE_B_LR,
        decay_steps=total_steps,
        alpha=MIN_LR,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )
    get_model_summary(model)

    # Custom callbacks with tighter patience for fine-tuning
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_stageB.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=STAGE_B_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, "stageB"),
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, "stageB_history.csv"),
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE_B_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "stageB_final.keras")
    model.save(final_path)
    logger.info(f"Stage B model saved to {final_path}")

    return model, final_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Emontic AI v2 — Training Pipeline"
    )
    parser.add_argument(
        "--stage", choices=["a", "b", "both"], default="both",
        help="Training stage: 'a' (AffectNet), 'b' (RAF-DB), 'both' (default)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for Stage B (overrides auto-detection)",
    )
    args = parser.parse_args()

    set_seed()
    setup_gpu()

    if args.stage in ("a", "both"):
        model, ckpt = train_stage_a()
        if args.stage == "both":
            args.checkpoint = ckpt

    if args.stage in ("b", "both"):
        train_stage_b(checkpoint_path=args.checkpoint)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
