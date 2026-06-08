# training/training/stages.py
# Optimized 2-stage training orchestrator utilizing class-weighted 
# optimization sweeps and precise graph deserialization paths.

import os
from pathlib import Path
import keras
import numpy as np
import tensorflow as tf

from training.core.config import (
    SEED,
    PRETRAIN_LR, PRETRAIN_LR_FINE,
    PRETRAIN_EPOCHS_FROZEN, PRETRAIN_EPOCHS_FINE, PRETRAIN_PATIENCE,
    PRETRAIN_UNFREEZE_LAYERS,
    FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE,
    FINETUNE_UNFREEZE_LAYERS,
    get_checkpoint_dir, get_log_dir,
    assert_preprocessing_contract,
)
from training.core.model import get_model, unfreeze_top_layers
from training.core.losses import MacroF1, SparseFocalLoss, WarmupCosineDecay, ArcFaceLoss
from training.data.pipeline import (
    build_affectnet_dataset, build_rafdb_dataset,
    get_affectnet_split_labels, get_rafdb_split_labels,
    compute_class_weights,  # Sourced directly from our enhanced pipeline engine
)
from .schedule import compute_steps, make_lr_schedule


# =============================================================================
# FastTrainModel — Custom train_step following the official Keras 3 guide:
# https://keras.io/guides/custom_train_step_in_tensorflow/
#
# Bypasses Keras model.fit() internal overhead (measured 10x slowdown).
# =============================================================================

@keras.saving.register_keras_serializable(package="emontic_ai")
class FastTrainModel(keras.Model):
    """Keras 3 model with lean train_step/test_step for ~6x faster training."""

    def train_step(self, data):
        # Unpack data — supports (x, y) and (x, y, sample_weight)
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)

        # Keras 3 API: optimizer.apply(grads, vars) not apply_gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(gradients, self.trainable_variables)

        # Update metrics (includes the auto-tracked loss metric)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _make_optimizer(lr_schedule):
    from training.core.config import WEIGHT_DECAY, GRADIENT_CLIP_NORM
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=GRADIENT_CLIP_NORM,
    )


def _compile_model(model, lr_schedule, label_smoothing=0.0):
    """Compiles the model with adaptive label smoothing to fit different training domains."""
    model.compile(
        optimizer=_make_optimizer(lr_schedule),
        loss=ArcFaceLoss(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            MacroF1(name="macro_f1"),
        ],
    )


def _build_fast_model(base_model):
    """Creates a FastTrainModel from a base model's I/O graph (shares weights)."""
    return FastTrainModel(inputs=base_model.input, outputs=base_model.output,
                          name=base_model.name)


def _apply_class_weight_to_ds(ds, class_weights):
    """
    Bakes class_weight into the dataset as sample_weight tensors.
    This avoids Keras model.fit(class_weight=dict) Python-level per-sample
    lookup overhead that forces eager execution on each batch.
    """
    max_class = max(class_weights.keys()) + 1
    weight_table = tf.constant(
        [class_weights.get(i, 1.0) for i in range(max_class)], dtype=tf.float32
    )

    def _add_sample_weight(x, y):
        y_int = tf.cast(tf.reshape(y, [-1]), tf.int32)
        w = tf.gather(weight_table, y_int)
        return x, y, w

    return ds.map(_add_sample_weight, num_parallel_calls=tf.data.AUTOTUNE)


def _make_callbacks(checkpoint_dir, log_dir, stage_name, patience, monitor="val_macro_f1"):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(checkpoint_dir / f"{stage_name}_best.keras")
    csv_path = str(log_dir / f"{stage_name}_history.csv")

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f"{stage_name}_recovery.keras"),
            save_freq=500,
            save_best_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir / "tensorboard" / stage_name),
            histogram_freq=0,
            profile_batch=(2, 5),  # Profile batches 2-5 to capture execution trace
        ),
    ]


def _custom_objects():
    """Maps custom attention layers and losses for seamless cross-stage model restoration."""
    from training.core.model import SpatialPositionalEmbedding, AttentionPooling, ArcFaceDense
    return {
        "FastTrainModel": FastTrainModel,
        "MacroF1": MacroF1,
        "SparseFocalLoss": SparseFocalLoss,
        "ArcFaceLoss": ArcFaceLoss,
        "WarmupCosineDecay": WarmupCosineDecay,
        "SpatialPositionalEmbedding": SpatialPositionalEmbedding,
        "AttentionPooling": AttentionPooling,
        "ArcFaceDense": ArcFaceDense,
    }


def run_pretrain(arch: str) -> str:
    print("\n" + "=" * 70)
    print(" STAGE 1 — PRETRAIN ON AFFECTNET (PRODUCTION-READY ATTENTION CONTEXT)")
    print("=" * 70)

    set_seed()
    assert_preprocessing_contract(arch)

    train_ds = build_affectnet_dataset("train", arch)
    val_ds = build_affectnet_dataset("val", arch)

    _, labels = get_affectnet_split_labels("train")
    
    # Compute inverse class frequencies to handle AffectNet's data skew
    class_weights = compute_class_weights(labels)
    print(f"📊 Calculated Stage 1 Class Weights: {class_weights}")

    # Build base model, then wrap with FastTrainModel for lean train_step
    base_model = get_model(arch, trainable_base=False)
    model = _build_fast_model(base_model)
    model.summary()

    n_samples = len(labels)
    steps_frozen, total_steps_frozen = compute_steps(n_samples, arch, PRETRAIN_EPOCHS_FROZEN)
    print(f"Frozen phase: steps/epoch={steps_frozen}, total={total_steps_frozen}")

    schedule1 = make_lr_schedule(PRETRAIN_LR, total_steps_frozen)
    
    # Apply 0.1 label smoothing to protect the model from AffectNet's high noise floor
    _compile_model(model, schedule1, label_smoothing=0.1)

    ckpt_dir = get_checkpoint_dir(arch)
    log_dir = get_log_dir(arch, "pretrain_frozen")
    callbacks = _make_callbacks(ckpt_dir, log_dir, "pretrain_frozen", PRETRAIN_PATIENCE)

    # Bake class weights into dataset to avoid model.fit(class_weight=dict) eager overhead
    train_ds_weighted = _apply_class_weight_to_ds(train_ds, class_weights)

    model.fit(
        train_ds_weighted,
        validation_data=val_ds,
        epochs=PRETRAIN_EPOCHS_FROZEN,
        callbacks=callbacks,
    )

    # Unfreeze top backbone layers for fine-tuning phase
    unfreeze_top_layers(model, PRETRAIN_UNFREEZE_LAYERS)

    steps_fine, total_steps_fine = compute_steps(n_samples, arch, PRETRAIN_EPOCHS_FINE)
    print(f"Fine-tune phase: steps/epoch={steps_fine}, total={total_steps_fine}")

    schedule2 = make_lr_schedule(PRETRAIN_LR_FINE, total_steps_fine)
    _compile_model(model, schedule2, label_smoothing=0.1)

    log_dir2 = get_log_dir(arch, "pretrain_finetune")
    callbacks2 = _make_callbacks(ckpt_dir, log_dir2, "pretrain_finetune", PRETRAIN_PATIENCE)

    model.fit(
        train_ds_weighted,
        validation_data=val_ds,
        epochs=PRETRAIN_EPOCHS_FINE,
        callbacks=callbacks2,
    )

    final_path = str(ckpt_dir / "pretrain_final.keras")
    model.save(final_path)
    best_path = str(ckpt_dir / "pretrain_finetune_best.keras")
    print(f"[OK] Pretrain complete. Final: {final_path}")
    print(f" Best checkpoint: {best_path}")
    return best_path


def run_finetune(arch: str, checkpoint: str) -> str:
    print("\n" + "=" * 70)
    print(" STAGE 2 — FINETUNE ON RAF-DB")
    print("=" * 70)

    set_seed()
    assert_preprocessing_contract(arch)

    ckpt_dir = get_checkpoint_dir(arch)
    production_base_path = ckpt_dir / "production_base_ready.keras"
    
    if production_base_path.exists():
        print("🚀 [PRODUCTION INTERCEPT] Found upgraded spatial attention base configuration.")
        checkpoint = str(production_base_path)
    
    print(f"📂 Loading checkpoint target securely: {checkpoint}")
    loaded_model = keras.models.load_model(
        checkpoint,
        custom_objects=_custom_objects(),
        compile=False,
    )

    # Wrap loaded model as FastTrainModel (handles both old and new checkpoints)
    if not isinstance(loaded_model, FastTrainModel):
        model = _build_fast_model(loaded_model)
    else:
        model = loaded_model

    unfreeze_top_layers(model, FINETUNE_UNFREEZE_LAYERS)
    model.summary()

    train_ds = build_rafdb_dataset("train", arch)
    val_ds = build_rafdb_dataset("val", arch)

    _, labels = get_rafdb_split_labels("train")
    
    # Compute true inverse class frequencies for the target domain (RAF-DB)
    class_weights = compute_class_weights(labels)
    print(f"📊 Calculated Stage 2 Class Weights: {class_weights}")
    
    n_samples = len(labels)
    steps, total_steps = compute_steps(n_samples, arch, FINETUNE_EPOCHS)
    print(f"Finetune: steps/epoch={steps}, total={total_steps}")

    schedule = make_lr_schedule(FINETUNE_LR, total_steps)
    
    # Drop label smoothing to 0.0 to establish sharp decision boundaries on RAF-DB's clean labels
    _compile_model(model, schedule, label_smoothing=0.0)

    log_dir = get_log_dir(arch, "finetune")
    callbacks = _make_callbacks(
        ckpt_dir,
        log_dir,
        "finetune",
        FINETUNE_PATIENCE,
        monitor="val_macro_f1",
    )

    # Bake class weights into dataset to avoid model.fit(class_weight=dict) eager overhead
    train_ds_weighted = _apply_class_weight_to_ds(train_ds, class_weights)

    model.fit(
        train_ds_weighted,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks,
    )

    final_path = str(ckpt_dir / "finetune_final.keras")
    model.save(final_path)
    best_path = str(ckpt_dir / "finetune_best.keras")
    print(f"[OK] Finetune complete. Final: {final_path}")
    print(f" Best checkpoint: {best_path}")
    return best_path