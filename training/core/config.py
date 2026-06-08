# training/core/config.py
# Centralized hyperparameter and environment specifications for Emontic AI.
# Upgraded to enforce an intentional 224x224 high-density training token matrix.

import os
from pathlib import Path

# Reproducibility
SEED = 42

# Unified 7-class labels
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(LABELS)

# Paths Configuration
PROJECT_ROOT = Path(
    os.environ.get("EMONTIC_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)
# Ensure this points to the native path, NOT /mnt/d/
DATA_DIR = Path.home() / "emontic-datasets"
TRAINING_DIR = PROJECT_ROOT / "training"
SPLITS_DIR = TRAINING_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# RAF-DB Data Structures
RAFDB_DIR = DATA_DIR / "raf-db"
RAFDB_DATASET_DIR = RAFDB_DIR / "DATASET"
RAFDB_TRAIN_CSV = RAFDB_DIR / "train_labels.csv"
RAFDB_TEST_CSV = RAFDB_DIR / "test_labels.csv"
RAFDB_TRAIN_IMAGES = RAFDB_DATASET_DIR / "train"
RAFDB_TEST_IMAGES = RAFDB_DATASET_DIR / "test"

# AffectNet Data Structures
AFFECTNET_DIR = DATA_DIR / "affectnet"
AFFECTNET_LABELS_CSV = AFFECTNET_DIR / "labels.csv"
AFFECTNET_TRAIN_IMAGES = AFFECTNET_DIR / "Train"
AFFECTNET_TEST_IMAGES = AFFECTNET_DIR / "Test"

# Serialization Folders
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
LOG_DIR = TRAINING_DIR / "logs"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "models"

# Upgraded Architecture Context Specifications
ARCHITECTURES = {
    "v2s": {
        "name": "EfficientNetV2S",
        "input_size": (224, 224),  # Upgraded to preserve micro-expression tokens
        "batch_size": 64,
    },
    "v2m": {
        "name": "EfficientNetV2M",
        "input_size": (224, 224),  # Unified spatial layout scaling
        "batch_size": 32,
    },
}

DEFAULT_ARCH = "v2s"


def get_input_shape(arch: str) -> tuple:
    h, w = ARCHITECTURES[arch]["input_size"]
    return (h, w, 3)


def get_batch_size(arch: str) -> int:
    return ARCHITECTURES[arch]["batch_size"]


def get_checkpoint_dir(arch: str) -> Path:
    d = CHECKPOINT_DIR / arch
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_log_dir(arch: str, stage: str) -> Path:
    d = LOG_DIR / arch / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_export_dir(arch: str) -> Path:
    return SAVED_MODEL_DIR / f"emontic_ai_{arch}"


def get_calibration_json(arch: str) -> Path:
    return SAVED_MODEL_DIR / f"calibration_{arch}.json"


def get_eval_report_path(arch: str) -> Path:
    return SAVED_MODEL_DIR / f"eval_report_{arch}.json"


def get_bias_report_path(arch: str) -> Path:
    return SAVED_MODEL_DIR / f"bias_report_{arch}.json"


def get_confusion_matrix_path(arch: str) -> Path:
    return SAVED_MODEL_DIR / f"confusion_matrix_{arch}.png"


def get_split_file(dataset_name: str, split_name: str) -> Path:
    d = SPLITS_DIR / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split_name}.csv"


# RAF-DB label mapping (1-indexed → unified 0-indexed)
_RAFDB_TO_UNIFIED = {1: 6, 2: 2, 3: 1, 4: 3, 5: 5, 6: 0, 7: 4}


def convert_rafdb_label(raf_label: int) -> int:
    return _RAFDB_TO_UNIFIED[raf_label]


# AffectNet label mapping (string → unified 0-indexed)
_AFFECTNET_TO_UNIFIED = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "neutral": 4,
    "sad": 5,
    "sadness": 5,
    "surprise": 6,
}

AFFECTNET_EXCLUDED = {"contempt"}
AFFECTNET_MIN_CONFIDENCE = 0.00


def convert_affectnet_label(label_str: str):
    key = label_str.strip().lower()
    return None if key in AFFECTNET_EXCLUDED else _AFFECTNET_TO_UNIFIED.get(key)


# Training Hyperparameters
PRETRAIN_LR = 1e-4
PRETRAIN_LR_FINE = 2.5e-5
PRETRAIN_EPOCHS_FROZEN = 15
PRETRAIN_EPOCHS_FINE = 25
PRETRAIN_PATIENCE = 8
PRETRAIN_UNFREEZE_LAYERS = 60   # Safely maps to block6 boundaries via block-aware unfreezing
PRETRAIN_VAL_FRACTION = 0.05

FINETUNE_LR = 2e-5  
FINETUNE_EPOCHS = 40
FINETUNE_PATIENCE = 12
FINETUNE_UNFREEZE_LAYERS = 120  # Safely maps to block5 and block6 boundaries
RAFDB_VAL_FRACTION = 0.10

WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0

# Regularization Head Structure Configuration
HEAD_DROPOUT_1 = 0.4        
HEAD_DROPOUT_2 = 0.3        
HEAD_DROPOUT_3 = 0.2        
HEAD_DENSE_UNITS_1 = 512
HEAD_DENSE_UNITS_2 = 256

# Preprocessing Pipeline Contract
PREPROCESSING_NOTE = (
    "Images: float32 [0,255]. EfficientNetV2 include_preprocessing=True. "
    "Resize: 224x224 utilizing BICUBIC mappings in the input data pipeline."
)


def assert_preprocessing_contract(arch: str):
    assert arch in ARCHITECTURES
    shape = get_input_shape(arch)
    assert shape[2] == 3
    assert shape[0] == shape[1]
    return True