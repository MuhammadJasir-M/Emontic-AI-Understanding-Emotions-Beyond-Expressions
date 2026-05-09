# training/config.py
# Central training configuration for Emontic AI v2.
# All hyperparameters, paths, and label mappings live here.

import os
import glob

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _find_kaggle_cache(dataset_slug):
    """Find a dataset in the Kaggle cache directory."""
    cache_base = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets")
    pattern = os.path.join(cache_base, dataset_slug, "versions", "*")
    versions = sorted(glob.glob(pattern))
    if versions:
        return versions[-1]  # latest version
    return None


def _resolve_affectnet_dir():
    """Resolve AffectNet directory: env var → project data → Kaggle cache."""
    if os.environ.get("AFFECTNET_PATH"):
        return os.environ["AFFECTNET_PATH"]
    # Check project data dir
    local = os.path.join(DATA_DIR, "affectnet")
    for sub in ["Train", "train", "Training"]:
        if os.path.isdir(os.path.join(local, sub)):
            return local
    # Check Kaggle cache
    cached = _find_kaggle_cache("mstjebashazida/affectnet")
    if cached:
        # Handle nested "archive (3)" or similar subfolder
        for item in os.listdir(cached):
            item_path = os.path.join(cached, item)
            if os.path.isdir(item_path):
                for sub in ["Train", "train"]:
                    if os.path.isdir(os.path.join(item_path, sub)):
                        return item_path
        return cached
    return local  # fallback


def _resolve_rafdb_dir():
    """Resolve RAF-DB directory: env var → project data → Kaggle cache."""
    if os.environ.get("RAFDB_PATH"):
        return os.environ["RAFDB_PATH"]
    # Check project data dir
    local = os.path.join(DATA_DIR, "raf-db")
    for sub in ["train", "Train", "DATASET", "EmoLabel"]:
        if os.path.isdir(os.path.join(local, sub)):
            # If the repository contains a nested DATASET folder (Kaggle mirror),
            # prefer returning that subfolder so loaders find train/test inside it.
            if sub == "DATASET":
                return os.path.join(local, "DATASET")
            return local
    # Check Kaggle cache
    cached = _find_kaggle_cache("shuvoalok/raf-db-dataset")
    if cached:
        # Handle nested "DATASET" subfolder
        dataset_sub = os.path.join(cached, "DATASET")
        if os.path.isdir(dataset_sub):
            return dataset_sub
        return cached
    return local  # fallback


AFFECTNET_DIR = _resolve_affectnet_dir()
RAFDB_DIR = _resolve_rafdb_dir()
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "training", "logs")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "saved_model", "emontic_ai")
BACKEND_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "emontic_ai")

# ── Emotion Classes (Target Order) ──────────────────────────────────────────
# This is the canonical label order used everywhere: training, inference, API.
NUM_CLASSES = 7
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── AffectNet Label Mapping ──────────────────────────────────────────────────
# AffectNet folder/class structure (8 classes):
#   0=Angry, 1=Contempt, 2=Disgust, 3=Fear, 4=Happy, 5=Neutral, 6=Sad, 7=Surprise
# We drop Contempt (index 1) and remap to our 7-class target order:
#   AffectNet → Target:  0→0(Angry), 2→1(Disgust), 3→2(Fear), 4→3(Happy),
#                         5→4(Neutral), 6→5(Sad), 7→6(Surprise)
AFFECTNET_LABEL_MAP = {
    0: 0,   # Angry → Angry
    # 1 = Contempt → EXCLUDED
    2: 1,   # Disgust → Disgust
    3: 2,   # Fear → Fear
    4: 3,   # Happy → Happy
    5: 4,   # Neutral → Neutral
    6: 5,   # Sad → Sad
    7: 6,   # Surprise → Surprise
}
AFFECTNET_EXCLUDE_CLASSES = {1}  # Contempt

# ── RAF-DB Label Mapping ────────────────────────────────────────────────────
# RAF-DB annotation (1-indexed):
#   1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
# Remap to 0-indexed target order:
RAFDB_LABEL_MAP = {
    1: 6,   # Surprise → Surprise
    2: 2,   # Fear → Fear
    3: 1,   # Disgust → Disgust
    4: 3,   # Happiness → Happy
    5: 5,   # Sadness → Sad
    6: 0,   # Anger → Angry
    7: 4,   # Neutral → Neutral
}

# ── Input Configuration ─────────────────────────────────────────────────────
INPUT_SHAPE = (224, 224, 3)
INPUT_SIZE = (224, 224)

# ── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 32
SEED = 42

# Stage A — AffectNet Pretraining
STAGE_A_PHASE1_LR = 3e-4
STAGE_A_PHASE1_EPOCHS = 20
STAGE_A_PHASE2_LR = 1e-5
STAGE_A_PHASE2_EPOCHS = 30
STAGE_A_UNFREEZE_LAYERS = 60

# Stage B — RAF-DB Fine-Tuning
STAGE_B_LR = 1e-5
STAGE_B_EPOCHS = 40
STAGE_B_UNFREEZE_LAYERS = 80
STAGE_B_PATIENCE = 7

# Shared
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT_DENSE_512 = 0.4
DROPOUT_DENSE_256 = 0.3
EARLY_STOP_PATIENCE = 10
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 4
MIN_LR = 1e-7

# ── Confidence / Inference ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
