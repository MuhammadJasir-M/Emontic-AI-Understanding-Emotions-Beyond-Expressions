# training/download_data.py
# Automatic dataset downloader for Emontic AI v2.
# Downloads AffectNet from Kaggle. RAF-DB requires manual download.
#
# Usage:
#   pip install kagglehub
#   python download_data.py
#
# Prerequisites:
#   Kaggle API credentials at ~/.kaggle/kaggle.json
#   Get yours from: https://www.kaggle.com/settings → API → Create New Token

import os
import sys
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AFFECTNET_DIR, RAFDB_DIR, DATA_DIR

logger = logging.getLogger("emontic_ai.download")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        logger.info(f"Kaggle credentials found: {kaggle_json}")
        return True

    # Also check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Kaggle credentials found in environment variables")
        return True

    logger.error(
        "Kaggle API credentials not found!\n"
        "To set up:\n"
        "  1. Go to https://www.kaggle.com/settings\n"
        "  2. Scroll to 'API' section → Click 'Create New Token'\n"
        "  3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
        "  OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    )
    return False


def download_affectnet():
    """Download AffectNet dataset from Kaggle."""
    logger.info("=" * 60)
    logger.info("Downloading AffectNet from Kaggle...")
    logger.info("Dataset: mstjebashazida/affectnet")
    logger.info("=" * 60)

    # Check if already downloaded
    if os.path.isdir(AFFECTNET_DIR):
        train_dir = os.path.join(AFFECTNET_DIR, "train")
        train_dir_cap = os.path.join(AFFECTNET_DIR, "Train")
        if os.path.isdir(train_dir) or os.path.isdir(train_dir_cap):
            logger.info("AffectNet already exists — skipping download")
            return True

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        return False

    if not check_kaggle_credentials():
        return False

    try:
        # Download via kagglehub
        logger.info("Starting download (this may take 10-30 minutes)...")
        path = kagglehub.dataset_download("mstjebashazida/affectnet")
        logger.info(f"Downloaded to: {path}")

        # Move to our data directory
        os.makedirs(DATA_DIR, exist_ok=True)

        # Check what's in the downloaded path
        contents = os.listdir(path)
        logger.info(f"Download contents: {contents}")

        # If the download contains a subfolder with the actual data, use that
        source = path
        for item in contents:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                sub_contents = os.listdir(item_path)
                # Check if this subfolder has Train/Test structure
                if any(d.lower() in ("train", "test", "val") for d in sub_contents):
                    source = item_path
                    break

        # Copy/move to target directory
        if os.path.abspath(source) != os.path.abspath(AFFECTNET_DIR):
            if os.path.exists(AFFECTNET_DIR):
                shutil.rmtree(AFFECTNET_DIR)
            logger.info(f"Copying to {AFFECTNET_DIR}...")
            shutil.copytree(source, AFFECTNET_DIR)

        # Verify structure
        for name in ["Train", "train", "Training"]:
            if os.path.isdir(os.path.join(AFFECTNET_DIR, name)):
                logger.info(f"AffectNet download complete! Train dir: {name}")
                # Count images
                total = 0
                for root, dirs, files in os.walk(os.path.join(AFFECTNET_DIR, name)):
                    total += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                logger.info(f"Total training images found: {total}")
                return True

        # If no standard train directory found, list what we have
        logger.warning("Downloaded but couldn't find Train directory. Contents:")
        for item in os.listdir(AFFECTNET_DIR):
            item_path = os.path.join(AFFECTNET_DIR, item)
            if os.path.isdir(item_path):
                count = len(os.listdir(item_path))
                logger.info(f"  {item}/ ({count} items)")
            else:
                logger.info(f"  {item}")
        return True

    except Exception as e:
        logger.error(f"AffectNet download failed: {e}")
        return False


def check_rafdb():
    """Check if RAF-DB is available (requires manual download)."""
    logger.info("=" * 60)
    logger.info("Checking RAF-DB dataset...")
    logger.info("=" * 60)

    # Check for annotation file
    annotation_paths = [
        os.path.join(RAFDB_DIR, "EmoLabel", "list_patition_label.txt"),
        os.path.join(RAFDB_DIR, "list_patition_label.txt"),
        os.path.join(RAFDB_DIR, "basic", "EmoLabel", "list_patition_label.txt"),
    ]

    for path in annotation_paths:
        if os.path.isfile(path):
            logger.info(f"RAF-DB annotations found: {path}")
            return True

    # Check for folder-based structure
    for split in ["train", "Train"]:
        split_dir = os.path.join(RAFDB_DIR, split)
        if os.path.isdir(split_dir):
            logger.info(f"RAF-DB folder structure found: {split_dir}")
            return True

    logger.warning(
        "RAF-DB dataset not found!\n"
        "RAF-DB requires manual download:\n"
        "  1. Visit: http://www.whdeng.cn/RAF/model1.html\n"
        "  2. Request access with a university/institutional email\n"
        "  3. Download and extract to: " + RAFDB_DIR + "\n"
        "\n"
        "Note: Stage A (AffectNet pretraining) can run without RAF-DB.\n"
        "      Stage B (fine-tuning) requires RAF-DB."
    )
    return False


def main():
    logger.info("Emontic AI v2 — Dataset Setup")
    logger.info(f"Data directory: {DATA_DIR}")
    print()

    # Download AffectNet
    affectnet_ok = download_affectnet()
    print()

    # Check RAF-DB
    rafdb_ok = check_rafdb()
    print()

    # Summary
    logger.info("=" * 60)
    logger.info("Dataset Status Summary")
    logger.info("=" * 60)
    logger.info(f"  AffectNet: {'✓ Ready' if affectnet_ok else '✗ Not available'}")
    logger.info(f"  RAF-DB:    {'✓ Ready' if rafdb_ok else '✗ Not available (manual download required)'}")
    print()

    if affectnet_ok and not rafdb_ok:
        logger.info("You can start Stage A training now (AffectNet only):")
        logger.info("  python train.py --stage a")
    elif affectnet_ok and rafdb_ok:
        logger.info("Both datasets ready! You can run full training:")
        logger.info("  python train.py --stage both")
    else:
        logger.info("Please download the datasets before training.")


if __name__ == "__main__":
    main()
