#!/bin/bash
# Exit on error
set -e

# Go to the directory where this script lives
cd "$(dirname "$0")"

STAGE="both"
if [ "$1" == "--stage" ] && [ -n "$2" ]; then
    STAGE="$2"
fi

echo "=== Emontic AI WSL Launcher ==="

# Set up virtual environment in Linux home directory
VENV_DIR="$HOME/.venvs/emontic_ai"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Linux virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating venv..."
source "$VENV_DIR/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements_train.txt

echo "Checking TensorFlow GPU visibility..."
python -c 'import tensorflow as tf; print("GPU Devices visible to TF:", tf.config.list_physical_devices("GPU"))'

echo "Starting training: Stage = $STAGE"
python train.py --stage "$STAGE"
