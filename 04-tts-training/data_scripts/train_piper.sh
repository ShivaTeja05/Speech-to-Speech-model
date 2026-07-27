#!/bin/bash
set -e

# Use specific python interpreter where piper is installed
PYTHON_EXEC="/Users/gaddamshivateja/miniconda3/bin/python"

# Configuration
DATASET_DIR="DATASET FOR TAMIL"
EXTRACT_SCRIPT="extract_dataset.py"
MODEL_NAME="tamil_model"
MAX_EPOCHS=1
BATCH_SIZE=4 # Adjust based on VRAM
QUALITY="medium"

# 0. Environment Setup
echo "Setting up environment..."
# Assuming we are running this from the root of the workspace
# and piper is cloned in ./piper

export PYTHONPATH=$PYTHONPATH:$(pwd)/piper/src/python_run

# 1. Extract Dataset
echo "Extracting dataset..."
$PYTHON_EXEC extract_dataset.py

# 2. Preprocess
echo "Preprocessing data..."
# Use piper's preprocessing
# We need to ensure we have the correct config or allow piper to generate it
# For now, let's assume we use default English process but with our data
# Actually, for a new language, we need to be careful with phonemization.
# Since we have espeak-ng installed, and it supports 'ta' (Tamil), we should use that.

# We need to create a config.json for training if it doesn't exist.
# But piper_train.preprocess can creating it?
# Let's try running preprocess command.

$PYTHON_EXEC -m piper_train.preprocess \
  --language ta \
  --input-dir "${DATASET_DIR}/extracted_wavs" \
  --output-dir "${DATASET_DIR}/training_dir" \
  --dataset-format ljspeech \
  --sample-rate 22050

# 3. Train
echo "Starting training for ${MAX_EPOCHS} epoch(s)..."
$PYTHON_EXEC -m piper_train \
  --dataset-dir "${DATASET_DIR}/training_dir" \
  --accelerator 'auto' \
  --devices 'auto' \
  --batch-size ${BATCH_SIZE} \
  --validation-split 0.0 \
  --max-epochs ${MAX_EPOCHS} \
  --quality ${QUALITY} \
  --checkpoint-epochs 1 \
  --precision 32

echo "Training complete!"
