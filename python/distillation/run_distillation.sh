#!/bin/bash
# Knowledge Distillation Pipeline for KataGo
# Train a small student model to match a larger teacher model

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KATAGO_PYTHON_DIR="$(dirname "$SCRIPT_DIR")"
KATAGO_DIR="$(dirname "$KATAGO_PYTHON_DIR")"

# Teacher model
TEACHER_CHECKPOINT="${KATAGO_DIR}/models/b18c384nbt-humanv0.ckpt"

# Student model
STUDENT_MODEL_KIND="b5c192nbt-fson-mish-rvglr-bnh-meta"

# Directories
DISTILL_DIR="${SCRIPT_DIR}"
POSITIONS_DIR="${DISTILL_DIR}/positions"
TARGETS_DIR="${DISTILL_DIR}/targets"
DATA_DIR="${DISTILL_DIR}/data"
TRAINING_DIR="${DISTILL_DIR}/training_output"
EXPORT_DIR="${DISTILL_DIR}/exported"

# Parameters
NUM_POSITIONS=10000
BOARD_SIZE=19
POS_LEN=19
BATCH_SIZE=32
DEVICE="mps"  # mps for Apple Silicon, cuda for NVIDIA, cpu for fallback

# Training parameters
TRAIN_BATCH_SIZE=32
SAMPLES_PER_EPOCH=5000
MAX_EPOCHS=10

echo "============================================"
echo "KataGo Knowledge Distillation Pipeline"
echo "============================================"
echo ""
echo "Teacher: ${TEACHER_CHECKPOINT}"
echo "Student: ${STUDENT_MODEL_KIND}"
echo "Device: ${DEVICE}"
echo ""

# Step 1: Generate positions
echo "Step 1: Generating ${NUM_POSITIONS} random positions..."
mkdir -p "${POSITIONS_DIR}"
python "${SCRIPT_DIR}/generate_positions.py" \
    --num-positions "${NUM_POSITIONS}" \
    --board-size "${BOARD_SIZE}" \
    --pos-len "${POS_LEN}" \
    --output-dir "${POSITIONS_DIR}" \
    --batch-size 1024 \
    --seed 42
echo "Done generating positions."
echo ""

# Step 2: Run teacher inference
echo "Step 2: Running teacher model inference..."
mkdir -p "${TARGETS_DIR}"
python "${SCRIPT_DIR}/teacher_inference.py" \
    --checkpoint "${TEACHER_CHECKPOINT}" \
    --positions-dir "${POSITIONS_DIR}" \
    --output-dir "${TARGETS_DIR}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --pos-len "${POS_LEN}"
echo "Done with teacher inference."
echo ""

# Step 3: Create training NPZ files
echo "Step 3: Creating training NPZ files..."
mkdir -p "${DATA_DIR}"
python "${SCRIPT_DIR}/create_training_npz.py" \
    --positions-dir "${POSITIONS_DIR}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${DATA_DIR}" \
    --pos-len "${POS_LEN}" \
    --val-split 0.1
echo "Done creating training data."
echo ""

# Step 4: Train student model
echo "Step 4: Training student model..."
mkdir -p "${TRAINING_DIR}"
cd "${KATAGO_PYTHON_DIR}"
python train.py \
    -traindir "${TRAINING_DIR}" \
    -datadir "${DATA_DIR}" \
    -pos-len "${POS_LEN}" \
    -batch-size "${TRAIN_BATCH_SIZE}" \
    -model-kind "${STUDENT_MODEL_KIND}" \
    -samples-per-epoch "${SAMPLES_PER_EPOCH}" \
    -max-epochs-this-instance "${MAX_EPOCHS}" \
    -lr-scale 1.0 \
    -soft-policy-weight-scale 8.0 \
    -no-export
echo "Done training student model."
echo ""

# Step 5: Export model to .bin.gz
echo "Step 5: Exporting model to .bin.gz..."
mkdir -p "${EXPORT_DIR}"
python "${KATAGO_PYTHON_DIR}/export_model_pytorch.py" \
    -checkpoint "${TRAINING_DIR}/checkpoint.ckpt" \
    -export-dir "${EXPORT_DIR}" \
    -model-name "b5c192nbt-distilled-v1" \
    -filename-prefix "b5c192nbt-distilled"

# Compress the .bin file
if [ -f "${EXPORT_DIR}/b5c192nbt-distilled.bin" ]; then
    gzip -f "${EXPORT_DIR}/b5c192nbt-distilled.bin"
    echo "Created ${EXPORT_DIR}/b5c192nbt-distilled.bin.gz"
fi

# Copy to test models directory
TEST_MODELS_DIR="${KATAGO_DIR}/cpp/tests/models"
if [ -d "${TEST_MODELS_DIR}" ]; then
    cp "${EXPORT_DIR}/b5c192nbt-distilled.bin.gz" "${TEST_MODELS_DIR}/"
    echo "Copied to ${TEST_MODELS_DIR}/"
fi

echo ""
echo "============================================"
echo "Distillation Complete!"
echo "============================================"
echo "Student model: ${EXPORT_DIR}/b5c192nbt-distilled.bin.gz"
echo ""
