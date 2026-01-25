#!/bin/bash
# ==============================================================================
# Knowledge Distillation Training Script
# ==============================================================================
#
# Trains a FastViT student model (ft6c96) using knowledge distillation
# from a larger teacher model (b28c512nbt).
#
# Usage:
#   # Set required environment variables
#   export DISTILL_TRAINDIR=/path/to/output
#   export DISTILL_DATADIR=/path/to/npz/data
#   export DISTILL_TEACHER_CHECKPOINT=/path/to/teacher.ckpt
#   ./train_distill.sh
#
# Required Environment Variables:
#   DISTILL_TRAINDIR          - Training output directory
#   DISTILL_DATADIR           - NPZ data directory (containing train/*.npz)
#   DISTILL_TEACHER_CHECKPOINT - Teacher model checkpoint path
#
# Optional Environment Variables:
#   KATAGO_PYTHON             - Python executable (default: python)
#
# ==============================================================================

set -e

# ==============================================================================
# ENVIRONMENT VARIABLES (REQUIRED)
# ==============================================================================

# Python executable (optional, defaults to 'python')
PYTHON_PATH="${KATAGO_PYTHON:-python}"

# Training output directory (required)
TRAINDIR="${DISTILL_TRAINDIR:?Error: Set DISTILL_TRAINDIR environment variable}"

# NPZ data directory (required)
DATADIR="${DISTILL_DATADIR:?Error: Set DISTILL_DATADIR environment variable}"

# Teacher model checkpoint (required)
TEACHER_CHECKPOINT="${DISTILL_TEACHER_CHECKPOINT:?Error: Set DISTILL_TEACHER_CHECKPOINT environment variable}"

# Student model architecture
STUDENT_MODEL="ft6c96"

# Board size
POS_LEN=19

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Batch size (reduce if running out of memory)
BATCH_SIZE=32

# Number of samples per epoch (30k for 2-hour run with 30 epochs)
SAMPLES_PER_EPOCH=30000

# Total epochs to train
MAX_EPOCHS=30

# Warmup epochs (linear LR ramp up)
WARMUP_EPOCHS=5

# ==============================================================================
# OPTIMIZER HYPERPARAMETERS
# ==============================================================================

# Peak learning rate
LR=0.001

# Weight decay for AdamW
WEIGHT_DECAY=0.05

# EMA decay rate (0.9995 is standard for 300 epochs)
EMA_DECAY=0.9995

# ==============================================================================
# DISTILLATION HYPERPARAMETERS
# ==============================================================================

# Distillation alpha: weight for soft (teacher) loss
# 0.0 = only hard labels, 1.0 = only distillation
# 0.5 is a balanced starting point
DISTILL_ALPHA=0.5

# Temperature for distillation (higher = softer distribution)
# 4.0 is common for classification distillation
TEMPERATURE=4.0

# Label smoothing for hard targets
LABEL_SMOOTHING=0.1

# ==============================================================================
# LOSS WEIGHTS
# ==============================================================================

# Policy loss weight
POLICY_WEIGHT=1.0

# Value loss weight
VALUE_WEIGHT=0.6

# Ownership loss weight
OWNERSHIP_WEIGHT=0.015

# ==============================================================================
# OTHER OPTIONS
# ==============================================================================

# Use mixed precision (FP16) - faster on supported GPUs
USE_FP16="-use-fp16"
# Uncomment below to disable FP16:
# USE_FP16=""

# Use SWA weights from teacher model
USE_TEACHER_SWA=""
# Uncomment to use SWA weights:
# USE_TEACHER_SWA="-use-teacher-swa"

# Save checkpoint every N epochs
SAVE_EVERY=1

# Log metrics every N batches
LOG_EVERY=100

# ==============================================================================
# RUN TRAINING
# ==============================================================================

# Change to python directory
cd "$(dirname "$0")/.."

echo "=============================================================="
echo "Knowledge Distillation Training"
echo "=============================================================="
echo "Student model:     ${STUDENT_MODEL}"
echo "Teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "Output directory:  ${TRAINDIR}"
echo "Data directory:    ${DATADIR}"
echo "Board size:        ${POS_LEN}x${POS_LEN}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Max epochs:        ${MAX_EPOCHS}"
echo "Peak LR:           ${LR}"
echo "Distill alpha:     ${DISTILL_ALPHA}"
echo "Temperature:       ${TEMPERATURE}"
echo "=============================================================="

# Create output directory if it doesn't exist
mkdir -p "${TRAINDIR}"

# Run training
"${PYTHON_PATH}" train_distill.py \
    -traindir "${TRAINDIR}" \
    -datadir "${DATADIR}" \
    -teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    -student-model "${STUDENT_MODEL}" \
    -pos-len ${POS_LEN} \
    -batch-size ${BATCH_SIZE} \
    -samples-per-epoch ${SAMPLES_PER_EPOCH} \
    -max-epochs ${MAX_EPOCHS} \
    -warmup-epochs ${WARMUP_EPOCHS} \
    -lr ${LR} \
    -weight-decay ${WEIGHT_DECAY} \
    -ema-decay ${EMA_DECAY} \
    -distill-alpha ${DISTILL_ALPHA} \
    -temperature ${TEMPERATURE} \
    -label-smoothing ${LABEL_SMOOTHING} \
    -policy-weight ${POLICY_WEIGHT} \
    -value-weight ${VALUE_WEIGHT} \
    -ownership-weight ${OWNERSHIP_WEIGHT} \
    -save-every ${SAVE_EVERY} \
    -log-every ${LOG_EVERY} \
    ${USE_FP16} \
    ${USE_TEACHER_SWA}

echo "=============================================================="
echo "Training completed!"
echo "=============================================================="
