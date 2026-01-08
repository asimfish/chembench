#!/bin/bash

# Training script for grasp_100ml_beaker task
# This script trains an ACT policy for grasping a 100ml beaker

# Configuration
TASK_NAME="sim_grasp_100ml_beaker"
CKPT_DIR="./checkpoints/grasp_100ml_beaker"
POLICY_CLASS="ACT"

# ACT Hyperparameters
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
DIM_FEEDFORWARD=3200
BATCH_SIZE=8
NUM_EPOCHS=2000
LR=1e-5
SEED=0

# Optional: Temporal aggregation for smoother actions during evaluation
# TEMPORAL_AGG="--temporal_agg"
TEMPORAL_AGG=""

# Wandb Configuration
USE_WANDB=true  # Set to false to disable wandb logging
WANDB_PROJECT="chembench-act"  # Change this to your wandb project name
WANDB_RUN_NAME="grasp_100ml_beaker_exp1"  # Change this for each experiment run

echo "=========================================="
echo "Training ACT Policy for ${TASK_NAME}"
echo "=========================================="
echo "Checkpoint directory: ${CKPT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Number of epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LR}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Hidden dim: ${HIDDEN_DIM}"
if [ "$USE_WANDB" = true ]; then
    echo "Wandb enabled: true"
    echo "Wandb project: ${WANDB_PROJECT}"
    echo "Wandb run name: ${WANDB_RUN_NAME}"
else
    echo "Wandb enabled: false"
fi
echo "=========================================="

# Create checkpoint directory
mkdir -p ${CKPT_DIR}

# Build wandb arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_NAME}"
fi

# Run training
python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight ${KL_WEIGHT} \
    --chunk_size ${CHUNK_SIZE} \
    --hidden_dim ${HIDDEN_DIM} \
    --batch_size ${BATCH_SIZE} \
    --dim_feedforward ${DIM_FEEDFORWARD} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --seed ${SEED} \
    ${TEMPORAL_AGG} \
    ${WANDB_ARGS}

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${CKPT_DIR}"
echo "=========================================="

