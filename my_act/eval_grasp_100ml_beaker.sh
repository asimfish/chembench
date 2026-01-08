#!/bin/bash

# Evaluation script for grasp_100ml_beaker task
# This script evaluates a trained ACT policy

# Configuration
TASK_NAME="sim_grasp_100ml_beaker"
CKPT_DIR="./checkpoints/grasp_100ml_beaker"
POLICY_CLASS="ACT"

# ACT Hyperparameters (must match training)
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
DIM_FEEDFORWARD=3200
BATCH_SIZE=8
SEED=0

# Evaluation options
ONSCREEN_RENDER=""  # Add "--onscreen_render" to visualize
TEMPORAL_AGG="--temporal_agg"  # Enable temporal aggregation for smoother actions

echo "=========================================="
echo "Evaluating ACT Policy for ${TASK_NAME}"
echo "=========================================="
echo "Checkpoint directory: ${CKPT_DIR}"
echo "Temporal aggregation: ${TEMPORAL_AGG}"
echo "=========================================="

# Run evaluation
python3 imitate_episodes.py \
    --eval \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight ${KL_WEIGHT} \
    --chunk_size ${CHUNK_SIZE} \
    --hidden_dim ${HIDDEN_DIM} \
    --batch_size ${BATCH_SIZE} \
    --dim_feedforward ${DIM_FEEDFORWARD} \
    --num_epochs 1 \
    --lr 1e-5 \
    --seed ${SEED} \
    ${TEMPORAL_AGG} \
    ${ONSCREEN_RENDER}

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${CKPT_DIR}"
echo "=========================================="

