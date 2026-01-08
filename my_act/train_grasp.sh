#!/bin/bash
# Training script for grasp task with Zarr dataset

# Set paths
ZARR_PATH="/share_data/liyufeng/code/chembench/data/final_real/data/grasp/part1/100ml玻璃烧杯.zarr"
DATASET_DIR="/share_data/liyufeng/code/chembench/act/data/grasp_100ml_beaker"
CKPT_DIR="/share_data/liyufeng/code/chembench/act/ckpts/grasp_100ml_beaker"

# Training parameters
BATCH_SIZE=8
NUM_EPOCHS=2000
LR=1e-5
SEED=0

# ACT parameters
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
DIM_FEEDFORWARD=3200

# Camera names (adjust based on your needs)
CAMERA_NAMES="head_camera chest_camera"

# Run training
python3 train_from_zarr.py \
    --zarr_path "$ZARR_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --ckpt_dir "$CKPT_DIR" \
    --camera_names $CAMERA_NAMES \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --seed $SEED \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --dim_feedforward $DIM_FEEDFORWARD

# Note: Add --skip_conversion flag if HDF5 files already exist and you want to skip conversion

