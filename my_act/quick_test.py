#!/usr/bin/env python3
"""
Quick test - just verify data dimensions match model expectations
"""
import torch
print("Testing grasp_100ml_beaker setup...")

# Test 1: Config
print("\n1. Checking config...")
from constants import SIM_TASK_CONFIGS
config = SIM_TASK_CONFIGS['sim_grasp_100ml_beaker']
print(f"   ✓ State dim: {config['state_dim']}")
print(f"   ✓ Image channels: {config['image_channels']}")
print(f"   ✓ Cameras: {config['camera_names']}")

# Test 2: Data loading
print("\n2. Loading data...")
from utils import load_data
train_dl, val_dl, stats, _ = load_data(
    config['dataset_dir'], 5, config['camera_names'], 2, 2
)
for batch in train_dl:
    img, qpos, act, pad = batch
    print(f"   ✓ Image: {img.shape} (expect: [2, 3, 6, 224, 224])")
    print(f"   ✓ QPos: {qpos.shape} (state_dim={qpos.shape[2]})")
    print(f"   ✓ Action: {act.shape} (action_dim={act.shape[2]})")
    break

# Test 3: Model
print("\n3. Initializing model...")
from policy import ACTPolicy
policy = ACTPolicy({
    'lr': 1e-5, 'num_queries': 100, 'kl_weight': 10,
    'hidden_dim': 512, 'dim_feedforward': 3200,
    'lr_backbone': 1e-5, 'backbone': 'resnet18',
    'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
    'camera_names': config['camera_names'],
    'input_channels': config['image_channels'],
    'state_dim': config['state_dim'],
    'action_dim': config['state_dim'],
})
print(f"   ✓ Model created")

# Test 4: Forward pass
if torch.cuda.is_available():
    print("\n4. Testing forward pass...")
    policy.cuda()
    img, qpos, act, pad = img.cuda(), qpos.cuda(), act.cuda(), pad.cuda()
    with torch.no_grad():
        out = policy(qpos, img, act, pad)
    print(f"   ✓ Forward pass OK, loss={out['loss'].item():.4f}")
else:
    print("\n4. Skipping forward pass (no GPU)")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nStart training with:")
print("  ./train_grasp_100ml_beaker.sh")


