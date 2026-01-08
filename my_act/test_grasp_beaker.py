#!/usr/bin/env python3
"""
Test script for grasp_100ml_beaker training setup
Tests: 6-channel images, 13-dim state/action, dataset loading, model initialization
"""

import torch
import numpy as np
import sys
import os

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_success(msg):
    print(f"✓ {msg}")

def print_error(msg):
    print(f"✗ {msg}")

def print_info(msg):
    print(f"  {msg}")


def test_task_config():
    """Test 1: Verify task configuration"""
    print_header("Test 1: Task Configuration")
    
    try:
        from constants import SIM_TASK_CONFIGS
        
        task_name = 'sim_grasp_100ml_beaker'
        if task_name not in SIM_TASK_CONFIGS:
            print_error(f"Task '{task_name}' not found in SIM_TASK_CONFIGS")
            return False
        
        config = SIM_TASK_CONFIGS[task_name]
        print_success(f"Task '{task_name}' found")
        
        # Check required fields
        required_fields = ['dataset_dir', 'num_episodes', 'episode_len', 'camera_names']
        for field in required_fields:
            if field in config:
                print_info(f"{field}: {config[field]}")
            else:
                print_error(f"Missing field: {field}")
                return False
        
        # Check new fields
        if 'image_channels' in config:
            print_info(f"image_channels: {config['image_channels']}")
        else:
            print_error("Missing 'image_channels' field")
            return False
            
        if 'state_dim' in config:
            print_info(f"state_dim: {config['state_dim']}")
        else:
            print_error("Missing 'state_dim' field")
            return False
        
        print_success("All required fields present")
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test 2: Load dataset and check dimensions"""
    print_header("Test 2: Dataset Loading")
    
    try:
        from constants import SIM_TASK_CONFIGS
        from utils import load_data
        
        task_name = 'sim_grasp_100ml_beaker'
        config = SIM_TASK_CONFIGS[task_name]
        
        dataset_dir = config['dataset_dir']
        num_episodes = min(config['num_episodes'], 10)  # Load only 10 episodes for testing
        camera_names = config['camera_names']
        batch_size = 2
        
        print_info(f"Loading {num_episodes} episodes from {dataset_dir}")
        print_info(f"Camera names: {camera_names}")
        
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir, num_episodes, camera_names, batch_size, batch_size
        )
        
        print_success("Dataset loaded successfully")
        print_info(f"Train batches: {len(train_dataloader)}")
        print_info(f"Val batches: {len(val_dataloader)}")
        print_info(f"Stats keys: {list(stats.keys())}")
        
        # Check one batch
        for batch in train_dataloader:
            image_data, qpos_data, action_data, is_pad = batch
            
            print_success("Batch data retrieved")
            print_info(f"Image shape: {image_data.shape}")
            print_info(f"  Expected: [batch, num_cameras, channels, height, width]")
            print_info(f"  Got: [{image_data.shape[0]}, {image_data.shape[1]}, {image_data.shape[2]}, {image_data.shape[3]}, {image_data.shape[4]}]")
            
            print_info(f"QPos shape: {qpos_data.shape}")
            print_info(f"  Expected: [batch, seq_len, state_dim]")
            
            print_info(f"Action shape: {action_data.shape}")
            print_info(f"  Expected: [batch, seq_len, action_dim]")
            
            print_info(f"Is_pad shape: {is_pad.shape}")
            
            # Verify dimensions
            expected_cameras = len(camera_names)
            expected_channels = config.get('image_channels', 3)
            expected_state_dim = config.get('state_dim', 14)
            
            if image_data.shape[1] != expected_cameras:
                print_error(f"Camera count mismatch: expected {expected_cameras}, got {image_data.shape[1]}")
                return False
            
            if image_data.shape[2] != expected_channels:
                print_error(f"Channel count mismatch: expected {expected_channels}, got {image_data.shape[2]}")
                return False
            
            if qpos_data.shape[2] != expected_state_dim:
                print_error(f"State dim mismatch: expected {expected_state_dim}, got {qpos_data.shape[2]}")
                return False
            
            if action_data.shape[2] != expected_state_dim:
                print_error(f"Action dim mismatch: expected {expected_state_dim}, got {action_data.shape[2]}")
                return False
            
            print_success(f"All dimensions correct: {expected_cameras} cameras, {expected_channels} channels, {expected_state_dim} state/action dim")
            break
        
        return True
        
    except Exception as e:
        print_error(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test 3: Initialize ACT model with correct dimensions"""
    print_header("Test 3: Model Initialization")
    
    try:
        from constants import SIM_TASK_CONFIGS
        from policy import ACTPolicy
        
        task_name = 'sim_grasp_100ml_beaker'
        config = SIM_TASK_CONFIGS[task_name]
        
        camera_names = config['camera_names']
        image_channels = config.get('image_channels', 3)
        state_dim = config.get('state_dim', 14)
        
        print_info(f"Initializing ACT model...")
        print_info(f"  Cameras: {len(camera_names)}")
        print_info(f"  Image channels: {image_channels}")
        print_info(f"  State dim: {state_dim}")
        
        policy_config = {
            'lr': 1e-5,
            'num_queries': 100,
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': camera_names,
            'input_channels': image_channels,
            'state_dim': state_dim,
            'action_dim': state_dim,
        }
        
        policy = ACTPolicy(policy_config)
        print_success("Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        print_info(f"Total parameters: {total_params:,}")
        print_info(f"Trainable parameters: {trainable_params:,}")
        
        return policy, policy_config
        
    except Exception as e:
        print_error(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(policy, policy_config):
    """Test 4: Test forward pass with dummy data"""
    print_header("Test 4: Forward Pass")
    
    if policy is None:
        print_error("Policy not initialized, skipping forward pass test")
        return False
    
    try:
        from constants import SIM_TASK_CONFIGS
        
        task_name = 'sim_grasp_100ml_beaker'
        config = SIM_TASK_CONFIGS[task_name]
        
        if not torch.cuda.is_available():
            print_error("CUDA not available, skipping forward pass test")
            return False
        
        policy = policy.cuda()
        print_success("Model moved to CUDA")
        
        # Create dummy inputs
        batch_size = 2
        seq_len = policy_config['num_queries']
        num_cameras = len(policy_config['camera_names'])
        image_channels = policy_config['input_channels']
        state_dim = policy_config['state_dim']
        
        print_info(f"Creating dummy inputs...")
        print_info(f"  Batch size: {batch_size}")
        print_info(f"  Sequence length: {seq_len}")
        print_info(f"  Num cameras: {num_cameras}")
        print_info(f"  Image channels: {image_channels}")
        print_info(f"  State dim: {state_dim}")
        
        qpos = torch.randn(batch_size, seq_len, state_dim).cuda()
        images = torch.randn(batch_size, num_cameras, image_channels, 224, 224).cuda()
        actions = torch.randn(batch_size, seq_len, state_dim).cuda()
        is_pad = torch.zeros(batch_size, seq_len).bool().cuda()
        
        print_success("Dummy inputs created")
        print_info(f"  QPos shape: {qpos.shape}")
        print_info(f"  Images shape: {images.shape}")
        print_info(f"  Actions shape: {actions.shape}")
        print_info(f"  Is_pad shape: {is_pad.shape}")
        
        # Forward pass
        print_info("Running forward pass...")
        with torch.no_grad():
            output = policy(qpos, images, actions, is_pad)
        
        print_success("Forward pass successful!")
        print_info(f"Output keys: {list(output.keys())}")
        
        for key, value in output.items():
            if torch.is_tensor(value):
                print_info(f"  {key}: {value.shape if value.numel() > 1 else value.item()}")
            else:
                print_info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print_error(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test 5: Check GPU availability"""
    print_header("Test 5: GPU Availability")
    
    if torch.cuda.is_available():
        print_success("CUDA is available")
        print_info(f"GPU count: {torch.cuda.device_count()}")
        print_info(f"Current device: {torch.cuda.current_device()}")
        print_info(f"Device name: {torch.cuda.get_device_name(0)}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_info(f"Total memory: {total_memory:.2f} GB")
        return True
    else:
        print_error("CUDA is NOT available")
        print_info("Training will be very slow on CPU")
        return False


def main():
    print("\n" + "=" * 70)
    print("  Testing grasp_100ml_beaker Training Setup")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Task configuration
    results['Task Config'] = test_task_config()
    
    # Test 2: Dataset loading
    results['Dataset Loading'] = test_dataset_loading()
    
    # Test 3: Model initialization
    policy, policy_config = test_model_initialization()
    results['Model Init'] = (policy is not None)
    
    # Test 4: Forward pass
    if policy is not None:
        results['Forward Pass'] = test_forward_pass(policy, policy_config)
    else:
        results['Forward Pass'] = False
    
    # Test 5: GPU
    results['GPU'] = test_gpu()
    
    # Summary
    print_header("Test Summary")
    
    all_critical_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        
        # GPU is optional, others are critical
        if test_name != 'GPU' and not passed:
            all_critical_passed = False
    
    print("\n" + "=" * 70)
    
    if all_critical_passed:
        print("✓ All critical tests passed!")
        print("\nYou can now start training with:")
        print("  ./train_grasp_100ml_beaker.sh")
        print("or")
        print("  python3 train_grasp_beaker.py --mode train")
        print("=" * 70)
        return 0
    else:
        print("✗ Some critical tests failed. Please fix the errors above.")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

