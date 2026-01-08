#!/usr/bin/env python3
"""
Quick test script to verify dataset loading and model initialization
"""

import torch
import sys
import os

def test_dataset():
    """Test if dataset can be loaded correctly"""
    print("=" * 60)
    print("Testing dataset loading...")
    print("=" * 60)
    
    from constants import SIM_TASK_CONFIGS
    from utils import load_data
    
    task_name = 'sim_grasp_100ml_beaker'
    task_config = SIM_TASK_CONFIGS[task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
    batch_size = 2
    
    print(f"\nTask: {task_name}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Num episodes: {num_episodes}")
    print(f"Camera names: {camera_names}")
    
    try:
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir, num_episodes, camera_names, batch_size, batch_size
        )
        
        print("\n✓ Dataset loaded successfully!")
        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Val batches: {len(val_dataloader)}")
        print(f"  Stats keys: {list(stats.keys())}")
        
        # Test one batch
        for batch in train_dataloader:
            image_data, qpos_data, action_data, is_pad = batch
            print(f"\n✓ Batch loaded successfully!")
            print(f"  Image shape: {image_data.shape}")
            print(f"  Qpos shape: {qpos_data.shape}")
            print(f"  Action shape: {action_data.shape}")
            print(f"  Is_pad shape: {is_pad.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test if model can be initialized"""
    print("\n" + "=" * 60)
    print("Testing model initialization...")
    print("=" * 60)
    
    from policy import ACTPolicy
    from constants import SIM_TASK_CONFIGS
    
    task_name = 'sim_grasp_100ml_beaker'
    task_config = SIM_TASK_CONFIGS[task_name]
    camera_names = task_config['camera_names']
    
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
    }
    
    try:
        policy = ACTPolicy(policy_config)
        print("\n✓ Model initialized successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            policy = policy.cuda()
            print("\n✓ Model moved to CUDA")
            
            # Create dummy inputs
            batch_size = 2
            qpos = torch.randn(batch_size, 100, 13).cuda()
            images = torch.randn(batch_size, 3, 3, 224, 224).cuda()  # 3 cameras
            actions = torch.randn(batch_size, 100, 13).cuda()
            is_pad = torch.zeros(batch_size, 100).bool().cuda()
            
            print("\n✓ Testing forward pass...")
            with torch.no_grad():
                output = policy(qpos, images, actions, is_pad)
            
            print(f"✓ Forward pass successful!")
            print(f"  Output keys: {list(output.keys())}")
            if 'loss' in output:
                print(f"  Loss: {output['loss'].item():.6f}")
        else:
            print("\n⚠ CUDA not available, skipping forward pass test")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test GPU availability"""
    print("\n" + "=" * 60)
    print("Testing GPU availability...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available!")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # Test memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Total memory: {total_memory:.2f} GB")
        
        return True
    else:
        print("\n✗ CUDA is NOT available!")
        print("  Training will be very slow on CPU")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing grasp_100ml_beaker training setup")
    print("=" * 60)
    
    # Run tests
    results = {
        'GPU': test_gpu(),
        'Dataset': test_dataset(),
        'Model': test_model(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ All tests passed! Ready to train.")
        print("=" * 60)
        print("\nRun training with:")
        print("  ./train_grasp_100ml_beaker.sh")
        print("or")
        print("  python3 train_grasp_beaker.py --mode train")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed. Please fix errors before training.")
        print("=" * 60)
        sys.exit(1)

