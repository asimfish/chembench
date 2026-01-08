#!/usr/bin/env python3
"""
Simple test: Load trained model and test forward pass
无需环境，只测试模型加载和前向传播
"""

import torch
import numpy as np
import pickle
import os

def test_load_model():
    """测试加载训练好的模型"""
    print("=" * 70)
    print("Testing Trained Model: grasp_100ml_beaker")
    print("=" * 70)
    
    # 配置
    ckpt_dir = './checkpoints/grasp_100ml_beaker'
    
    # 检查checkpoint文件
    print("\n1. Checking checkpoint files...")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    print(f"   Found {len(ckpt_files)} checkpoint files:")
    for ckpt_file in sorted(ckpt_files):
        size_mb = os.path.getsize(os.path.join(ckpt_dir, ckpt_file)) / 1e6
        print(f"   - {ckpt_file} ({size_mb:.1f} MB)")
    
    # 加载配置
    print("\n2. Loading task configuration...")
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS['sim_grasp_100ml_beaker']
    
    camera_names = task_config['camera_names']
    image_channels = task_config['image_channels']
    state_dim = task_config['state_dim']
    
    print(f"   ✓ Camera names: {camera_names}")
    print(f"   ✓ Image channels: {image_channels}")
    print(f"   ✓ State dim: {state_dim}")
    
    # 创建policy配置
    print("\n3. Creating policy configuration...")
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
    print(f"   ✓ Policy config created")
    
    # 初始化模型
    print("\n4. Initializing model...")
    from policy import ACTPolicy
    policy = ACTPolicy(policy_config)
    print(f"   ✓ Model initialized")
    
    # 加载权重 - 尝试多个可能的checkpoint
    print("\n5. Loading model weights...")
    checkpoint_to_try = [
        'policy_best.ckpt',
        'policy_last.ckpt', 
        'policy_epoch_200_seed_0.ckpt',
        'policy_epoch_100_seed_0.ckpt',
    ]
    
    loaded = False
    for ckpt_name in checkpoint_to_try:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"   Loading: {ckpt_name}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            policy.load_state_dict(state_dict)
            print(f"   ✓ Loaded checkpoint: {ckpt_name}")
            loaded = True
            break
    
    if not loaded:
        print(f"   ✗ No checkpoint found!")
        return False
    
    # 加载数据集统计信息
    print("\n6. Loading dataset statistics...")
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f"   ✓ Stats loaded: {list(stats.keys())}")
    
    # 测试前向传播
    print("\n7. Testing forward pass...")
    
    if torch.cuda.is_available():
        policy = policy.cuda()
        device = 'cuda'
        print(f"   ✓ Using CUDA")
    else:
        device = 'cpu'
        print(f"   ✓ Using CPU")
    
    policy.eval()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 100
    num_cameras = len(camera_names)
    
    with torch.no_grad():
        # 模拟输入
        qpos = torch.randn(batch_size, seq_len, state_dim).to(device)
        images = torch.randn(batch_size, num_cameras, image_channels, 224, 224).to(device)
        actions = torch.randn(batch_size, seq_len, state_dim).to(device)
        is_pad = torch.zeros(batch_size, seq_len).bool().to(device)
        
        # 训练模式测试
        print(f"\n   Testing training mode (with actions)...")
        output = policy(qpos, images, actions, is_pad)
        print(f"   ✓ Output keys: {list(output.keys())}")
        for key, value in output.items():
            if torch.is_tensor(value):
                print(f"     - {key}: {value.item():.6f}")
        
        # 推理模式测试  
        print(f"\n   Testing inference mode (without actions)...")
        pred_actions = policy(qpos, images)
        print(f"   ✓ Predicted actions shape: {pred_actions.shape}")
        print(f"     Expected: [{batch_size}, {seq_len}, {state_dim}]")
    
    # 成功
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nModel is ready for evaluation or deployment.")
    print("\nNext steps:")
    print("1. Run full evaluation with environment:")
    print("   ./eval_grasp_100ml_beaker.sh")
    print("\n2. Or run evaluation manually:")
    print(f"   python3 imitate_episodes.py --eval \\")
    print(f"       --task_name sim_grasp_100ml_beaker \\")
    print(f"       --ckpt_dir {ckpt_dir} \\")
    print(f"       --policy_class ACT \\")
    print(f"       --kl_weight 10 --chunk_size 100 \\")
    print(f"       --hidden_dim 512 --dim_feedforward 3200 \\")
    print(f"       --batch_size 8 --num_epochs 1 \\")
    print(f"       --lr 1e-5 --seed 0")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    try:
        success = test_load_model()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

