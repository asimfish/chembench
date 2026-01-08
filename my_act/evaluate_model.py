#!/usr/bin/env python3
"""
Full evaluation: Test model on real dataset
使用真实数据集评估模型性能
"""

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm

def evaluate_on_dataset():
    """在数据集上评估模型"""
    print("=" * 70)
    print("Evaluating Model on Dataset: grasp_100ml_beaker")
    print("=" * 70)
    
    # 配置
    ckpt_dir = './checkpoints/grasp_100ml_beaker'
    
    # 加载配置
    print("\n1. Loading configuration...")
    from constants import SIM_TASK_CONFIGS
    from utils import load_data
    from policy import ACTPolicy
    
    task_config = SIM_TASK_CONFIGS['sim_grasp_100ml_beaker']
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
    image_channels = task_config['image_channels']
    state_dim = task_config['state_dim']
    
    print(f"   ✓ Dataset: {dataset_dir}")
    print(f"   ✓ Episodes: {num_episodes}")
    print(f"   ✓ State dim: {state_dim}")
    
    # 加载数据
    print("\n2. Loading dataset...")
    batch_size = 8
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size, batch_size
    )
    print(f"   ✓ Train batches: {len(train_dataloader)}")
    print(f"   ✓ Val batches: {len(val_dataloader)}")
    
    # 初始化模型
    print("\n3. Initializing and loading model...")
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
    
    # 加载最佳checkpoint
    checkpoint_to_try = [
        'policy_best.ckpt',
        'policy_epoch_200_seed_0.ckpt',
        'policy_epoch_100_seed_0.ckpt',
    ]
    
    for ckpt_name in checkpoint_to_try:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"   Loading: {ckpt_name}")
            state_dict = torch.load(ckpt_path)
            policy.load_state_dict(state_dict)
            print(f"   ✓ Loaded: {ckpt_name}")
            break
    
    if torch.cuda.is_available():
        policy = policy.cuda()
        print(f"   ✓ Using CUDA")
    else:
        print(f"   ✓ Using CPU")
    
    policy.eval()
    
    # 在验证集上评估
    print("\n4. Evaluating on validation set...")
    
    def forward_pass(data, policy):
        image_data, qpos_data, action_data, is_pad = data
        if torch.cuda.is_available():
            image_data = image_data.cuda()
            qpos_data = qpos_data.cuda()
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad)
    
    val_losses = []
    val_l1_losses = []
    val_kl_losses = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_dataloader, desc="   Validating")):
            output = forward_pass(data, policy)
            val_losses.append(output['loss'].item())
            val_l1_losses.append(output['l1'].item())
            val_kl_losses.append(output['kl'].item())
    
    # 计算平均损失
    avg_val_loss = np.mean(val_losses)
    avg_l1_loss = np.mean(val_l1_losses)
    avg_kl_loss = np.mean(val_kl_losses)
    
    print(f"\n   Validation Results:")
    print(f"   ✓ Average Loss: {avg_val_loss:.6f}")
    print(f"   ✓ Average L1 Loss: {avg_l1_loss:.6f}")
    print(f"   ✓ Average KL Loss: {avg_kl_loss:.6f}")
    
    # 在训练集上评估（可选）
    print("\n5. Evaluating on training set (sample)...")
    
    train_losses = []
    train_l1_losses = []
    train_kl_losses = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(train_dataloader, desc="   Training")):
            if batch_idx >= 10:  # 只评估前10个batch
                break
            output = forward_pass(data, policy)
            train_losses.append(output['loss'].item())
            train_l1_losses.append(output['l1'].item())
            train_kl_losses.append(output['kl'].item())
    
    avg_train_loss = np.mean(train_losses)
    avg_train_l1 = np.mean(train_l1_losses)
    avg_train_kl = np.mean(train_kl_losses)
    
    print(f"\n   Training Results (sample):")
    print(f"   ✓ Average Loss: {avg_train_loss:.6f}")
    print(f"   ✓ Average L1 Loss: {avg_train_l1:.6f}")
    print(f"   ✓ Average KL Loss: {avg_train_kl:.6f}")
    
    # 测试单个样本的预测
    print("\n6. Testing single sample prediction...")
    
    for data in val_dataloader:
        image_data, qpos_data, action_data, is_pad = data
        
        # 取第一个样本
        sample_qpos = qpos_data[0:1]  # [1, seq, state_dim]
        sample_image = image_data[0:1]  # [1, num_cam, channels, h, w]
        sample_action = action_data[0:1]  # [1, seq, state_dim]
        
        if torch.cuda.is_available():
            sample_qpos = sample_qpos.cuda()
            sample_image = sample_image.cuda()
            sample_action = sample_action.cuda()
        
        with torch.no_grad():
            # 推理模式（不提供真实动作）
            pred_actions = policy(sample_qpos, sample_image)
        
        # 计算预测误差
        pred_actions_np = pred_actions.cpu().numpy()
        true_actions_np = sample_action.cpu().numpy()
        
        mae = np.abs(pred_actions_np - true_actions_np).mean()
        mse = ((pred_actions_np - true_actions_np) ** 2).mean()
        
        print(f"   ✓ Prediction shape: {pred_actions.shape}")
        print(f"   ✓ MAE (Mean Absolute Error): {mae:.6f}")
        print(f"   ✓ RMSE (Root Mean Squared Error): {np.sqrt(mse):.6f}")
        
        # 显示一些预测值vs真实值
        print(f"\n   Sample predictions (first 3 timesteps, first 5 dims):")
        for t in range(min(3, pred_actions.shape[1])):
            print(f"   Timestep {t}:")
            print(f"     Predicted: {pred_actions_np[0, t, :5]}")
            print(f"     True:      {true_actions_np[0, t, :5]}")
        
        break
    
    # 总结
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Validation Loss:    {avg_val_loss:.6f}")
    print(f"  - L1 Loss:        {avg_l1_loss:.6f}")
    print(f"  - KL Loss:        {avg_kl_loss:.6f}")
    print(f"\nTraining Loss:      {avg_train_loss:.6f}")
    print(f"  - L1 Loss:        {avg_train_l1:.6f}")
    print(f"  - KL Loss:        {avg_train_kl:.6f}")
    print(f"\nSingle Sample:")
    print(f"  - MAE:            {mae:.6f}")
    print(f"  - RMSE:           {np.sqrt(mse):.6f}")
    print("=" * 70)
    
    # 保存评估结果
    result_path = os.path.join(ckpt_dir, 'dataset_evaluation.txt')
    with open(result_path, 'w') as f:
        f.write("Dataset Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Loss: {avg_val_loss:.6f}\n")
        f.write(f"  - L1 Loss: {avg_l1_loss:.6f}\n")
        f.write(f"  - KL Loss: {avg_kl_loss:.6f}\n\n")
        f.write(f"Training Loss (sample): {avg_train_loss:.6f}\n")
        f.write(f"  - L1 Loss: {avg_train_l1:.6f}\n")
        f.write(f"  - KL Loss: {avg_train_kl:.6f}\n\n")
        f.write(f"Single Sample Metrics:\n")
        f.write(f"  - MAE: {mae:.6f}\n")
        f.write(f"  - RMSE: {np.sqrt(mse):.6f}\n")
    
    print(f"\n✓ Results saved to: {result_path}")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    try:
        success = evaluate_on_dataset()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

