#!/usr/bin/env python3
"""
Training script for grasp_100ml_beaker task
Provides a more flexible Python interface for training
"""

import os
import sys
import argparse

def train_grasp_beaker(
    ckpt_dir='./checkpoints/grasp_100ml_beaker',
    batch_size=8,
    num_epochs=2000,
    lr=1e-5,
    kl_weight=10,
    chunk_size=100,
    hidden_dim=512,
    dim_feedforward=3200,
    seed=0,
    temporal_agg=False
):
    """
    Train ACT policy for grasping 100ml beaker task
    
    Args:
        ckpt_dir: Directory to save checkpoints
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        kl_weight: KL divergence weight for VAE
        chunk_size: Number of action chunks to predict
        hidden_dim: Hidden dimension for transformer
        dim_feedforward: Feedforward dimension for transformer
        seed: Random seed
        temporal_agg: Whether to use temporal aggregation during evaluation
    """
    
    # Import the main function
    from imitate_episodes import main
    
    # Prepare arguments
    args = {
        'eval': False,
        'onscreen_render': False,
        'ckpt_dir': ckpt_dir,
        'policy_class': 'ACT',
        'task_name': 'sim_grasp_100ml_beaker',
        'batch_size': batch_size,
        'seed': seed,
        'num_epochs': num_epochs,
        'lr': lr,
        'kl_weight': kl_weight,
        'chunk_size': chunk_size,
        'hidden_dim': hidden_dim,
        'dim_feedforward': dim_feedforward,
        'temporal_agg': temporal_agg,
    }
    
    print("=" * 60)
    print("Training ACT Policy for grasp_100ml_beaker")
    print("=" * 60)
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Chunk size: {chunk_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"KL weight: {kl_weight}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Create checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Run training
    main(args)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Model saved to: {ckpt_dir}")
    print("=" * 60)


def evaluate_grasp_beaker(
    ckpt_dir='./checkpoints/grasp_100ml_beaker',
    batch_size=8,
    kl_weight=10,
    chunk_size=100,
    hidden_dim=512,
    dim_feedforward=3200,
    seed=0,
    temporal_agg=True,
    onscreen_render=False
):
    """
    Evaluate trained ACT policy for grasping 100ml beaker task
    
    Args:
        ckpt_dir: Directory containing trained checkpoint
        batch_size: Batch size (not used during eval but required)
        kl_weight: KL divergence weight (must match training)
        chunk_size: Number of action chunks (must match training)
        hidden_dim: Hidden dimension (must match training)
        dim_feedforward: Feedforward dimension (must match training)
        seed: Random seed
        temporal_agg: Whether to use temporal aggregation
        onscreen_render: Whether to render on screen
    """
    
    # Import the main function
    from imitate_episodes import main
    
    # Prepare arguments
    args = {
        'eval': True,
        'onscreen_render': onscreen_render,
        'ckpt_dir': ckpt_dir,
        'policy_class': 'ACT',
        'task_name': 'sim_grasp_100ml_beaker',
        'batch_size': batch_size,
        'seed': seed,
        'num_epochs': 1,  # Not used during eval
        'lr': 1e-5,  # Not used during eval
        'kl_weight': kl_weight,
        'chunk_size': chunk_size,
        'hidden_dim': hidden_dim,
        'dim_feedforward': dim_feedforward,
        'temporal_agg': temporal_agg,
    }
    
    print("=" * 60)
    print("Evaluating ACT Policy for grasp_100ml_beaker")
    print("=" * 60)
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Temporal aggregation: {temporal_agg}")
    print(f"Onscreen render: {onscreen_render}")
    print("=" * 60)
    
    # Run evaluation
    main(args)
    
    print("=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {ckpt_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate ACT policy for grasp_100ml_beaker task')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Mode: train or eval')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/grasp_100ml_beaker',
                        help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=int, default=10,
                        help='KL divergence weight')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Action chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dimension')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Use temporal aggregation')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='Render on screen (eval only)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_grasp_beaker(
            ckpt_dir=args.ckpt_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            kl_weight=args.kl_weight,
            chunk_size=args.chunk_size,
            hidden_dim=args.hidden_dim,
            dim_feedforward=args.dim_feedforward,
            seed=args.seed,
            temporal_agg=args.temporal_agg
        )
    elif args.mode == 'eval':
        evaluate_grasp_beaker(
            ckpt_dir=args.ckpt_dir,
            batch_size=args.batch_size,
            kl_weight=args.kl_weight,
            chunk_size=args.chunk_size,
            hidden_dim=args.hidden_dim,
            dim_feedforward=args.dim_feedforward,
            seed=args.seed,
            temporal_agg=args.temporal_agg,
            onscreen_render=args.onscreen_render
        )

