"""
Complete script to convert Zarr dataset and train ACT policy.
This script:
1. Converts Zarr data to HDF5 format
2. Trains ACT policy on the converted data
"""

import torch
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from convert_zarr_to_hdf5 import convert_zarr_to_hdf5
from utils import load_data, compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy
import matplotlib.pyplot as plt
from copy import deepcopy
import subprocess
import sys


def train_bc(train_dataloader, val_dataloader, config):
    """Training function for behavior cloning."""
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']

    set_seed(seed)

    # Create policy
    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        
        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                image_data, qpos_data, action_data, is_pad = data
                image_data = image_data.cuda()
                qpos_data = qpos_data.cuda()
                action_data = action_data.cuda()
                is_pad = is_pad.cuda()
                
                forward_dict = policy(qpos_data, image_data, action_data, is_pad)
                epoch_dicts.append(forward_dict)
            
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            image_data, qpos_data, action_data, is_pad = data
            image_data = image_data.cuda()
            qpos_data = qpos_data.cuda()
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            
            forward_dict = policy(qpos_data, image_data, action_data, is_pad)
            
            # Backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # Save last checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    # Save best checkpoint
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """Plot and save training curves."""
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


def main():
    parser = argparse.ArgumentParser(description='Convert Zarr and train ACT policy')
    
    # Data parameters
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to input .zarr file')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory to save/load HDF5 episodes')
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Skip conversion if HDF5 files already exist')
    parser.add_argument('--camera_names', type=str, nargs='+',
                        default=['head_camera', 'chest_camera'],
                        help='List of camera names to use')
    
    # Training parameters
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # ACT parameters
    parser.add_argument('--kl_weight', type=int, default=10,
                        help='KL divergence weight')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Action chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dimension')
    
    args = parser.parse_args()
    
    # Step 1: Convert Zarr to HDF5 if needed
    if not args.skip_conversion:
        print("=" * 50)
        print("Step 1: Converting Zarr to HDF5")
        print("=" * 50)
        num_episodes, episode_len, camera_names = convert_zarr_to_hdf5(
            args.zarr_path, 
            args.dataset_dir, 
            args.camera_names
        )
    else:
        print("Skipping conversion, using existing HDF5 files")
        # Estimate from existing files
        episode_files = [f for f in os.listdir(args.dataset_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
        num_episodes = len(episode_files)
        camera_names = args.camera_names
        
        # Get episode length from first file
        import h5py
        with h5py.File(os.path.join(args.dataset_dir, 'episode_0.hdf5'), 'r') as f:
            episode_len = f['action'].shape[0]
        
        print(f"Found {num_episodes} episodes, episode length: {episode_len}")
    
    # Step 2: Setup training configuration
    print("\n" + "=" * 50)
    print("Step 2: Setting up training")
    print("=" * 50)
    
    # Determine state dimension from the data
    import h5py
    with h5py.File(os.path.join(args.dataset_dir, 'episode_0.hdf5'), 'r') as f:
        state_dim = f['observations/qpos'].shape[1]
        action_dim = f['action'].shape[1]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Note: The original ACT implementation hardcodes state_dim=14 in detr_vae.py
    # If your data has different dimensions, you'll need to modify:
    # - detr/models/detr_vae.py: line 230 (state_dim = 14) and line 258
    # - All references to hardcoded dimension 14 in the model
    if state_dim != 14 or action_dim != 14:
        print(f"\nWARNING: Your data has state_dim={state_dim} and action_dim={action_dim}")
        print(f"The original ACT model expects state_dim=14 (bimanual setup).")
        print(f"Attempting to patch detr_vae.py to support your dimensions...")
        
        # Try to apply patch
        try:
            patch_script = os.path.join(os.path.dirname(__file__), 'patch_detr_for_custom_dims.py')
            result = subprocess.run([sys.executable, patch_script], 
                                    capture_output=True, text=True, check=True)
            print(result.stdout)
            print("Patch applied successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply patch: {e}")
            print(e.stdout)
            print(e.stderr)
            print("\nPlease manually patch detr/models/detr_vae.py or use data with state_dim=14")
            sys.exit(1)
        except Exception as e:
            print(f"Error applying patch: {e}")
            print("\nPlease manually patch detr/models/detr_vae.py or use data with state_dim=14")
            sys.exit(1)
        print()
    
    # Policy configuration
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    
    policy_config = {
        'lr': args.lr,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': camera_names,
        'state_dim': state_dim,  # Add state_dim to config
        'action_dim': action_dim,  # Add action_dim to config
        # Required by detr/main.py argument parser
        'policy_class': 'ACT',
        'task_name': 'custom_task',
        'ckpt_dir': args.ckpt_dir,
        'seed': args.seed,
        'num_epochs': args.num_epochs,
        'temporal_agg': False,
    }
    
    config = {
        'num_epochs': args.num_epochs,
        'ckpt_dir': args.ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args.lr,
        'policy_config': policy_config,
        'seed': args.seed,
        'camera_names': camera_names,
    }
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Step 3: Load data
    print("\n" + "=" * 50)
    print("Step 3: Loading data")
    print("=" * 50)
    
    train_dataloader, val_dataloader, stats, _ = load_data(
        args.dataset_dir,
        num_episodes,
        camera_names,
        args.batch_size,
        args.batch_size,
        chunk_size=args.chunk_size  # Pass chunk_size to ensure data is padded to this length
    )
    
    # Save dataset stats
    stats_path = os.path.join(args.ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved dataset stats to {stats_path}")
    
    # Step 4: Train
    print("\n" + "=" * 50)
    print("Step 4: Training ACT policy")
    print("=" * 50)
    
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    
    # Save best checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'\n' + "=" * 50)
    print(f'Training Complete!')
    print(f'Best checkpoint: val loss {min_val_loss:.6f} @ epoch {best_epoch}')
    print(f'Saved to: {ckpt_path}')
    print("=" * 50)


if __name__ == '__main__':
    main()

