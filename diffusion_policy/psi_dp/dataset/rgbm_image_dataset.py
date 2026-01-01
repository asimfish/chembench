"""
RGBM (RGB + Mask) 图像数据集
用于处理带有 mask 通道的图像数据
"""
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from psi_dp.common.streaming_replay_buffer import StreamingReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import torch.nn.functional as F


class RGBMImageDataset(BaseImageDataset):
    """
    处理 RGBM (RGB + Mask) 格式图像的数据集
    
    输入图像格式: [H, W, 4] 其中最后一个通道是 mask
    输出图像格式: [4, H, W] 归一化后的 RGBM
    """
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(224, 224),
            use_velocity=False,  # 是否使用速度观测
            ):
        
        super().__init__()
        self.image_size = image_size
        self.use_velocity = use_velocity
        
        # 基础观测键
        keys = [
            'chest_camera_rgb',  # 胸部相机 RGBM
            'head_camera_rgb',   # 头部相机 RGBM
            'arm2_pos',
            'hand2_pos',
            'arm2_eef_pos',
            'arm2_eef_quat',
            'target_pose',
            'action'
        ]
        
        # 如果使用速度观测，添加速度键
        if use_velocity:
            keys.insert(3, 'arm2_vel')   # 在 arm2_pos 后插入
            keys.insert(5, 'hand2_vel')  # 在 hand2_pos 后插入
        
        # 加载数据
        self.replay_buffer = StreamingReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def _process_rgbm_batch(self, images):
        """
        批量处理 RGBM 图像
        
        输入: images [T, H, W, 4] - RGB(3通道) + Mask(1通道)
        输出: [T, 4, H, W] - 归一化后的 RGBM
        """
        # 分离 RGB 和 Mask
        rgb = torch.from_numpy(images[..., :3]).float()   # [T, H, W, 3]
        mask = torch.from_numpy(images[..., 3:]).float()  # [T, H, W, 1]
        
        # 处理 RGB：归一化并 resize
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]
        rgb = F.interpolate(
            rgb / 255.0,              # 归一化到 [0, 1]
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # 处理 Mask：resize 并二值化
        mask = mask.permute(0, 3, 1, 2)  # [T, 1, H, W]
        mask = F.interpolate(
            mask / 255.0 if mask.max() > 1 else mask,  # 处理 0-255 或 0-1 的 mask
            size=self.image_size,
            mode='nearest'  # 最近邻保持边缘清晰
        )
        mask = (mask > 0.5).float()  # 二值化

        # 合并 RGBM
        combined = torch.cat([rgb, mask], dim=1)  # [T, 4, H, W]
        return combined.numpy()

    def _process_rgb_batch(self, images):
        """
        批量处理纯 RGB 图像（兼容性方法）
        """
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]
        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        return rgb.numpy()
    
    def _sample_to_data(self, sample):
        T_slice = slice(self.n_obs_steps)
        
        # 使用 RGBM 处理方法
        chest_frames = self._process_rgbm_batch(sample['chest_camera_rgb'][T_slice])
        head_frames = self._process_rgbm_batch(sample['head_camera_rgb'][T_slice])

        # 基础观测数据
        obs_data = {
            'chest_camera_rgb': chest_frames,
            'head_camera_rgb': head_frames,
            'arm2_pos': sample['arm2_pos'][T_slice].astype(np.float32),
            'hand2_pos': sample['hand2_pos'][T_slice].astype(np.float32),
            'arm2_eef_pos': sample['arm2_eef_pos'][T_slice].astype(np.float32),
            'arm2_eef_quat': sample['arm2_eef_quat'][T_slice].astype(np.float32),
            'target_pose': sample['target_pose'][T_slice].astype(np.float32)
        }
        
        # 如果使用速度观测，添加速度数据
        if self.use_velocity:
            obs_data['arm2_vel'] = sample['arm2_vel'][T_slice].astype(np.float32)
            obs_data['hand2_vel'] = sample['hand2_vel'][T_slice].astype(np.float32)

        data = {
            'obs': obs_data,
            'action': sample['action'].astype(np.float32)
        }
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        """创建数据归一化器"""
        # 基础数据
        data = {
            'action': self.replay_buffer['action'],
            'arm2_pos': self.replay_buffer['arm2_pos'],
            'hand2_pos': self.replay_buffer['hand2_pos'],
            'arm2_eef_pos': self.replay_buffer['arm2_eef_pos'],
            'arm2_eef_quat': self.replay_buffer['arm2_eef_quat'],
            'target_pose': self.replay_buffer['target_pose']
        }
        
        # 如果使用速度观测，添加速度数据
        if self.use_velocity:
            data['arm2_vel'] = self.replay_buffer['arm2_vel']
            data['hand2_vel'] = self.replay_buffer['hand2_vel']
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 添加图像 normalizer（RGBM 已经在 _process_rgbm_batch 中归一化）
        normalizer['chest_camera_rgb'] = get_image_range_normalizer()
        normalizer['head_camera_rgb'] = get_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

