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
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import torch.nn.functional as F

class GraspImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(224, 224)  # 添加image_size参数
            ):
        
        super().__init__()
        self.image_size = image_size
        # 使用StreamingReplayBuffer
        self.replay_buffer = StreamingReplayBuffer.copy_from_path(
            zarr_path, keys=['top_camera', 'state', 'action'])
            # zarr_path, keys=['right_camera', 'top_camera', 'state', 'action'])
            
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


    def _process_mask_image_batch(self, images):
        """批量处理图像"""
        # 转换整个批次
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        mask = torch.from_numpy(images[..., 3:]).float() # [T, H, W, 1]
        
        # 处理RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # 处理mask
        mask = mask.permute(0, 3, 1, 2)  # [T, 1, H, W]
        mask = F.interpolate(
            mask,
            size=self.image_size,
            mode='nearest'
        )

        mask = (mask > 0.5).float()

        # 合并
        combined = torch.cat([rgb, mask], dim=1)  # [T, 4, H, W]
        return combined.numpy()

    def _process_image_batch(self, images):
        """批量处理图像"""
        # 转换整个批次
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
        # 处理RGB
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
        state = sample['state'].astype(np.float32)

        # 批量处理所有图像
        top_processed_frames = self._process_image_batch(sample['top_camera'][T_slice])
        # right_processed_frames = self._process_image_batch(sample['right_camera'][T_slice])

        data = {
            'obs': {
                # 'top_camera': top_processed_frames,
                # 'right_camera': right_processed_frames,
                'state': state[T_slice]
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        """修改normalizer以处理流式数据"""
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['top_camera'] = get_image_range_normalizer()
        # normalizer['right_camera'] = get_image_range_normalizer()
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


class GraspLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        # 初始化数据缓冲区，只加载状态和动作数据
        self.replay_buffer = StreamingReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action'])
            
        # 创建验证集掩码
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        # 如果需要，对训练数据进行下采样
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 初始化序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        # 保存参数
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def _sample_to_data(self, sample):
        """将采样数据转换为模型输入格式"""
        T_slice = slice(self.n_obs_steps)
        
        data = {
            'obs': {
                'state': sample['state'][T_slice].astype(np.float32)
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        """创建数据标准化器"""
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本"""
        sample = self.sampler.sample_sequence(idx)        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_all_actions(self) -> torch.Tensor:
        """获取所有动作数据"""
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        """返回数据集长度"""
        return len(self.sampler)


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
