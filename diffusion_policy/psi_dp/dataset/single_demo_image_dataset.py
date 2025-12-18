from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
# Feng Yunduo, Start
# from diffusion_policy.common.streaming_replay_buffer import StreamingReplayBuffer
from psi_dp.common.streaming_replay_buffer import StreamingReplayBuffer
# Feng Yunduo, End
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import torch.nn.functional as F

class SingleDemoImageDataset(BaseImageDataset):
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
            # image_size=(384, 384)
            ):
        
        super().__init__()
        self.image_size = image_size
        # 更新keys以匹配新的数据结构
        self.replay_buffer = StreamingReplayBuffer.copy_from_path(
            zarr_path, keys=[
                # FYD修改
                'chest_camera_rgb', # 胸部相机
                'head_camera_rgb',# 头部相机
                'third_person_camera_rgb', # 第三人称相机
                'arm2_pos',
                'arm2_vel',
                'hand2_pos',
                'hand2_vel',
                'arm2_eef_pos',
                'arm2_eef_quat',
                'target_pose',  # 目标物体位姿
                'action'
            ])
            
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

    # def _process_image_batch(self, images):
    #     """批量处理图像"""
    #     # 转换整个批次
    #     rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
    #     # 处理RGB
    #     rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

    #     rgb = F.interpolate(
    #         rgb / 255.0,
    #         size=self.image_size,
    #         mode='bilinear',
    #         align_corners=False
    #     )

    #     return rgb.numpy()

    def _process_image_batch(self, images):
        """批量处理图像"""
        # 转换整个批次
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
        # 处理RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        return rgb.numpy()
    
    def _sample_to_data(self, sample):
        T_slice = slice(self.n_obs_steps)
        
        # 修改：FYD
        # 处理图像数据
        sensor1_frames = self._process_image_batch(sample['head_camera_rgb'][T_slice])
        sensor2_frames = self._process_image_batch(sample['chest_camera_rgb'][T_slice])
        sensor3_frames = self._process_image_batch(sample['third_person_camera_rgb'][T_slice])

        
        # 修改：FYD
        # 处理低维数据
        data = {
            'obs': {
                'head_camera_rgb': sensor1_frames,
                'chest_camera_rgb': sensor2_frames,
                'third_person_camera_rgb': sensor3_frames,
                'arm2_pos': sample['arm2_pos'][T_slice].astype(np.float32),
                'arm2_vel': sample['arm2_vel'][T_slice].astype(np.float32),
                'hand2_pos': sample['hand2_pos'][T_slice].astype(np.float32),
                'hand2_vel': sample['hand2_vel'][T_slice].astype(np.float32),
                'arm2_eef_pos': sample['arm2_eef_pos'][T_slice].astype(np.float32),
                'arm2_eef_quat': sample['arm2_eef_quat'][T_slice].astype(np.float32),
                'target_pose': sample['target_pose'][T_slice].astype(np.float32)
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

     # 修改：FYD
    def get_normalizer(self, mode='limits', **kwargs):
        """修改normalizer以适配新的数据结构"""
        data = {
            'action': self.replay_buffer['action'],
            'arm2_pos': self.replay_buffer['arm2_pos'],
            'arm2_vel': self.replay_buffer['arm2_vel'],
            'hand2_pos': self.replay_buffer['hand2_pos'],
            'hand2_vel': self.replay_buffer['hand2_vel'],
            'arm2_eef_pos': self.replay_buffer['arm2_eef_pos'],
            'arm2_eef_quat': self.replay_buffer['arm2_eef_quat'],
            'target_pose': self.replay_buffer['target_pose']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # 添加图像normalizer
        # 修改：FYD
        normalizer['head_camera_rgb'] = get_image_range_normalizer()
        normalizer['chest_camera_rgb'] = get_image_range_normalizer()
        normalizer['third_person_camera_rgb'] = get_image_range_normalizer()
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

def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
