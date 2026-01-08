"""
State-based 数据集
加载分离的状态组件（target_pose, arm2_pos, arm2_vel, hand2_pos, hand2_vel, arm2_eef_pos, arm2_eef_quat）
支持有速度和无速度两种模式
"""
from typing import Dict
import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from psi_dp.common.streaming_replay_buffer import StreamingReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class GraspStateDataset(BaseLowdimDataset):
    """
    State-based 抓取数据集
    
    加载分离的状态组件：
    - target_pose: 目标物体位姿 (7)
    - arm2_pos: 机械臂关节位置 (7)
    - arm2_vel: 机械臂关节速度 (7) [可选]
    - hand2_pos: 机械手关节位置 (6)
    - hand2_vel: 机械手关节速度 (6) [可选]
    - arm2_eef_pos: 末端执行器位置 (3)
    - arm2_eef_quat: 末端执行器四元数 (4)
    """
    
    def __init__(self,
            zarr_path: str, 
            horizon: int = 1,
            n_obs_steps: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
            use_velocity: bool = False
            ):
        """
        Args:
            zarr_path: Zarr 数据集路径
            horizon: 动作预测长度
            n_obs_steps: 观测步数
            pad_before: 序列前填充
            pad_after: 序列后填充
            seed: 随机种子
            val_ratio: 验证集比例
            max_train_episodes: 最大训练 episode 数
            use_velocity: 是否使用速度观测
        """
        self.use_velocity = use_velocity
        
        # 根据 use_velocity 确定需要加载的数据键
        keys = [
            'target_pose',
            'arm2_pos',
            'hand2_pos',
            'arm2_eef_pos',
            'arm2_eef_quat',
            'action'
        ]
        
        # 如果使用速度观测，添加速度键
        if use_velocity:
            keys.insert(2, 'arm2_vel')   # 在 arm2_pos 后插入
            keys.insert(4, 'hand2_vel')  # 在 hand2_pos 后插入
        
        print(f"[GraspStateDataset] 初始化:")
        print(f"  - use_velocity: {use_velocity}")
        print(f"  - 加载的数据键: {keys}")
        
        # 初始化数据缓冲区
        self.replay_buffer = StreamingReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
            
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
        
        # 打印数据集信息
        print(f"  - 总 episodes: {self.replay_buffer.n_episodes}")
        print(f"  - 训练 episodes: {train_mask.sum()}")
        print(f"  - 总样本数: {len(self.sampler)}")

    def _sample_to_data(self, sample: dict) -> dict:
        """将采样数据转换为模型输入格式"""
        T_slice = slice(self.n_obs_steps)
        
        # 基础观测数据
        obs_data = {
            'target_pose': sample['target_pose'][T_slice].astype(np.float32),
            'arm2_pos': sample['arm2_pos'][T_slice].astype(np.float32),
            'hand2_pos': sample['hand2_pos'][T_slice].astype(np.float32),
            'arm2_eef_pos': sample['arm2_eef_pos'][T_slice].astype(np.float32),
            'arm2_eef_quat': sample['arm2_eef_quat'][T_slice].astype(np.float32),
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
        """创建数据标准化器"""
        # 基础数据
        data = {
            'action': self.replay_buffer['action'],
            'target_pose': self.replay_buffer['target_pose'],
            'arm2_pos': self.replay_buffer['arm2_pos'],
            'hand2_pos': self.replay_buffer['hand2_pos'],
            'arm2_eef_pos': self.replay_buffer['arm2_eef_pos'],
            'arm2_eef_quat': self.replay_buffer['arm2_eef_quat'],
        }
        
        # 如果使用速度观测，添加速度数据
        if self.use_velocity:
            data['arm2_vel'] = self.replay_buffer['arm2_vel']
            data['hand2_vel'] = self.replay_buffer['hand2_vel']
        
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

