"""
多模态图像数据集 (Multi-Modal Image Dataset)
支持多种图像观测模式:
  - rgb:    纯 RGB 3通道
  - rgbm:   RGB + Mask 4通道
  - nd:     Normal + Depth 4通道
  - rgbnd:  RGB + Normal + Depth 7通道
  - rgb_masked: RGB * Mask 3通道 (背景置黑)
  - rgb_masked_rgb: RGB + RGB*Mask 6通道 (原始RGB + 背景置黑RGB)
"""
from typing import Dict, Literal
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


# 观测模式类型
ObsModeType = Literal["rgb", "rgbm", "nd", "rgbnd", "rgb_masked", "rgb_masked_rgb"]


class MultiModalImageDataset(BaseImageDataset):
    """
    多模态图像数据集，支持多种图像观测模式
    
    观测模式:
        - rgb:    纯 RGB 图像 [H, W, 3] -> [3, H, W]
        - rgbm:   RGB + Mask  [H, W, 4] -> [4, H, W]
        - nd:     Normal(3) + Depth(1) -> [4, H, W]
        - rgbnd:  RGB(3) + Normal(3) + Depth(1) -> [7, H, W]
        - rgb_masked: RGB * Mask -> [3, H, W] (仅保留 Mask 区域的 RGB)
        - rgb_masked_rgb: RGB + RGB*Mask -> [6, H, W] (原始RGB + 背景置黑RGB)
    
    Depth 数据处理:
        存储格式: uint16, 0表示无效, 1-65535表示有效深度
        解码公式: depth_meters = (d_u16 - 1) / 65534 * (far - near) + near
        其中 near=0.2m, far=1.8m
        
    Normal 数据处理:
        存储格式: uint8 [0, 255]
        解码公式: normal = n_u8 / 255.0 * 2.0 - 1.0  (映射到 [-1, 1])
    """
    
    # Depth 解码参数 (与 zarr_utils.py 保持一致)
    DEPTH_NEAR = 0.2
    DEPTH_FAR = 1.8
    
    def __init__(self,
            zarr_path: str, 
            horizon: int = 1,
            n_obs_steps: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
            image_size: tuple = (224, 224),
            obs_mode: ObsModeType = "rgbm",
            use_velocity: bool = False,
            use_third_camera: bool = False,
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
            image_size: 输出图像尺寸 (H, W)
            obs_mode: 图像观测模式 ["rgb", "rgbm", "nd", "rgbnd", "rgb_masked", "rgb_masked_rgb"]
            use_velocity: 是否使用速度观测
            use_third_camera: 是否使用第三人称相机
        """
        super().__init__()
        self.image_size = image_size
        self.obs_mode = obs_mode
        self.use_velocity = use_velocity
        self.use_third_camera = use_third_camera
        
        # 验证 obs_mode
        valid_modes = ["rgb", "rgbm", "nd", "rgbnd", "rgb_masked", "rgb_masked_rgb"]
        if obs_mode not in valid_modes:
            raise ValueError(f"obs_mode must be one of {valid_modes}, got {obs_mode}")
        
        # 根据 obs_mode 确定需要加载的数据键
        keys = self._get_data_keys()
        
        print(f"[MultiModalImageDataset] 初始化:")
        print(f"  - obs_mode: {obs_mode}")
        print(f"  - use_velocity: {use_velocity}")
        print(f"  - use_third_camera: {use_third_camera}")
        print(f"  - image_size: {image_size}")
        print(f"  - 加载的数据键: {keys}")
        
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
        
        # 打印数据集信息
        print(f"  - 总 episodes: {self.replay_buffer.n_episodes}")
        print(f"  - 训练 episodes: {train_mask.sum()}")
        print(f"  - 总样本数: {len(self.sampler)}")

    def _get_data_keys(self) -> list:
        """
        根据 obs_mode、use_velocity 和 use_third_camera 确定需要加载的数据键
        """
        # 基础状态键
        keys = [
            'arm2_pos',
            'hand2_pos',
            'arm2_eef_pos',
            'arm2_eef_quat',
            'target_pose',
            'action'
        ]
        
        # 如果使用速度观测，添加速度键
        if self.use_velocity:
            keys.insert(1, 'arm2_vel')   # 在 arm2_pos 后插入
            keys.insert(3, 'hand2_vel')  # 在 hand2_pos 后插入
        
        # 根据 obs_mode 添加图像相关的键
        if self.obs_mode == "rgb":
            # 纯 RGB
            keys.extend([
                'chest_camera_rgb',
                'head_camera_rgb',
            ])
            if self.use_third_camera:
                keys.append('third_camera_rgb')
        elif self.obs_mode == "rgbm":
            # RGB + Mask
            keys.extend([
                'chest_camera_rgb',
                'head_camera_rgb',
                'chest_camera_mask',
                'head_camera_mask',
            ])
            if self.use_third_camera:
                keys.extend([
                    'third_camera_rgb',
                    'third_camera_mask',
                ])
        elif self.obs_mode == "rgb_masked":
            # RGB * Mask (需要同时读取 RGB 和 Mask)
            keys.extend([
                'chest_camera_rgb',
                'head_camera_rgb',
                'chest_camera_mask',
                'head_camera_mask',
            ])
            if self.use_third_camera:
                keys.extend([
                    'third_camera_rgb',
                    'third_camera_mask',
                ])
        elif self.obs_mode == "rgb_masked_rgb":
            # RGB + RGB*Mask (需要同时读取 RGB 和 Mask)
            keys.extend([
                'chest_camera_rgb',
                'head_camera_rgb',
                'chest_camera_mask',
                'head_camera_mask',
            ])
            if self.use_third_camera:
                keys.extend([
                    'third_camera_rgb',
                    'third_camera_mask',
                ])
        elif self.obs_mode == "nd":
            # Normal + Depth
            keys.extend([
                'chest_camera_normals',
                'head_camera_normals',
                'chest_camera_depth',
                'head_camera_depth',
            ])
            if self.use_third_camera:
                keys.extend([
                    'third_camera_normals',
                    'third_camera_depth',
                ])
        elif self.obs_mode == "rgbnd":
            # RGB + Normal + Depth
            keys.extend([
                'chest_camera_rgb',
                'head_camera_rgb',
                'chest_camera_normals',
                'head_camera_normals',
                'chest_camera_depth',
                'head_camera_depth',
            ])
            if self.use_third_camera:
                keys.extend([
                    'third_camera_rgb',
                    'third_camera_normals',
                    'third_camera_depth',
                ])
        
        return keys

    def _process_rgb(self, images: np.ndarray) -> np.ndarray:
        """
        处理 RGB 图像
        
        输入: images [T, H, W, 3] uint8 [0, 255]
        输出: [T, 3, H, W] float32 [0, 1]
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

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        处理 Mask 图像
        
        输入: mask [T, H, W] 或 [T, H, W, 1] uint8 [0, 255]
        输出: [T, 1, H, W] float32 [0, 1] (二值化)
        """
        if mask.ndim == 3:
            mask = mask[..., None]  # [T, H, W, 1]
        
        mask_t = torch.from_numpy(mask).float()  # [T, H, W, 1]
        mask_t = mask_t.permute(0, 3, 1, 2)  # [T, 1, H, W]
        
        mask_t = F.interpolate(
            mask_t / 255.0 if mask_t.max() > 1 else mask_t,
            size=self.image_size,
            mode='nearest'
        )
        mask_t = (mask_t > 0.5).float()  # 二值化
        return mask_t.numpy()

    def _process_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        处理 Depth 图像
        
        存储格式: uint16, 0 表示无效深度, 1-65535 表示有效深度
        解码公式: depth_normalized = (d_u16 - 1) / 65534
        
        输入: depth [T, H, W] 或 [T, H, W, 1] uint16 [0, 65535]
        输出: [T, 1, H, W] float32 [0, 1] (归一化深度)
        """
        if depth.ndim == 4:
            depth = depth[..., 0]  # [T, H, W]
        
        # 转换为 float 并解码
        depth_f = depth.astype(np.float32)
        
        # 无效区域 (0) 保持为 0，有效区域 (1-65535) 映射到 (0, 1]
        valid_mask = depth_f > 0
        depth_normalized = np.zeros_like(depth_f)
        depth_normalized[valid_mask] = (depth_f[valid_mask] - 1.0) / 65534.0
        
        # Resize
        depth_t = torch.from_numpy(depth_normalized).float()  # [T, H, W]
        depth_t = depth_t.unsqueeze(1)  # [T, 1, H, W]
        depth_t = F.interpolate(
            depth_t,
            size=self.image_size,
            mode='nearest'  # 最近邻保持深度边缘
        )
        return depth_t.numpy()

    def _process_normals(self, normals: np.ndarray) -> np.ndarray:
        """
        处理 Normal 图像
        
        存储格式: uint8 [0, 255]
        解码公式: normal = n_u8 / 255.0 * 2.0 - 1.0 (映射到 [-1, 1])
        然后归一化到 [0, 1] 用于网络输入: (normal + 1) / 2
        
        输入: normals [T, H, W, 3] uint8 [0, 255]
        输出: [T, 3, H, W] float32 [0, 1]
        """
        # 直接归一化到 [0, 1]，网络会学习处理
        normals_f = normals.astype(np.float32) / 255.0  # [T, H, W, 3]
        
        normals_t = torch.from_numpy(normals_f)  # [T, H, W, 3]
        normals_t = normals_t.permute(0, 3, 1, 2)  # [T, 3, H, W]
        normals_t = F.interpolate(
            normals_t,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        return normals_t.numpy()

    def _process_rgbm_batch(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        处理 RGBM (RGB + Mask) 图像
        
        输入: 
            rgb [T, H, W, 3] uint8
            mask [T, H, W] 或 [T, H, W, 1] uint8
        输出: [T, 4, H, W] float32
        """
        rgb_processed = self._process_rgb(rgb)  # [T, 3, H, W]
        mask_processed = self._process_mask(mask)  # [T, 1, H, W]
        return np.concatenate([rgb_processed, mask_processed], axis=1)

    def _process_rgb_masked_batch(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        处理 RGB Masked (RGB * Mask) 图像
        只保留 Mask 区域的 RGB，背景置黑
        
        输入:
            rgb [T, H, W, 3] uint8
            mask [T, H, W] 或 [T, H, W, 1] uint8
        输出: [T, 3, H, W] float32
        """
        rgb_processed = self._process_rgb(rgb)  # [T, 3, H, W]
        mask_processed = self._process_mask(mask)  # [T, 1, H, W]
        
        # 将 RGB 与 Mask 相乘，mask为0的地方RGB变为0
        return rgb_processed * mask_processed

    def _process_rgb_masked_rgb_batch(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        处理 RGB + RGB*Mask 图像
        同时输出原始 RGB 和背景置黑的 RGB
        
        输入:
            rgb [T, H, W, 3] uint8
            mask [T, H, W] 或 [T, H, W, 1] uint8
        输出: [T, 6, H, W] float32 (RGB 3通道 + RGB*Mask 3通道)
        """
        rgb_processed = self._process_rgb(rgb)  # [T, 3, H, W]
        mask_processed = self._process_mask(mask)  # [T, 1, H, W]
        
        # RGB * Mask
        rgb_masked = rgb_processed * mask_processed  # [T, 3, H, W]
        
        # 拼接: RGB + RGB*Mask
        return np.concatenate([rgb_processed, rgb_masked], axis=1)

    def _process_nd_batch(self, normals: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        处理 ND (Normal + Depth) 图像
        
        输入:
            normals [T, H, W, 3] uint8
            depth [T, H, W] 或 [T, H, W, 1] uint16
        输出: [T, 4, H, W] float32 (normal 3通道 + depth 1通道)
        """
        normals_processed = self._process_normals(normals)  # [T, 3, H, W]
        depth_processed = self._process_depth(depth)  # [T, 1, H, W]
        return np.concatenate([normals_processed, depth_processed], axis=1)

    def _process_rgbnd_batch(self, rgb: np.ndarray, normals: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        处理 RGBND (RGB + Normal + Depth) 图像
        
        输入:
            rgb [T, H, W, 3] uint8
            normals [T, H, W, 3] uint8
            depth [T, H, W] 或 [T, H, W, 1] uint16
        输出: [T, 7, H, W] float32 (rgb 3通道 + normal 3通道 + depth 1通道)
        """
        rgb_processed = self._process_rgb(rgb)  # [T, 3, H, W]
        normals_processed = self._process_normals(normals)  # [T, 3, H, W]
        depth_processed = self._process_depth(depth)  # [T, 1, H, W]
        return np.concatenate([rgb_processed, normals_processed, depth_processed], axis=1)

    def _sample_to_data(self, sample: dict) -> dict:
        """
        将采样的数据转换为训练格式
        """
        T_slice = slice(self.n_obs_steps)
        
        # 根据 obs_mode 处理图像数据
        if self.obs_mode == "rgb":
            chest_frames = self._process_rgb(sample['chest_camera_rgb'][T_slice])
            head_frames = self._process_rgb(sample['head_camera_rgb'][T_slice])
            if self.use_third_camera:
                third_frames = self._process_rgb(sample['third_camera_rgb'][T_slice])
        elif self.obs_mode == "rgbm":
            chest_frames = self._process_rgbm_batch(
                sample['chest_camera_rgb'][T_slice],
                sample['chest_camera_mask'][T_slice]
            )
            head_frames = self._process_rgbm_batch(
                sample['head_camera_rgb'][T_slice],
                sample['head_camera_mask'][T_slice]
            )
            if self.use_third_camera:
                third_frames = self._process_rgbm_batch(
                    sample['third_camera_rgb'][T_slice],
                    sample['third_camera_mask'][T_slice]
                )
        elif self.obs_mode == "rgb_masked":
            chest_frames = self._process_rgb_masked_batch(
                sample['chest_camera_rgb'][T_slice],
                sample['chest_camera_mask'][T_slice]
            )
            head_frames = self._process_rgb_masked_batch(
                sample['head_camera_rgb'][T_slice],
                sample['head_camera_mask'][T_slice]
            )
            if self.use_third_camera:
                third_frames = self._process_rgb_masked_batch(
                    sample['third_camera_rgb'][T_slice],
                    sample['third_camera_mask'][T_slice]
                )
        elif self.obs_mode == "rgb_masked_rgb":
            chest_frames = self._process_rgb_masked_rgb_batch(
                sample['chest_camera_rgb'][T_slice],
                sample['chest_camera_mask'][T_slice]
            )
            head_frames = self._process_rgb_masked_rgb_batch(
                sample['head_camera_rgb'][T_slice],
                sample['head_camera_mask'][T_slice]
            )
            if self.use_third_camera:
                third_frames = self._process_rgb_masked_rgb_batch(
                    sample['third_camera_rgb'][T_slice],
                    sample['third_camera_mask'][T_slice]
                )
        elif self.obs_mode == "nd":
            chest_frames = self._process_nd_batch(
                sample['chest_camera_normals'][T_slice],
                sample['chest_camera_depth'][T_slice]
            )
            head_frames = self._process_nd_batch(
                sample['head_camera_normals'][T_slice],
                sample['head_camera_depth'][T_slice]
            )
            if self.use_third_camera:
                third_frames = self._process_nd_batch(
                    sample['third_camera_normals'][T_slice],
                    sample['third_camera_depth'][T_slice]
                )
        elif self.obs_mode == "rgbnd":
            chest_frames = self._process_rgbnd_batch(
                sample['chest_camera_rgb'][T_slice],
                sample['chest_camera_normals'][T_slice],
                sample['chest_camera_depth'][T_slice]
            )
            head_frames = self._process_rgbnd_batch(
                sample['head_camera_rgb'][T_slice],
                sample['head_camera_normals'][T_slice],
                sample['head_camera_depth'][T_slice]
            )
            if self.use_third_camera:
                third_frames = self._process_rgbnd_batch(
                    sample['third_camera_rgb'][T_slice],
                    sample['third_camera_normals'][T_slice],
                    sample['third_camera_depth'][T_slice]
                )

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
        
        # 如果使用第三人称相机，添加第三人称相机数据
        if self.use_third_camera:
            obs_data['third_camera_rgb'] = third_frames
        
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
        
        # 添加图像 normalizer（图像已经在处理方法中归一化到 [0, 1]）
        normalizer['chest_camera_rgb'] = get_image_range_normalizer()
        normalizer['head_camera_rgb'] = get_image_range_normalizer()
        
        # 如果使用第三人称相机，添加第三人称相机的 normalizer
        if self.use_third_camera:
            normalizer['third_camera_rgb'] = get_image_range_normalizer()
        
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_all_actions(self) -> torch.Tensor:
        """
        获取所有动作，用于 K-Means 聚类等
        """
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    @property
    def obs_mode_info(self) -> dict:
        """返回当前观测模式的信息"""
        info = {
            "rgb": {"channels": 3, "description": "RGB 3通道"},
            "rgbm": {"channels": 4, "description": "RGB + Mask 4通道"},
            "rgb_masked": {"channels": 3, "description": "RGB * Mask 3通道 (背景置黑)"},
            "rgb_masked_rgb": {"channels": 6, "description": "RGB + RGB*Mask 6通道 (原始RGB + 背景置黑RGB)"},
            "nd": {"channels": 4, "description": "Normal + Depth 4通道"},
            "rgbnd": {"channels": 7, "description": "RGB + Normal + Depth 7通道"},
        }
        return info.get(self.obs_mode, {})
