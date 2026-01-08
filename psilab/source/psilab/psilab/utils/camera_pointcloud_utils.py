"""
扩展 TiledCamera 以支持 get_pointcloud() 方法

使用方法：
1. 在 room_cfg.py 中确保相机包含 "depth" 和 "rgb" 
2. 数据采集时会自动调用 get_pointcloud() 保存点云
"""

import torch
import numpy as np
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd


def add_pointcloud_method_to_camera(camera: TiledCamera):
    """
    为 TiledCamera 添加 get_pointcloud() 方法
    
    Args:
        camera: TiledCamera 实例
    """
    
    def get_pointcloud(self, env_id: int = 0, with_rgb: bool = True) -> np.ndarray:
        """
        从深度图生成点云（类似 IsaacSim Camera 的 get_pointcloud）
        
        Args:
            env_id: 环境ID
            with_rgb: 是否包含RGB颜色
            
        Returns:
            点云数据 (N, 3) 或 (N, 6) [x,y,z] 或 [x,y,z,r,g,b]
        """
        # 检查是否有必需的数据类型
        if "depth" not in self.cfg.data_types:
            raise ValueError("Camera must have 'depth' in data_types to generate pointcloud")
        
        # 获取深度图
        depth = self.data.output["depth"][env_id]  # (H, W, 1) or (H, W)
        if depth.dim() == 3:
            depth = depth.squeeze(-1)  # (H, W)
        
        # 获取相机内参
        intrinsic = self.data.intrinsic_matrices[env_id]  # (3, 3)
        
        if with_rgb and "rgb" in self.cfg.data_types:
            # 获取RGB图像
            rgb = self.data.output["rgb"][env_id]  # (H, W, 3)
            
            # 生成带颜色的点云
            points, colors = create_pointcloud_from_rgbd(
                intrinsic_matrix=intrinsic,
                depth=depth.float(),
                rgb=rgb.float(),
                normalize_rgb=True,  # 归一化到 [0, 1]
                device=depth.device
            )
            
            # 合并为 (N, 6)
            pointcloud = torch.cat([points, colors], dim=-1)
        else:
            # 仅生成XYZ点云
            from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
            points = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic,
                depth=depth.float(),
                device=depth.device
            )
            pointcloud = points  # (N, 3)
        
        # 转换为numpy
        return pointcloud.cpu().numpy()
    
    # 动态添加方法到实例
    camera.get_pointcloud = get_pointcloud.__get__(camera, TiledCamera)
    

def get_pointcloud_from_camera(camera: TiledCamera, env_id: int = 0, with_rgb: bool = True) -> np.ndarray:
    """
    辅助函数：从TiledCamera获取点云
    
    Args:
        camera: TiledCamera 实例
        env_id: 环境ID
        with_rgb: 是否包含RGB颜色
        
    Returns:
        点云数据 (N, 3) 或 (N, 6)
    """
    # 如果相机没有 get_pointcloud 方法，先添加
    if not hasattr(camera, 'get_pointcloud'):
        add_pointcloud_method_to_camera(camera)
    
    return camera.get_pointcloud(env_id, with_rgb)

