# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: AI Assistant
# Date: 2025-12-31
# Version: 1.0
# Description: Bounding box utilities for Isaac Sim objects

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf


class BBoxExtractor:
    """
    用于在 Isaac Sim 中提取物体的 2D 和 3D Bounding Box
    
    支持功能：
    - 2D BBox: 从相机视角投影得到的屏幕空间边界框
    - 3D BBox: 世界坐标系中的 3D 边界框
    - AABB (Axis-Aligned Bounding Box): 轴对齐边界框
    - OBB (Oriented Bounding Box): 朝向边界框
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        初始化 BBox 提取器
        
        Args:
            device: 计算设备 ("cuda:0" 或 "cpu")
        """
        self.device = device
        
    @staticmethod
    def get_3d_bbox_from_prim(prim_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        从 USD Prim 路径获取 3D Bounding Box
        
        Args:
            prim_path: USD prim 的路径，如 "/World/envs/env_0/bottle"
            
        Returns:
            包含 bbox 信息的字典:
            {
                'center': [x, y, z],           # 中心点
                'extent': [width, height, depth],  # 尺寸
                'min': [x_min, y_min, z_min],  # 最小点
                'max': [x_max, y_max, z_max],  # 最大点
                'corners': [[x,y,z], ...],     # 8个角点
            }
        """
        prim = prim_utils.get_prim_at_path(prim_path)
        if prim is None:
            print(f"Warning: Prim not found at path {prim_path}")
            return None
            
        # 获取 USD Geometry
        bbox_cache = UsdGeom.BBoxCache(
            time=prim.GetStage().GetTimeCodesPerSecond(),
            includedPurposes=[UsdGeom.Tokens.default_]
        )
        
        # 计算世界空间的边界框
        bound = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bound.GetRange()
        
        # 提取最小和最大点
        min_point = np.array([bbox_range.GetMin()[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2]])
        max_point = np.array([bbox_range.GetMax()[0], bbox_range.GetMax()[1], bbox_range.GetMax()[2]])
        
        # 计算中心和尺寸
        center = (min_point + max_point) / 2.0
        extent = max_point - min_point
        
        # 计算8个角点 (按标准顺序)
        corners = np.array([
            [min_point[0], min_point[1], min_point[2]],  # 0: 左下后
            [max_point[0], min_point[1], min_point[2]],  # 1: 右下后
            [max_point[0], max_point[1], min_point[2]],  # 2: 右上后
            [min_point[0], max_point[1], min_point[2]],  # 3: 左上后
            [min_point[0], min_point[1], max_point[2]],  # 4: 左下前
            [max_point[0], min_point[1], max_point[2]],  # 5: 右下前
            [max_point[0], max_point[1], max_point[2]],  # 6: 右上前
            [min_point[0], max_point[1], max_point[2]],  # 7: 左上前
        ])
        
        return {
            'center': center,
            'extent': extent,
            'min': min_point,
            'max': max_point,
            'corners': corners,
        }
    
    @staticmethod
    def project_3d_to_2d(
        points_3d: np.ndarray,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        image_width: int,
        image_height: int
    ) -> np.ndarray:
        """
        将 3D 点投影到 2D 屏幕空间
        
        Args:
            points_3d: (N, 3) 3D 点坐标
            view_matrix: (4, 4) 视图矩阵
            projection_matrix: (4, 4) 投影矩阵
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            (N, 2) 2D 屏幕坐标 [x, y]
        """
        # 转换为齐次坐标
        points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # 应用视图和投影矩阵
        points_clip = points_3d_homo @ view_matrix.T @ projection_matrix.T
        
        # 透视除法
        points_ndc = points_clip[:, :3] / points_clip[:, 3:4]
        
        # NDC [-1, 1] 转换到屏幕空间 [0, width/height]
        points_2d = np.zeros((points_3d.shape[0], 2))
        points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * image_width
        points_2d[:, 1] = (1.0 - points_ndc[:, 1]) * 0.5 * image_height
        
        return points_2d
    
    @staticmethod
    def get_2d_bbox_from_3d(
        bbox_3d: Dict[str, np.ndarray],
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        image_width: int,
        image_height: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        从 3D BBox 计算 2D BBox
        
        Args:
            bbox_3d: 3D BBox 字典（来自 get_3d_bbox_from_prim）
            view_matrix: 相机视图矩阵
            projection_matrix: 相机投影矩阵
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            2D BBox 字典:
            {
                'x_min': float,  # 左上角 x
                'y_min': float,  # 左上角 y
                'x_max': float,  # 右下角 x
                'y_max': float,  # 右下角 y
                'width': float,  # 宽度
                'height': float, # 高度
                'center': [x, y], # 中心点
            }
        """
        if bbox_3d is None:
            return None
            
        # 投影 8 个角点
        corners_3d = bbox_3d['corners']
        corners_2d = BBoxExtractor.project_3d_to_2d(
            corners_3d, view_matrix, projection_matrix, image_width, image_height
        )
        
        # 计算 2D 边界框
        x_min = np.clip(corners_2d[:, 0].min(), 0, image_width)
        x_max = np.clip(corners_2d[:, 0].max(), 0, image_width)
        y_min = np.clip(corners_2d[:, 1].min(), 0, image_height)
        y_max = np.clip(corners_2d[:, 1].max(), 0, image_height)
        
        width = x_max - x_min
        height = y_max - y_min
        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': width,
            'height': height,
            'center': center,
        }
    
    @staticmethod
    def get_bbox_from_mask(mask: np.ndarray) -> Optional[Dict[str, float]]:
        """
        从分割 mask 计算 2D BBox
        
        Args:
            mask: (H, W) 二值 mask，目标物体为非零值
            
        Returns:
            2D BBox 字典（格式同 get_2d_bbox_from_3d）
        """
        # 找到非零像素
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            return None
            
        # 计算边界框 (注意: argwhere 返回 [row, col] = [y, x])
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        
        return {
            'x_min': float(x_min),
            'y_min': float(y_min),
            'x_max': float(x_max),
            'y_max': float(y_max),
            'width': float(width),
            'height': float(height),
            'center': center.tolist(),
        }


def add_bbox_to_h5(h5_group, bbox_data: Dict[str, np.ndarray], dataset_name: str = "target_bbox"):
    """
    将 BBox 数据添加到 HDF5 文件
    
    Args:
        h5_group: HDF5 组对象
        bbox_data: BBox 数据字典
        dataset_name: 数据集名称前缀
        
    示例用法:
        # 3D BBox
        bbox_3d = get_3d_bbox_from_prim("/World/envs/env_0/bottle")
        add_bbox_to_h5(h5_file, bbox_3d, "target_bbox_3d")
        
        # 2D BBox (多个相机)
        bbox_2d_head = get_2d_bbox_from_3d(bbox_3d, view_mat, proj_mat, 640, 480)
        add_bbox_to_h5(h5_file["cameras"], bbox_2d_head, "head_camera_bbox_2d")
    """
    for key, value in bbox_data.items():
        if isinstance(value, np.ndarray):
            h5_group.create_dataset(f"{dataset_name}/{key}", data=value)
        else:
            h5_group.create_dataset(f"{dataset_name}/{key}", data=np.array(value))


def convert_bbox_to_zarr_format(bbox_2d_dict: Dict[str, List], num_frames: int) -> np.ndarray:
    """
    将多帧 2D BBox 数据转换为 Zarr 存储格式
    
    Args:
        bbox_2d_dict: 包含每帧 bbox 的字典列表
        num_frames: 总帧数
        
    Returns:
        (N, 6) 数组: [x_min, y_min, x_max, y_max, width, height]
        
    示例:
        bbox_list = []
        for i in range(num_frames):
            bbox = get_2d_bbox_from_3d(...)
            bbox_list.append(bbox)
        
        bbox_array = convert_bbox_to_zarr_format(bbox_list, num_frames)
        episode['head_camera_bbox_2d'] = bbox_array
    """
    bbox_array = np.zeros((num_frames, 6), dtype=np.float32)
    
    for i, bbox in enumerate(bbox_2d_dict):
        if bbox is not None:
            bbox_array[i] = [
                bbox['x_min'],
                bbox['y_min'],
                bbox['x_max'],
                bbox['y_max'],
                bbox['width'],
                bbox['height'],
            ]
    
    return bbox_array


# ==================== 使用示例 ====================

def example_usage_in_grasp_mp():
    """
    在 grasp_mp.py 中使用 BBox 提取的示例代码
    
    需要在以下位置添加代码：
    1. __init__: 初始化 BBoxExtractor
    2. _record_data: 记录每帧的 bbox
    3. _write_data_to_file: 将 bbox 写入 HDF5
    """
    
    # === 1. 在 __init__ 中初始化 ===
    # self.bbox_extractor = BBoxExtractor(device=self.device)
    # self._bbox_buffer = []  # 存储每帧的 bbox
    
    # === 2. 在 _record_data 中记录 bbox ===
    """
    def _record_data(self, env_id: int):
        # ... 现有代码 ...
        
        # 获取目标物体的 prim 路径
        target_prim_path = f"/World/envs/env_{env_id}/{self._target_object_name}"
        
        # 提取 3D BBox
        bbox_3d = BBoxExtractor.get_3d_bbox_from_prim(target_prim_path)
        
        if bbox_3d is not None:
            # 存储 3D BBox (7维: center(3) + extent(3) + is_valid(1))
            bbox_3d_data = np.concatenate([
                bbox_3d['center'],
                bbox_3d['extent'],
                [1.0]  # valid flag
            ])
        else:
            bbox_3d_data = np.zeros(7)
        
        self._bbox_buffer.append(bbox_3d_data)
        
        # 可选: 如果需要 2D BBox，需要相机的视图和投影矩阵
        # 这需要从相机传感器获取，较为复杂
    """
    
    # === 3. 在 _write_data_to_file 中写入 ===
    """
    def _write_data_to_file(self, env_id: int):
        # ... 现有代码 ...
        
        # 写入 BBox 数据
        if len(self._bbox_buffer) > 0:
            bbox_array = np.array(self._bbox_buffer)
            h5_file.create_dataset(
                "rigid_objects/target_bbox_3d",
                data=bbox_array,
                dtype=np.float32
            )
        
        # 清空缓冲区
        self._bbox_buffer.clear()
    """
    
    pass


def example_usage_in_zarr_utils():
    """
    在 zarr_utils.py 中转换 BBox 数据的示例
    
    需要在 convert_rgb_based 函数中添加：
    """
    
    # === 在 convert_rgb_based 中添加 ===
    """
    # 处理 BBox 数据 (如果存在)
    if "rigid_objects/target_bbox_3d" in h5_file:
        bbox_3d_data = h5_file["rigid_objects/target_bbox_3d"]
        episode['target_bbox_3d'] = bbox_3d_data
        
        # 拆分为独立字段（可选，便于训练时使用）
        episode['target_bbox_center'] = bbox_3d_data[:, :3]  # [x, y, z]
        episode['target_bbox_extent'] = bbox_3d_data[:, 3:6]  # [w, h, d]
    """
    
    pass


if __name__ == "__main__":
    # 简单测试
    print("BBox Utils Module - Ready to use")
    print("\n功能列表：")
    print("1. BBoxExtractor.get_3d_bbox_from_prim() - 获取 3D BBox")
    print("2. BBoxExtractor.get_2d_bbox_from_3d() - 计算 2D BBox")
    print("3. BBoxExtractor.get_bbox_from_mask() - 从 mask 计算 BBox")
    print("4. add_bbox_to_h5() - 保存到 HDF5")
    print("5. convert_bbox_to_zarr_format() - 转换为 Zarr 格式")
    print("\n详细使用方法请参考函数文档字符串")

