# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import argparse
import h5py
import sys
from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property
import cv2
from datetime import datetime

# Import Isaac Lab's pointcloud utilities
try:
    from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    print("Warning: Isaac Lab not available, will use fallback pointcloud generation")

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert HDF5 data to Zarr format for diffusion policy training.")
parser.add_argument("--h5_dir", type=str, default="", help="Directory containing HDF5 files.")
parser.add_argument("--zarr_dir", type=str, default="", help="Output directory for Zarr files.")
parser.add_argument("--mode", type=str, default="state", choices=["state", "rgb"], 
                    help="Conversion mode: 'state' for pure state-based, 'rgb' for state + camera images.")
parser.add_argument("--with_mask", action="store_true", 
                    help="Whether to concatenate mask channel to RGB images (4 channels: RGBM).")
parser.add_argument("--with_depth", action="store_true", 
                    help="Whether to include depth images.")
parser.add_argument("--with_normals", action="store_true", 
                    help="Whether to include normal images.")
parser.add_argument("--save_rgb_separate", action="store_true",
                    help="Whether to save RGB images as separate 3-channel datasets for inspection.")
parser.add_argument("--save_mask_separate", action="store_true",
                    help="Whether to save mask images as separate 1-channel datasets for inspection.")
parser.add_argument("--with_pointcloud", action="store_true",
                    help="Whether to generate point clouds from depth images.")
parser.add_argument("--num_points", type=int, default=1024,
                    help="Number of points to sample from point cloud (default: 1024).")
parser.add_argument("--task_type", type=str, default="auto", 
                    choices=["auto", "single_hand", "bimanual"],
                    help="Task type: 'auto' (auto-detect), 'single_hand' (grasp/pick_place), 'bimanual' (handover).")
parser.add_argument("--max_episodes", type=int, default=50,
                    help="Maximum number of episodes to convert (default: 50).")

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0 # type: ignore


def depth_to_pointcloud_isaaclab(depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float, 
                                  rgb_image: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                                  near: float = 0.2, far: float = 1.8) -> np.ndarray:
    """
    使用 Isaac Lab 官方工具从深度图转换为点云（推荐）
    
    Args:
        depth_map: 深度图 (H, W) uint16格式，使用near/far范围编码
        fx, fy: 相机焦距
        cx, cy: 相机光心
        rgb_image: 可选的RGB图像 (H, W, 3) uint8
        mask: 可选的mask (H, W) uint8，仅提取mask区域的点云
        near, far: 深度范围
        
    Returns:
        point_cloud: (N, 3) 或 (N, 6) 如果提供了RGB，格式为 [x, y, z] 或 [x, y, z, r, g, b]
    """
    H, W = depth_map.shape
    
    # 1. 将uint16深度解码回真实深度值 (float32)
    depth_float = np.zeros_like(depth_map, dtype=np.float32)
    valid_mask = depth_map > 0
    if np.sum(valid_mask) > 0:
        depth_norm = (depth_map[valid_mask].astype(np.float32) - 1.0) / 65534.0
        depth_float[valid_mask] = depth_norm * (far - near) + near
    
    # 2. 构建相机内参矩阵
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 3. 使用 Isaac Lab 工具生成点云
    if rgb_image is not None:
        # 生成带颜色的点云
        rgb_float = rgb_image.astype(np.float32)  # (H, W, 3)
        points, colors = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth_float,
            rgb=rgb_float,
            normalize_rgb=True,  # 归一化到 [0, 1]
            device="cpu"
        )
        # points: (N, 3), colors: (N, 3)
        points = points.numpy() if hasattr(points, 'numpy') else points
        colors = colors.numpy() if hasattr(colors, 'numpy') else colors
    else:
        # 仅生成XYZ点云
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        points = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth_float,
            device="cpu"
        )
        points = points.numpy() if hasattr(points, 'numpy') else points
        colors = None
    
    # 4. 应用 mask 过滤
    if mask is not None:
        # Isaac Lab 已经生成了所有有效深度点的点云
        # 我们需要找出哪些点对应mask区域
        # 这需要将点云投影回图像平面
        u = (points[:, 0] * fx / points[:, 2] + cx).astype(np.int32)
        v = (points[:, 1] * fy / points[:, 2] + cy).astype(np.int32)
        
        # 确保在图像范围内
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = np.clip(u, 0, W-1)
        v = np.clip(v, 0, H-1)
        
        # 检查mask
        mask_values = mask[v, u]
        keep_mask = valid & (mask_values > 0)
        
        points = points[keep_mask]
        if colors is not None:
            colors = colors[keep_mask]
    
    # 5. 组合结果
    # if colors is not None:
    #     result = np.concatenate([points, colors], axis=1)  # (N, 6)
    # else:
        # 用零填充RGB通道，保持和 read_point_cloud.py 一致
    zeros_rgb = np.zeros((points.shape[0], 3), dtype=np.float32)  # (N, 3) [0, 0, 0]
    result = np.concatenate([points, zeros_rgb], axis=1)  # (N, 6) [x, y, z, 0, 0, 0]

    return result


def depth_to_pointcloud(depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float, 
                        rgb_image: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                        near: float = 0.2, far: float = 1.8) -> np.ndarray:
    """
    将深度图转换为点云（优先使用 Isaac Lab 官方实现）
    
    Args:
        depth_map: 深度图 (H, W) uint16格式，使用near/far范围编码
        fx, fy: 相机焦距
        cx, cy: 相机光心
        rgb_image: 可选的RGB图像 (H, W, 3) uint8
        mask: 可选的mask (H, W) uint8，仅提取mask区域的点云
        near, far: 深度范围
        
    Returns:
        point_cloud: (N, 3) 或 (N, 6) 如果提供了RGB，格式为 [x, y, z] 或 [x, y, z, r, g, b]
    """
    if ISAACLAB_AVAILABLE:
        return depth_to_pointcloud_isaaclab(depth_map, fx, fy, cx, cy, rgb_image, mask, near, far)
    else:
        return depth_to_pointcloud_fallback(depth_map, fx, fy, cx, cy, rgb_image, mask, near, far)


def depth_to_pointcloud_fallback(depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float, 
                                  rgb_image: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                                  near: float = 0.2, far: float = 1.8) -> np.ndarray:
    """
    将深度图转换为点云（备用实现，不依赖 Isaac Lab）
    
    Args:
        depth_map: 深度图 (H, W) uint16格式，使用near/far范围编码
        fx, fy: 相机焦距
        cx, cy: 相机光心
        rgb_image: 可选的RGB图像 (H, W, 3) uint8
        mask: 可选的mask (H, W) uint8，仅提取mask区域的点云
        near, far: 深度范围
        
    Returns:
        point_cloud: (N, 3) 或 (N, 6) 如果提供了RGB，格式为 [x, y, z] 或 [x, y, z, r, g, b]
    """
    H, W = depth_map.shape
    
    # 将uint16深度解码回真实深度值
    depth_float = np.zeros_like(depth_map, dtype=np.float32)
    valid_mask = depth_map > 0
    if np.sum(valid_mask) > 0:
        depth_norm = (depth_map[valid_mask].astype(np.float32) - 1.0) / 65534.0
        depth_float[valid_mask] = depth_norm * (far - near) + near
    
    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 应用mask过滤
    if mask is not None:
        object_mask = (mask > 0) & valid_mask & (depth_float > 0)
    else:
        object_mask = valid_mask & (depth_float > 0)
    
    if np.sum(object_mask) == 0:
        # 返回空点云
        return np.zeros((0, 6 if rgb_image is not None else 3), dtype=np.float32)
    
    # 提取有效点
    u_valid = u[object_mask]
    v_valid = v[object_mask]
    z_valid = depth_float[object_mask]
    
    # 反投影到3D空间
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid
    
    # 组合xyz
    points_3d = np.stack([x, y, z], axis=-1)  # (N, 3)
    
    # 如果提供了RGB，添加颜色信息；否则用零填充
    # if rgb_image is not None:
    #     rgb_valid = rgb_image[object_mask]  # (N, 3) uint8
    #     rgb_normalized = rgb_valid.astype(np.float32) / 255.0  # 归一化到[0,1]
    #     points_3d = np.concatenate([points_3d, rgb_normalized], axis=-1)  # (N, 6)
    # else:
        # 用零填充RGB通道，保持和 read_point_cloud.py 一致
    zeros_rgb = np.zeros((points_3d.shape[0], 3), dtype=np.float32)  # (N, 3) [0, 0, 0]
    points_3d = np.concatenate([points_3d, zeros_rgb], axis=-1)  # (N, 6) [x, y, z, 0, 0, 0]

    return points_3d


def furthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    最远点采样 (Furthest Point Sampling)
    
    Args:
        points: 点云 (N, D) 其中D可以是3或6
        n_samples: 采样点数
        
    Returns:
        sampled_points: 采样后的点云 (n_samples, D)
    """
    N = points.shape[0]
    
    if N == 0:
        return np.zeros((n_samples, points.shape[1]), dtype=np.float32)
    
    if N <= n_samples:
        # 不足采样数，用零填充
        padded = np.zeros((n_samples, points.shape[1]), dtype=np.float32)
        padded[:N] = points
        return padded
    
    # FPS算法（仅使用xyz坐标）
    xyz = points[:, :3]
    
    centroids = np.zeros(n_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return points[centroids]

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if isinstance(value, zarr.Group):
                    continue
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)


def convert_state_based(h5_file, h5_temp, task_type: str = "auto") -> dict:
    """
    纯状态模式转换
    
    支持手动指定任务类型或自动检测：
    - 单手任务（Grasp, Pick Place）:
      - action: arm2_pos_target(7) + hand2_pos_target(6) = 13维
      - state: bottle_pose(7) + arm2_pos(7) + hand2_pos(6) = 20维
    
    - 双手任务（Handover）:
      - action: arm2_pos_target(7) + hand2_pos_target(6) + arm1_pos_target(7) + hand1_pos_target(6) = 26维
      - state: bottle_pose(7) + arm2_pos(7) + hand2_pos(6) + arm1_pos(7) + hand1_pos(6) = 33维
    
    Args:
        h5_file: HDF5文件对象
        h5_temp: 临时HDF5文件对象
        task_type: 任务类型 ("auto", "single_hand", "bimanual")
    """
    episode = dict()
    
    episode['timestamps'] = h5_file["sim_time"]
    
    # 检测是否为双手任务
    if task_type == "auto":
        is_bimanual = ("arm1_pos_target" in h5_file["robots"]["robot"]) and \
                      ("hand1_pos_target" in h5_file["robots"]["robot"])
        print(f"  自动检测: {'双手任务' if is_bimanual else '单手任务'}")
    elif task_type == "bimanual":
        is_bimanual = True
        print(f"  手动指定: 双手任务")
    else:  # single_hand
        is_bimanual = False
        print(f"  手动指定: 单手任务")
    
    if is_bimanual:
        # ========== 双手任务（Handover） ==========
        
        # Action: arm2(7) + hand2(6) + arm1(7) + hand1(6) = 26维
        h5_temp.create_dataset(
            "action",
            shape=(h5_file["robots"]["robot"]["arm2_pos_target"].shape[0], 26),
            dtype=h5_file["robots"]["robot"]["arm2_pos_target"].dtype
        )
        h5_temp["action"][:, :7] = h5_file["robots"]["robot"]["arm2_pos_target"]
        h5_temp["action"][:, 7:13] = h5_file["robots"]["robot"]["hand2_pos_target"][:, :6]
        h5_temp["action"][:, 13:20] = h5_file["robots"]["robot"]["arm1_pos_target"]
        h5_temp["action"][:, 20:26] = h5_file["robots"]["robot"]["hand1_pos_target"][:, :6]
        
        # State: bottle_pose(7) + arm2_pos(7) + hand2_pos(6) + arm1_pos(7) + hand1_pos(6) = 33维
        h5_temp.create_dataset(
            "state",
            shape=(h5_file["robots"]["robot"]["arm2_pos"].shape[0], 33),
            dtype=h5_file["robots"]["robot"]["arm2_pos"].dtype
        )
        h5_temp["state"][:, :7] = h5_file["rigid_objects"]["bottle"]
        h5_temp["state"][:, 7:14] = h5_file["robots"]["robot"]["arm2_pos"]
        h5_temp["state"][:, 14:20] = h5_file["robots"]["robot"]["hand2_pos"][:, :6]
        h5_temp["state"][:, 20:27] = h5_file["robots"]["robot"]["arm1_pos"]
        h5_temp["state"][:, 27:33] = h5_file["robots"]["robot"]["hand1_pos"][:, :6]
        
    else:
        # ========== 单手任务（Grasp, Pick Place） ==========
        
        # Action: arm2_pos_target(7) + hand2_pos_target(6) = 13维
        h5_temp.create_dataset(
            "action",
            shape=(h5_file["robots"]["robot"]["arm2_pos_target"].shape[0], 13),
            dtype=h5_file["robots"]["robot"]["arm2_pos_target"].dtype
        )
        h5_temp["action"][:, :7] = h5_file["robots"]["robot"]["arm2_pos_target"]
        h5_temp["action"][:, 7:] = h5_file["robots"]["robot"]["hand2_pos_target"][:, :6]
        
        # State: bottle_pose(7) + arm2_pos(7) + hand2_pos(6) = 20维
        h5_temp.create_dataset(
            "state",
            shape=(h5_file["robots"]["robot"]["arm2_pos"].shape[0], 20),
            dtype=h5_file["robots"]["robot"]["arm2_pos"].dtype
        )
        h5_temp["state"][:, :7] = h5_file["rigid_objects"]["bottle"]
        h5_temp["state"][:, 7:14] = h5_file["robots"]["robot"]["arm2_pos"]
        h5_temp["state"][:, 14:] = h5_file["robots"]["robot"]["hand2_pos"][:, :6]
    
    episode['action'] = h5_temp["action"]
    episode['state'] = h5_temp["state"]
    
    return episode


def convert_rgb_based(h5_file, h5_temp, image_size: int = 224, with_mask: bool = False, 
                       with_depth: bool = False, with_normals: bool = False,
                       save_rgb_separate: bool = False, save_mask_separate: bool = False,
                       with_pointcloud: bool = False, num_points: int = 1024,
                       task_type: str = "auto") -> dict:
    """
    RGB图像 + 状态模式转换
    
    自动检测任务类型（单手 or 双手）：
    
    单手任务（Grasp, Pick Place）包含数据：
    - timestamps: 仿真时间
    - action: arm2_pos_target(7) + hand2_pos_target(6) = 13维
    - arm2_pos, arm2_vel, hand2_pos, hand2_vel
    - arm2_eef_pos(3), arm2_eef_quat(4)
    - target_pose(7)
    - head_camera_rgb, chest_camera_rgb, third_camera_rgb (N, 224, 224, 3)
    
    双手任务（Handover）额外包含：
    - action: arm2(7) + hand2(6) + arm1(7) + hand1(6) = 26维
    - arm1_pos, arm1_vel, hand1_pos, hand1_vel
    - arm1_eef_pos(3), arm1_eef_quat(4)
    
    可选数据（根据参数）：
    - (可选) *_camera_mask (N, 224, 224) uint8
    - (可选) *_camera_depth (N, 224, 224, 1) uint16
    - (可选) *_camera_normals (N, 224, 224, 3) uint8
    - (可选) *_camera_pointcloud (N, num_points, 6) float32 [x,y,z,r,g,b]
      注意：如果HDF5中已有点云数据，直接使用；否则从深度图生成
    
    Args:
        h5_file: HDF5文件对象
        h5_temp: 临时HDF5文件对象
        image_size: 输出图像尺寸
        with_mask: 是否存储mask通道
        with_depth: 是否存储深度图
        with_normals: 是否存储法线图
        save_rgb_separate: 是否单独存储RGB图像（用于检测）
        save_mask_separate: 是否单独存储mask图像（用于检测）
        with_pointcloud: 是否存储点云（如HDF5中有则直接读取，否则从深度图生成）
        num_points: 点云采样点数
        task_type: 任务类型 ("auto", "grasp", "pick_place", "handover")
                   - "auto": 自动检测（通过检查 arm1_pos_target 是否存在）
                   - "grasp" 或 "pick_place": 强制单手任务（只保存右手数据）
                   - "handover": 强制双手任务（保存双手数据）
    """
    episode = dict()
    
    episode['timestamps'] = h5_file["sim_time"]
    
    # 检测是否为双手任务
    if task_type == "auto":
        is_bimanual = ("arm1_pos_target" in h5_file["robots"]["robot"]) and \
                      ("hand1_pos_target" in h5_file["robots"]["robot"])
        print(f"  自动检测: {'双手任务' if is_bimanual else '单手任务'}")
    elif task_type == "bimanual":
        is_bimanual = True
        print(f"  手动指定: 双手任务")
    else:  # single_hand
        is_bimanual = False
        print(f"  手动指定: 单手任务")
    
    if is_bimanual:
        # ========== 双手任务（Handover） ==========
        
        # Action: arm2(7) + hand2(6) + arm1(7) + hand1(6) = 26维
        h5_temp.create_dataset(
            "action",
            shape=(h5_file["robots"]["robot"]["arm2_pos_target"].shape[0], 26),
            dtype=h5_file["robots"]["robot"]["arm2_pos_target"].dtype
        )
        h5_temp["action"][:, :7] = h5_file["robots"]["robot"]["arm2_pos_target"]
        h5_temp["action"][:, 7:13] = h5_file["robots"]["robot"]["hand2_pos_target"][:, :6]
        h5_temp["action"][:, 13:20] = h5_file["robots"]["robot"]["arm1_pos_target"]
        h5_temp["action"][:, 20:26] = h5_file["robots"]["robot"]["hand1_pos_target"][:, :6]
        
        episode['action'] = h5_temp["action"]
        
        # 右手状态
        episode['arm2_pos'] = h5_file["robots"]["robot"]["arm2_pos"]
        episode['arm2_vel'] = h5_file["robots"]["robot"]["arm2_vel"]
        episode['hand2_pos'] = h5_file["robots"]["robot"]["hand2_pos"][:, :6]
        episode['hand2_vel'] = h5_file["robots"]["robot"]["hand2_vel"][:, :6]
        episode['arm2_eef_pos'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, :3]
        episode['arm2_eef_quat'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, 3:]
        
        # 左手状态
        episode['arm1_pos'] = h5_file["robots"]["robot"]["arm1_pos"]
        episode['arm1_vel'] = h5_file["robots"]["robot"]["arm1_vel"]
        episode['hand1_pos'] = h5_file["robots"]["robot"]["hand1_pos"][:, :6]
        episode['hand1_vel'] = h5_file["robots"]["robot"]["hand1_vel"][:, :6]
        episode['arm1_eef_pos'] = h5_file["robots"]["robot"]["arm1_eef_pose"][:, :3]
        episode['arm1_eef_quat'] = h5_file["robots"]["robot"]["arm1_eef_pose"][:, 3:]
        
    else:
        # ========== 单手任务（Grasp, Pick Place） ==========
        
        # Action: arm2_pos_target(7) + hand2_pos_target(6) = 13维
        h5_temp.create_dataset(
            "action",
            shape=(h5_file["robots"]["robot"]["arm2_pos_target"].shape[0], 13),
            dtype=h5_file["robots"]["robot"]["arm2_pos_target"].dtype
        )
        h5_temp["action"][:, :7] = h5_file["robots"]["robot"]["arm2_pos_target"]
        h5_temp["action"][:, 7:] = h5_file["robots"]["robot"]["hand2_pos_target"][:, :6]
        
        episode['action'] = h5_temp["action"]
        
        # 右手状态
        episode['arm2_pos'] = h5_file["robots"]["robot"]["arm2_pos"]
        episode['arm2_vel'] = h5_file["robots"]["robot"]["arm2_vel"]
        episode['hand2_pos'] = h5_file["robots"]["robot"]["hand2_pos"][:, :6]
        episode['hand2_vel'] = h5_file["robots"]["robot"]["hand2_vel"][:, :6]
        episode['arm2_eef_pos'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, :3]
        episode['arm2_eef_quat'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, 3:]
    
    # 物体位姿（所有任务都有）
    episode['target_pose'] = h5_file["rigid_objects"]["bottle"][:, :7]
    
    # ========== 处理 Ground Truth 点云 ==========
    if with_pointcloud:
        # 从顶层读取真值点云（新数据格式）
        if "ground_truth_pointcloud" in h5_file:
            print(f"  ✅ 读取 Ground Truth 点云...")
            gt_pc_data = np.array(h5_file["ground_truth_pointcloud"])
            print(f"    原始形状: {gt_pc_data.shape}")
            
            # 采样到固定数量（如果需要）
            if gt_pc_data.shape[1] != num_points:
                print(f"    采样到 {num_points} 点...")
                gt_pointcloud = np.zeros((gt_pc_data.shape[0], num_points, 3), dtype=np.float32)
                for i in range(gt_pc_data.shape[0]):
                    pc = gt_pc_data[i]  # (N, 3) [x, y, z]
                    
                    # 最远点采样
                    if pc.shape[0] > 0:
                        # 对于3D点云，需要先添加dummy维度用于FPS
                        pc_6d = np.hstack([pc, np.zeros((pc.shape[0], 3), dtype=np.float32)])
                        pc_sampled = furthest_point_sampling(pc_6d, num_points)
                        gt_pointcloud[i] = pc_sampled[:, :3]  # 只保留xyz
            else:
                # 已经是正确的数量，直接使用
                gt_pointcloud = gt_pc_data.astype(np.float32)
            
            # 保存到episode
            h5_temp.create_dataset("ground_truth_pointcloud", data=gt_pointcloud)
            episode["ground_truth_pointcloud"] = h5_temp["ground_truth_pointcloud"]
            print(f"    Ground Truth 点云已保存: {gt_pointcloud.shape}")
        else:
            print(f"  ⚠️  HDF5中未找到 ground_truth_pointcloud（可能是旧数据）")
    
    # 处理相机图像（包括第三个相机）
    camera_names = ["head_camera.rgb", "chest_camera.rgb", "third_camera.rgb"]
    mask_names = ["head_camera.instance_segmentation_fast", "chest_camera.instance_segmentation_fast", "third_camera.instance_segmentation_fast"]
    
    # 用于收集所有相机的点云，稍后组合
    all_camera_pointclouds = {}
    
    for cam_name, mask_name in zip(camera_names, mask_names):
        if cam_name in h5_file["robots"]["robot"]:
            base_name = cam_name.replace(".rgb", "")
            
            # --- 1. 读取原始 RGB ---
            rgb_data = np.array(h5_file["robots"]["robot"][cam_name])
            
            # 准备数据容器
            rgb_resized = np.zeros((rgb_data.shape[0], image_size, image_size, 3), dtype=np.uint8)
            mask_resized = None
            depth_resized = None # uint16
            normals_resized = None # uint8
            pointcloud_data = None # float32 (N, num_points, 6)
            # depth_raw_resized = None # float32 - 移除
            # normals_raw_resized = None # float32 - 移除
            
            # 相机内参 (640x480分辨率)
            # 根据实际相机配置设置不同的内参
            original_width, original_height = 640, 480
            
            # 为不同相机设置正确的内参（基于真实测量值）
            if "head_camera" in base_name:
                fx_original = 615.8730
                fy_original = 615.8730
            elif "chest_camera" in base_name:
                fx_original = 316.1445
                fy_original = 421.5259
            elif "third_camera" in base_name:
                fx_original = 615.8730
                fy_original = 615.8730
            else:
                # 默认值（如果有其他相机）
                fov_horizontal = 69.0  # 度
                fx_original = original_width / (2.0 * np.tan(np.radians(fov_horizontal) / 2.0))
                fy_original = fx_original
                print(f"Warning: Using default intrinsics for camera: {base_name}")
            
            cx_original = 320.0
            cy_original = 240.0
            
            # 缩放内参到resize后的尺寸
            scale_x = image_size / original_width
            scale_y = image_size / original_height
            fx = fx_original * scale_x
            fy = fy_original * scale_y
            cx = cx_original * scale_x
            cy = cy_original * scale_y
            
            # Resize RGB
            for i in range(rgb_data.shape[0]):
                img = rgb_data[i]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_res_bgr = cv2.resize(img_bgr, (image_size, image_size))
                img_res_rgb = cv2.cvtColor(img_res_bgr, cv2.COLOR_BGR2RGB)
                rgb_resized[i] = img_res_rgb

            # --- 2. 处理 Mask ---
            if with_mask and mask_name in h5_file["robots"]["robot"]:
                mask_data = np.array(h5_file["robots"]["robot"][mask_name])
                mask_resized = np.zeros((rgb_data.shape[0], image_size, image_size), dtype=np.uint8)
                for i in range(mask_data.shape[0]):
                    m = mask_data[i]
                    if m.ndim == 3: m = m[:,:,0]
                    m_bin = (m > 0).astype(np.uint8) * 255
                    m_res = cv2.resize(m_bin, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                    mask_resized[i] = m_res

            # --- 3. 处理 Depth ---
            if with_depth:
                depth_key = cam_name.replace(".rgb", ".depth")
                if depth_key in h5_file["robots"]["robot"]:
                    depth_data = np.array(h5_file["robots"]["robot"][depth_key])
                    depth_resized = np.zeros((rgb_data.shape[0], image_size, image_size), dtype=np.uint16)
                    # depth_raw_resized = np.zeros((rgb_data.shape[0], image_size, image_size), dtype=np.float32)
                    
                    near, far = 0.2, 1.8
                    for i in range(depth_data.shape[0]):
                        d = depth_data[i]
                        d = d.squeeze()
                        
                        # 原始 Resize (已移除组合逻辑，无需保留 raw)
                        # d_res_raw = cv2.resize(d, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                        # depth_raw_resized[i] = d_res_raw
                        
                        # 压缩存储 (uint16)
                        d_valid = np.isfinite(d) & (d > 0)
                        d_u16 = np.zeros_like(d, dtype=np.uint16)
                        if np.sum(d_valid) > 0:
                            d_clip = np.clip(d[d_valid], near, far)
                            d_norm = (d_clip - near) / (far - near)
                            d_u16[d_valid] = (d_norm * 65534.0 + 1.0 + 0.5).astype(np.uint16)
                        
                        d_res_u16 = cv2.resize(d_u16, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                        depth_resized[i] = d_res_u16
                else:
                    print(f"Warning: Depth requested but {depth_key} not found.")

            # --- 4. 处理 Normals ---
            if with_normals:
                normals_key = cam_name.replace(".rgb", ".normals")
                if normals_key in h5_file["robots"]["robot"]:
                    normals_data = np.array(h5_file["robots"]["robot"][normals_key])
                    normals_resized = np.zeros((rgb_data.shape[0], image_size, image_size, 3), dtype=np.uint8)
                    # normals_raw_resized = np.zeros((rgb_data.shape[0], image_size, image_size, 3), dtype=np.float32)
                    
                    for i in range(normals_data.shape[0]):
                        n = normals_data[i]
                        
                        # 原始 Resize (已移除组合逻辑，无需保留 raw)
                        # n_res_raw = cv2.resize(n, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                        # normals_raw_resized[i] = n_res_raw
                        
                        # 压缩存储 (uint8)
                        n_map = (n + 1.0) / 2.0
                        n_map = np.clip(n_map, 0.0, 1.0)
                        n_u8 = (n_map * 255.0 + 0.5).astype(np.uint8)
                        n_res_u8 = cv2.resize(n_u8, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                        normals_resized[i] = n_res_u8
                else:
                    print(f"Warning: Normals requested but {normals_key} not found.")

            # --- 5. 处理/生成点云 ---
            if with_pointcloud:
                # 优先使用HDF5中已有的点云数据（通过 camera.get_pointcloud() 采集的）
                pointcloud_key = cam_name.replace(".rgb", ".pointcloud")
                
                if pointcloud_key in h5_file["robots"]["robot"]:
                    # 方案1: 直接从HDF5读取已保存的点云 ⭐
                    print(f"  ✅ 使用HDF5中的点云: {pointcloud_key}")
                    pc_data = np.array(h5_file["robots"]["robot"][pointcloud_key])
                    print(f"    原始点云形状: {pc_data.shape}")
                    
                    # 采样到固定数量
                    pointcloud_data = np.zeros((rgb_data.shape[0], num_points, 6), dtype=np.float32)
                    for i in range(pc_data.shape[0]):
                        pc = pc_data[i]  # (N, 6) [x,y,z,r,g,b] 或 (N, 3) [x,y,z]
                        
                        # 如果只有xyz没有rgb，用零填充颜色
                        if pc.shape[1] == 3:
                            print(f"    ⚠️  点云无颜色，添加零颜色")
                            pc = np.hstack([pc, np.zeros((pc.shape[0], 3), dtype=np.float32)])
                        
                        # 最远点采样
                        if pc.shape[0] > 0:
                            pc_sampled = furthest_point_sampling(pc, num_points)
                            pointcloud_data[i] = pc_sampled
                    
                    print(f"    采样后形状: {pointcloud_data.shape}")
                    
                elif depth_resized is not None:
                    # 方案2: 从深度图生成点云（备用）
                    print(f"  ⚠️  HDF5无点云，从深度图生成: {base_name}_pointcloud...")
                    pointcloud_data = np.zeros((rgb_data.shape[0], num_points, 6), dtype=np.float32)
                    
                    for i in range(depth_resized.shape[0]):
                        depth_map = depth_resized[i]
                        rgb_img = rgb_resized[i]
                        mask_img = mask_resized[i] if mask_resized is not None else None
                        
                        # 生成点云 (带RGB颜色)
                        pc = depth_to_pointcloud(
                            depth_map=depth_map,
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            rgb_image=rgb_img,
                            mask=mask_img,
                            near=0.2, far=1.8
                        )  # (N, 6) [x, y, z, r, g, b]
                        
                        # 最远点采样到固定数量
                        if pc.shape[0] > 0:
                            pc_sampled = furthest_point_sampling(pc, num_points)
                            pointcloud_data[i] = pc_sampled
                    
                    print(f"  点云生成完成: shape={pointcloud_data.shape}")
                else:
                    print(f"  ⚠️ 无法生成点云：既无HDF5点云数据，也无深度图")
                    pointcloud_data = None

            # ================= 存储 Dataset =================
            
            # 1. RGB (主要输出)
            key = f"{base_name}_rgb"
            h5_temp.create_dataset(key, data=rgb_resized)
            episode[key] = h5_temp[key]
                
            # 2. Mask (单独)
            if with_mask and mask_resized is not None:
                key = f"{base_name}_mask"
                h5_temp.create_dataset(key, data=mask_resized)
                episode[key] = h5_temp[key]
                
            # 3. Depth (单独 uint16)
            if with_depth and depth_resized is not None:
                key = f"{base_name}_depth"
                data = depth_resized[..., None]
                h5_temp.create_dataset(key, data=data)
                episode[key] = h5_temp[key]
                
            # 4. Normals (单独 uint8)
            if with_normals and normals_resized is not None:
                key = f"{base_name}_normals"
                h5_temp.create_dataset(key, data=normals_resized)
                episode[key] = h5_temp[key]
            
            # 5. Point Cloud (float32)
            if with_pointcloud and pointcloud_data is not None:
                key = f"{base_name}_pointcloud"
                h5_temp.create_dataset(key, data=pointcloud_data)
                episode[key] = h5_temp[key]
                print(f"  存储点云: {key}, shape={pointcloud_data.shape}")
                
                # 收集点云用于后续组合
                # all_camera_pointclouds[base_name] = pointcloud_data
 


    return episode


def extract_path_structure(h5_dir: str) -> tuple:
    """
    从输入路径中提取目录结构
    
    例如：
    输入：/path/to/data/motion_plan/grasp/100ml玻璃烧杯/20251218_215604
    输出：('motion_plan', 'grasp', '100ml玻璃烧杯')
    
    会自动识别以下关键词作为结构起点：
    - motion_plan
    - 如果没有找到，则使用最后3级目录
    """
    parts = h5_dir.rstrip('/').split('/')
    
    # 查找 motion_plan 的位置
    try:
        mp_idx = parts.index('motion_plan')
        # 返回 motion_plan 之后的结构（不包含时间戳目录）
        # 假设结构为：motion_plan/task_type/object_name/[timestamp]
        structure_parts = parts[mp_idx:]
        # 如果最后一部分看起来像时间戳（纯数字或下划线数字），则排除
        if structure_parts and structure_parts[-1].replace('_', '').isdigit():
            structure_parts = structure_parts[:-1]
        return tuple(structure_parts)
    except ValueError:
        # 没找到 motion_plan，使用最后2-3级目录
        relevant_parts = parts[-3:] if len(parts) >= 3 else parts
        # 排除时间戳目录
        if relevant_parts and relevant_parts[-1].replace('_', '').isdigit():
            relevant_parts = relevant_parts[:-1]
        return tuple(relevant_parts)


if __name__ == "__main__":
    args = parser.parse_args()
    h5_dir = args.h5_dir
    zarr_dir = args.zarr_dir
    mode = args.mode
    with_mask = args.with_mask
    with_depth = args.with_depth
    with_normals = args.with_normals
    save_rgb_separate = args.save_rgb_separate
    save_mask_separate = args.save_mask_separate
    with_pointcloud = args.with_pointcloud
    num_points = args.num_points
    task_type = args.task_type
    max_episodes = args.max_episodes

    # 提取输入路径的结构
    path_structure = extract_path_structure(h5_dir)
    
    # 构建输出路径：zarr_dir/motion_plan/task_type/object_name_timestamp.zarr
    # 例如：zarr_state/motion_plan/grasp/100ml玻璃烧杯_20251220_123045.zarr
    zarr_subdir = os.path.join(zarr_dir, *path_structure[:-1]) if len(path_structure) > 1 else zarr_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zarr_filename = (path_structure[-1] if path_structure else "output") + "_" + timestamp
    zarr_path = os.path.join(zarr_subdir, zarr_filename + ".zarr")
    
    # 确保输出目录存在
    if not os.path.exists(zarr_subdir):
        os.makedirs(zarr_subdir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"HDF5 to Zarr 转换器")
    print(f"=" * 60)
    print(f"模式: {mode}")
    print(f"任务类型: {task_type}")
    print(f"最大转换数量: {max_episodes} 条")
    if mode == "rgb":
        print(f"包含Mask: {'是 (4通道 RGBM)' if with_mask else '否 (3通道 RGB)'}")
        print(f"包含Depth: {'是' if with_depth else '否'}")
        print(f"包含Normals: {'是' if with_normals else '否'}")
        print(f"包含PointCloud: {'是 (每相机 ' + str(num_points) + ' 点)' if with_pointcloud else '否'}")
        print(f"单独存储RGB: {'是' if save_rgb_separate else '否'}")
        print(f"单独存储Mask: {'是' if save_mask_separate else '否'}")
    print(f"输入目录: {h5_dir}")
    print(f"路径结构: {'/'.join(path_structure)}")
    print(f"输出路径: {zarr_path}")
    print(f"=" * 60)

    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path,
        mode="w"
    )

    # 获取所有 HDF5 文件
    h5_file_names = []
    for file in os.listdir(h5_dir):
        if file.split(".")[-1] == "hdf5":
            h5_file_names.append(file)
    
    # 限制处理数量
    total_files = len(h5_file_names)
    h5_file_names = h5_file_names[:max_episodes]
    
    print(f"找到 {total_files} 个 HDF5 文件，将转换 {len(h5_file_names)} 个文件")
    
    h5_temp = h5py.File(os.path.join(h5_dir, "temp"), 'w')
    episode_lengths = []

    for idx, h5_file_name in enumerate(h5_file_names):
        h5_temp.clear()
        print(f"[{idx + 1}/{len(h5_file_names)}] 处理: {h5_file_name}")
        
        try:
            h5_file = h5py.File(os.path.join(h5_dir, h5_file_name), 'r')
        except (OSError, IOError) as e:
            print(f"  ❌ 跳过损坏的文件 {h5_file_name}: {e}")
            print(f"  建议：删除或修复此文件")
            continue

        try:
            # 根据模式选择转换函数
            if mode == "state":
                episode = convert_state_based(h5_file, h5_temp, task_type=task_type)
            else:  # mode == "rgb"
                episode = convert_rgb_based(h5_file, h5_temp, with_mask=with_mask,
                                            with_depth=with_depth, with_normals=with_normals,
                                            save_rgb_separate=save_rgb_separate,
                                            save_mask_separate=save_mask_separate,
                                            with_pointcloud=with_pointcloud,
                                            num_points=num_points,
                                            task_type=task_type)
            
            # 记录长度
            episode_lengths.append(episode['timestamps'].shape[0])
            replay_buffer.add_episode(episode, compressors="disk")
            print(f"  ✅ 成功处理，episode 长度: {episode['timestamps'].shape[0]}")
        except Exception as e:
            print(f"  ❌ 处理文件时出错 {h5_file_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  继续处理下一个文件...")
        finally:
            h5_file.close()

    h5_temp.close()
    os.remove(os.path.join(h5_dir, "temp"))
    
    print(f"=" * 60)
    print(f"转换完成！共处理 {len(h5_file_names)} 个文件")
    if episode_lengths:
        print(f"数据统计:")
        print(f"  - 平均长度: {sum(episode_lengths) / len(episode_lengths):.2f}")
        print(f"  - 最大长度: {max(episode_lengths)}")
        print(f"  - 最短长度: {min(episode_lengths)}")
    print(f"输出路径: {zarr_path}")
    print(f"=" * 60)