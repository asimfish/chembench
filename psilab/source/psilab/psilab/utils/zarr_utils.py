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

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0 # type: ignore

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


def convert_state_based(h5_file, h5_temp) -> dict:
    """
    纯状态模式转换
    
    包含数据：
    - timestamps: 仿真时间
    - action: 手臂位置目标(7) + 手指位置目标(6) = 13维
    - state: 物体位姿(7) + 手臂位置(7) + 手指位置(6) = 20维
    """
    episode = dict()
    
    episode['timestamps'] = h5_file["sim_time"]
    
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
                       save_rgb_separate: bool = False, save_mask_separate: bool = False) -> dict:
    """
    RGB图像 + 状态模式转换
    
    包含数据：
    - timestamps: 仿真时间
    - action: 手臂位置目标(7) + 手指位置目标(6) = 13维
    - arm2_pos: 手臂关节位置(7)
    - arm2_vel: 手臂关节速度(7)
    - hand2_pos: 手指关节位置(6)
    - hand2_vel: 手指关节速度(6)
    - arm2_eef_pos: 末端执行器位置(3)
    - arm2_eef_quat: 末端执行器四元数(4)
    - target_pose: 目标物体位姿(7)
    - head_camera_rgb: 头部相机RGB图像 (N, 224, 224, 3)
    - chest_camera_rgb: 胸部相机RGB图像 (N, 224, 224, 3)
    - third_camera_rgb: 第三相机RGB图像 (N, 224, 224, 3)
    - (可选) head_camera_mask, chest_camera_mask, third_camera_mask (N, 224, 224) uint8
    - (可选) head_camera_depth, chest_camera_depth, third_camera_depth (N, 224, 224, 1) uint16
    - (可选) head_camera_normals, chest_camera_normals, third_camera_normals (N, 224, 224, 3) uint8
    
    Args:
        h5_file: HDF5文件对象
        h5_temp: 临时HDF5文件对象
        image_size: 输出图像尺寸
        with_mask: 是否将mask通道拼接到RGB图像后（变成4通道RGBM）
        with_depth: 是否存储深度图
        with_normals: 是否存储法线图
        save_rgb_separate: 是否单独存储RGB图像（用于检测）
        save_mask_separate: 是否单独存储mask图像（用于检测）
    """
    episode = dict()
    
    episode['timestamps'] = h5_file["sim_time"]
    
    # Action: arm2_pos_target(7) + hand2_pos_target(6) = 13维
    h5_temp.create_dataset(
        "action",
        shape=(h5_file["robots"]["robot"]["arm2_pos_target"].shape[0], 13),
        dtype=h5_file["robots"]["robot"]["arm2_pos_target"].dtype
    )
    h5_temp["action"][:, :7] = h5_file["robots"]["robot"]["arm2_pos_target"]
    h5_temp["action"][:, 7:] = h5_file["robots"]["robot"]["hand2_pos_target"][:, :6]
    
    episode['action'] = h5_temp["action"]
    episode['arm2_pos'] = h5_file["robots"]["robot"]["arm2_pos"]
    episode['arm2_vel'] = h5_file["robots"]["robot"]["arm2_vel"]
    episode['hand2_pos'] = h5_file["robots"]["robot"]["hand2_pos"][:, :6]
    episode['hand2_vel'] = h5_file["robots"]["robot"]["hand2_vel"][:, :6]
    episode['arm2_eef_pos'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, :3]
    episode['arm2_eef_quat'] = h5_file["robots"]["robot"]["arm2_eef_pose"][:, 3:]
    episode['target_pose'] = h5_file["rigid_objects"]["bottle"][:, :7]
    
    # 处理相机图像（包括第三个相机）
    camera_names = ["head_camera.rgb", "chest_camera.rgb", "third_camera.rgb"]
    mask_names = ["head_camera.instance_segmentation_fast", "chest_camera.instance_segmentation_fast", "third_camera.instance_segmentation_fast"]
    
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
            # depth_raw_resized = None # float32 - 移除
            # normals_raw_resized = None # float32 - 移除
            
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
    if mode == "rgb":
        print(f"包含Mask: {'是 (4通道 RGBM)' if with_mask else '否 (3通道 RGB)'}")
        print(f"包含Depth: {'是' if with_depth else '否'}")
        print(f"包含Normals: {'是' if with_normals else '否'}")
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
    
    print(f"找到 {len(h5_file_names)} 个 HDF5 文件")
    
    h5_temp = h5py.File(os.path.join(h5_dir, "temp"), 'w')
    episode_lengths = []

    for idx, h5_file_name in enumerate(h5_file_names):
        h5_temp.clear()
        print(f"[{idx + 1}/{len(h5_file_names)}] 处理: {h5_file_name}")
        h5_file = h5py.File(os.path.join(h5_dir, h5_file_name), 'r')

        # 根据模式选择转换函数
        if mode == "state":
            episode = convert_state_based(h5_file, h5_temp)
        else:  # mode == "rgb"
            episode = convert_rgb_based(h5_file, h5_temp, with_mask=with_mask,
                                        with_depth=with_depth, with_normals=with_normals,
                                        save_rgb_separate=save_rgb_separate,
                                        save_mask_separate=save_mask_separate)
        
        # 记录长度
        episode_lengths.append(episode['timestamps'].shape[0])
        replay_buffer.add_episode(episode, compressors="disk")
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