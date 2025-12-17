from typing import Union, Dict, Optional
import zarr
import numpy as np
from functools import cached_property
import numbers
from diffusion_policy.common.replay_buffer import ReplayBuffer

class ZarrImageReference:
    """延迟加载的图像数据引用"""
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
        self.chunks = zarr_array.chunks

    def __getitem__(self, idx):
        return self.zarr_array[idx]

    def __len__(self):
        return len(self.zarr_array)

class StreamingReplayBuffer(ReplayBuffer):
    """
    流式加载的ReplayBuffer，主要用于处理大规模数据
    """
    def __init__(self, root=None):
        self._data = None
        self._meta = None
        self._episode_ends = None
        self.zarr_path = None
        
    @classmethod
    def copy_from_path(cls, path, keys=None):
        """从zarr文件创建StreamingReplayBuffer"""
        buffer = cls()
        buffer.zarr_path = path
        
        # 打开zarr存储
        src = zarr.open(path, mode='r')
        
        # 加载元数据
        buffer._meta = dict()
        if 'meta' in src:
            for key, value in src['meta'].items():
                if len(value.shape) == 0:
                    buffer._meta[key] = np.array(value)
                else:
                    buffer._meta[key] = value[:]
        
        # 设置数据引用
        buffer._data = dict()
        if 'data' not in src:
            src_data = src
        else:
            src_data = src['data']
            
        if keys is None:
            keys = list(src_data.keys())
            
        for key in keys:
            arr = src_data[key]
            # 图像数据使用引用，其他数据完整加载
            if key in ['rgbm', 'right_cam_img']:
                buffer._data[key] = ZarrImageReference(arr)
            else:
                buffer._data[key] = arr[:]
                
        return buffer

    @property
    def data(self):
        """获取数据字典"""
        return self._data

    @property
    def meta(self):
        """获取元数据字典"""
        return self._meta

    @property
    def episode_ends(self):
        """获取episode结束索引"""
        if self._episode_ends is None:
            if 'episode_ends' in self.meta:
                self._episode_ends = self.meta['episode_ends']
            else:
                # 获取第一个数据的长度
                first_data = next(iter(self.data.values()))
                self._episode_ends = np.array([len(first_data)])
        return self._episode_ends

    def get_episode(self, idx, copy=False):
        """获取指定episode的数据"""
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        
        result = dict()
        for key, value in self.data.items():
            x = value[start_idx:end_idx]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """获取指定步骤范围的数据"""
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def __len__(self):
        """获取数据长度"""
        if len(self.episode_ends) == 0:
            return 0
        return int(self.episode_ends[-1])

    def __getitem__(self, key):
        """获取指定key的数据"""
        return self.data[key]

    def keys(self):
        """获取所有数据键"""
        return self.data.keys()

    def values(self):
        """获取所有数据值"""
        return self.data.values()

    def items(self):
        """获取所有数据项"""
        return self.data.items()

    def __contains__(self, key):
        """检查是否包含指定键"""
        return key in self.data

    @property
    def n_episodes(self):
        """获取episode总数"""
        return len(self.episode_ends)

    def get_episode_lengths(self):
        """获取每个episode的长度"""
        ends = self.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        return lengths