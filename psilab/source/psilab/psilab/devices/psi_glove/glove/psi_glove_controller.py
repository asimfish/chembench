#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

"""
PSI手套控制器主接口

提供高层次的PSI手套控制接口，封装了通信、数据解析和平滑处理。
"""

import logging
import struct
from collections import deque
from typing import Optional, Deque, List

from .communication_interface import CommunicationInterface
from .types import RequestMessage, RequestType, StatusMessage

logger = logging.getLogger(__name__)

# 可选依赖：NumPy用于高效的数值计算
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.info("NumPy not available, using pure Python implementation")


class PSIGloveController:
    """
    PSI手套控制器
    
    核心控制器类，负责：
    - 与PSI手套设备的通信管理
    - Modbus协议数据解析
    - 关节数据的移动平均平滑
    - 状态缓存和查询
    
    数据平滑算法:
    使用滑动窗口移动平均，减少传感器噪声影响，提高数据稳定性。
    如果安装了NumPy，会使用高效的向量化计算。
    
    使用示例:
        >>> # 创建通信接口
        >>> from psi_glove_sdk import SerialInterface, PSIGloveController
        >>> serial = SerialInterface("/dev/ttyUSB0")
        >>>
        >>> # 创建控制器（10个样本的平滑窗口）
        >>> controller = PSIGloveController(serial, smoothing_window_size=5)
        >>>
        >>> # 连接设备
        >>> if not controller.connect():
        ...     print("连接失败")
        ...     exit(1)
        >>>
        >>> # 主循环
        >>> import time
        >>> while controller.is_connected():
        ...     # 读取并更新状态
        ...     status = controller.loop()
        ...
        ...     if status:
        ...         # 处理关节数据
        ...         print(f"拇指指尖: {status.thumb[0]}")
        ...
        ...     time.sleep(0.01)  # 100Hz
    """
    
    def __init__(
        self,
        communication_interface: CommunicationInterface,
        smoothing_window_size: int = 5
    ):
        """
        构造控制器
        
        Args:
            communication_interface: 通信接口实例
            smoothing_window_size: 平滑窗口大小（样本数量，默认5）
            
        Raises:
            TypeError: 如果communication_interface不是CommunicationInterface的实例
            ValueError: 如果smoothing_window_size小于1
            
        Note:
            更大的窗口提供更平滑的数据，但会增加延迟
        """
        # 输入验证
        if not isinstance(communication_interface, CommunicationInterface):
            raise TypeError(
                f"communication_interface必须是CommunicationInterface的实例，"
                f"收到: {type(communication_interface).__name__}"
            )
        
        if not isinstance(smoothing_window_size, int) or smoothing_window_size < 1:
            raise ValueError(
                f"smoothing_window_size必须是大于0的整数，收到: {smoothing_window_size}"
            )
        
        self._interface = communication_interface
        self._smoothing_window_size = smoothing_window_size
        self._last_status: Optional[StatusMessage] = None
        
        # 数据平滑队列：存储历史关节位置数据
        self._position_queue: Deque[List[int]] = deque(
            maxlen=smoothing_window_size
        )
    
    def connect(self) -> bool:
        """
        连接到手套设备
        
        通过底层通信接口建立连接。
        
        Returns:
            bool: 连接成功返回True，失败返回False
            
        Note:
            此方法会清空之前的状态缓存
        """
        success = self._interface.connect()
        if success:
            # 清空历史数据
            self._position_queue.clear()
            self._last_status = None
        return success
    
    def disconnect(self) -> None:
        """
        断开连接
        
        关闭与手套设备的连接，清空状态缓存。
        """
        self._interface.disconnect()
        self._position_queue.clear()
        self._last_status = None
    
    def is_connected(self) -> bool:
        """
        检查连接状态
        
        Returns:
            bool: 已连接返回True，否则返回False
        """
        return self._interface.is_connected()
    
    def read_joint_positions(self) -> Optional[StatusMessage]:
        """
        读取关节位置数据
        
        发送Modbus请求，读取所有21个关节的位置数据。
        数据会经过CRC校验和平滑处理。
        
        Returns:
            Optional[StatusMessage]: 成功时返回StatusMessage，失败返回None
            
        Note:
            此方法会阻塞直到收到响应或超时
        """
        # 构造读取关节位置的Modbus请求
        request_message = RequestMessage(RequestType.READ_JOINT_POSITION)
        
        # 发送请求并接收响应
        response = self._interface.send_and_receive(request_message)
        
        if response is None:
            logger.debug("Failed to receive response")
            return None
        
        # 解析响应数据
        status = self._parse_response(response)
        return status
    
    def loop(self) -> Optional[StatusMessage]:
        """
        主循环处理函数
        
        读取关节数据并更新内部状态缓存。
        这是推荐的获取数据方式，因为它会缓存最后一次成功的读取。
        
        Returns:
            Optional[StatusMessage]: 成功时返回StatusMessage，失败返回None
            
        Note:
            即使读取失败，可以通过get_last_status()获取上次的有效数据
        """
        status = self.read_joint_positions()
        
        # 更新状态缓存（即使读取失败也保留上次的数据）
        if status is not None:
            self._last_status = status
            return status
        else:
            return self._last_status
        
        
    
    def get_last_status(self) -> Optional[StatusMessage]:
        """
        获取上次成功读取的状态
        
        返回最后一次成功读取的关节数据。
        在通信临时中断时，可用于获取最后的有效状态。
        
        Returns:
            Optional[StatusMessage]: 如果存在历史状态返回StatusMessage，
                                    否则返回None
        """
        return self._last_status
    
    def _parse_response(self, raw_bytes: bytes) -> Optional[StatusMessage]:
        """
        解析Modbus响应数据
        
        解析从PSI手套接收的原始字节流，提取21个关节的位置值。
        
        协议格式:
        - 响应头: 01 03 2A (从机地址 + 功能码 + 数据长度)
        - 数据区: 42字节关节数据（21个关节×2字节，大端序）
        - CRC校验: 2字节
        
        关节顺序:
        - 拇指: 5个关节 [tip, mid, back, side, rotate]
        - 食指: 4个关节 [tip, mid, back, side]
        - 中指: 4个关节 [tip, mid, back, side]
        - 无名指: 4个关节 [tip, mid, back, side]
        - 小指: 4个关节 [tip, mid, back, side]
        
        Args:
            raw_bytes: 原始字节数据
            
        Returns:
            Optional[StatusMessage]: 解析成功返回StatusMessage，失败返回None
        """
        # 验证数据长度（至少需要45字节）
        MIN_RESPONSE_SIZE = 45  # 3(头) + 42(数据) + 2(CRC) - 2
        if len(raw_bytes) < MIN_RESPONSE_SIZE:
            logger.error(
                f"Invalid response size: {len(raw_bytes)} "
                f"(expected at least {MIN_RESPONSE_SIZE})"
            )
            return None
        
        # 验证响应头
        if raw_bytes[0] != 0x01 or raw_bytes[1] != 0x03 or raw_bytes[2] != 0x2A:
            logger.debug(
                f"Invalid response header: "
                f"{raw_bytes[0]:02X} {raw_bytes[1]:02X} {raw_bytes[2]:02X}"
            )
            return None
        
        try:
            # 解析21个关节数据（大端序: '>21H'）
            # '>' = 大端序, 'B' = 无符号字节, 'H' = 无符号短整数(2字节)
            # 格式: 3个字节头 + 21个短整数
            header_and_joints = struct.unpack(">BBB21H", raw_bytes[:45])
            
            # 提取关节数据（跳过头部的3个字节）
            joint_positions = list(header_and_joints[3:24])
            
            # 修正特定关节数据（从1开始计数：第1、4、6、10、14、18个）
            # 这些位置的数据需要用4096减去原值
            correction_indices = [0, 3, 5, 9, 13, 17]  # specific joints requiring inversion
            for idx in correction_indices:
                joint_positions[idx] = 4096 - joint_positions[idx]
            
            # 应用数据平滑
            smoothed = self._apply_smoothing(joint_positions)
            
            # 构造状态消息并分配到各个手指
            status_message = StatusMessage(
                thumb=smoothed[0:5],      # 拇指: 0-4
                index=smoothed[5:9],      # 食指: 5-8
                middle=smoothed[9:13],    # 中指: 9-12
                ring=smoothed[13:17],     # 无名指: 13-16
                pinky=smoothed[17:21],    # 小指: 17-20
            )
            
            logger.debug(
                f"Parsed joint positions: "
                f"thumb={status_message.thumb}, "
                f"index={status_message.index}"
            )
            
            return status_message
            
        except struct.error as e:
            logger.error(f"Failed to unpack response data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return None
    
    def _apply_smoothing(self, joints: List[int]) -> List[int]:
        """
        应用移动平均平滑滤波
        
        对关节数据应用滑动窗口平均，减少噪声和抖动。
        
        算法:
        1. 将新数据加入队列
        2. 如果队列已满（达到窗口大小），移除最老的数据
        3. 计算队列中所有数据的算术平均值
        
        Args:
            joints: 原始关节数据（21个关节）
            
        Returns:
            List[int]: 平滑后的关节数据
            
        Note:
            在窗口未满时返回原始数据，避免初期过度平滑
        """
        # 将新数据加入滑动窗口队列
        self._position_queue.append(joints)
        
        # 如果队列未满，返回原始数据（避免初期数据不足导致的误差）
        if len(self._position_queue) < self._smoothing_window_size:
            return joints
        
        # 计算移动平均值
        if HAS_NUMPY:
            # NumPy实现：高效的向量化计算 (numpy 2.0 compatible)
            smoothed = np.mean(self._position_queue, axis=0).astype(np.int64).tolist()
        else:
            # 纯Python实现
            smoothed = []
            for i in range(len(joints)):
                # 累加窗口内所有样本的第i个关节值
                total = sum(positions[i] for positions in self._position_queue)
                # 计算平均值
                avg = total // len(self._position_queue)
                smoothed.append(avg)
        
        return smoothed
    
    def __repr__(self) -> str:
        """字符串表示"""
        status = "connected" if self.is_connected() else "disconnected"
        return (
            f"PSIGloveController("
            f"interface={self._interface.__class__.__name__}, "
            f"status={status}, "
            f"smoothing_window={self._smoothing_window_size})"
        )
