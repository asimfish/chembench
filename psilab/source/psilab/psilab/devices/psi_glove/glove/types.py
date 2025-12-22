#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

"""
PSI手套SDK数据类型定义

定义了与PSI手套通信和数据处理相关的所有数据结构和枚举类型。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List


class JointType(IntEnum):
    """
    关节类型枚举
    
    定义手指关节的类型，用于索引关节数据。
    """
    TIP = 0      # 指尖关节
    MID = 1      # 中间关节
    BACK = 2     # 根部关节
    SIDE = 3     # 侧向关节（拇指特有）
    ROTATE = 4   # 旋转关节（拇指特有）


class FingerType(IntEnum):
    """
    手指类型枚举
    
    定义五根手指的类型标识。
    """
    THUMB = 0    # 拇指
    INDEX = 1    # 食指
    MIDDLE = 2   # 中指
    RING = 3     # 无名指
    PINKY = 4    # 小指


class RequestType:
    """
    Modbus协议请求类型
    
    预定义的Modbus请求消息，用于与PSI手套通信。
    使用Modbus RTU协议，包含CRC校验。
    """
    # 读取关节位置请求
    # 格式: [从机地址, 功能码, 起始地址高, 起始地址低, 数量高, 数量低, CRC低, CRC高]
    READ_JOINT_POSITION = bytes([0x01, 0x03, 0x00, 0x01, 0x00, 0x15, 0xD5, 0xC5])
    
    # 读取手套ID请求
    READ_GLOVE_ID = bytes([0x01, 0x03, 0x00, 0x07, 0x00, 0x15, 0x35, 0xC4])


@dataclass
class RequestMessage:
    """
    请求消息结构
    
    封装发送到PSI手套的请求数据。
    
    Attributes:
        data: 请求数据字节序列
    """
    data: bytes
    
    def __init__(self, request_type: bytes):
        """
        从预定义请求类型构造请求消息
        
        Args:
            request_type: 预定义的请求类型（来自RequestType）
        """
        self.data = request_type


@dataclass
class StatusMessage:
    """
    状态消息结构
    
    存储从PSI手套读取的原始关节位置数据。
    每个关节值范围: 0-4095 (12位ADC)
    
    关节数据布局:
    - 拇指: 5个关节 [tip, mid, back, side, rotate]
    - 其他手指: 4个关节 [tip, mid, back, side]
    
    Attributes:
        thumb: 拇指5个关节位置
        index: 食指4个关节位置
        middle: 中指4个关节位置
        ring: 无名指4个关节位置
        pinky: 小指4个关节位置
    """
    thumb: List[int] = field(default_factory=lambda: [0] * 5)
    index: List[int] = field(default_factory=lambda: [0] * 4)
    middle: List[int] = field(default_factory=lambda: [0] * 4)
    ring: List[int] = field(default_factory=lambda: [0] * 4)
    pinky: List[int] = field(default_factory=lambda: [0] * 4)
    
    def to_dict(self) -> dict:
        """
        将状态消息转换为字典
        
        Returns:
            包含所有关节数据的字典
        """
        return {
            "thumb": self.thumb,
            "index": self.index,
            "middle": self.middle,
            "ring": self.ring,
            "pinky": self.pinky,
        }
    
    def to_list(self) -> List[int]:
        """
        将所有关节数据展平为列表
        
        Returns:
            包含21个关节值的列表
        """
        return self.thumb + self.index + self.middle + self.ring + self.pinky


@dataclass
class CalibrationParams:
    """
    校准参数结构
    
    存储单个关节的校准范围，用于将原始ADC值映射到标准化范围。
    
    Attributes:
        min: 最小值（完全伸展）
        max: 最大值（完全弯曲）
    """
    min: int = 0
    max: int = 4095  # 12位ADC最大值


@dataclass
class FingerCalibration:
    """
    单根手指的校准配置
    
    存储手指主要关节的校准参数。
    
    Attributes:
        back: 根部关节校准参数
        side: 侧向关节校准参数
        rotate: 旋转关节校准参数（仅拇指）
    """
    back: CalibrationParams = field(default_factory=CalibrationParams)
    side: CalibrationParams = field(default_factory=CalibrationParams)
    rotate: CalibrationParams = field(default_factory=CalibrationParams)


@dataclass
class GloveCalibration:
    """
    整只手套的校准配置
    
    存储一只手套所有手指的校准参数。
    
    Attributes:
        thumb: 拇指校准
        index: 食指校准
        middle: 中指校准
        ring: 无名指校准
        pinky: 小指校准
    """
    thumb: FingerCalibration = field(default_factory=FingerCalibration)
    index: FingerCalibration = field(default_factory=FingerCalibration)
    middle: FingerCalibration = field(default_factory=FingerCalibration)
    ring: FingerCalibration = field(default_factory=FingerCalibration)
    pinky: FingerCalibration = field(default_factory=FingerCalibration)


@dataclass
class Config:
    """
    SDK配置结构
    
    存储左右手套的校准配置。
    
    Attributes:
        left_glove: 左手手套配置
        right_glove: 右手手套配置
    """
    left_glove: GloveCalibration = field(default_factory=GloveCalibration)
    right_glove: GloveCalibration = field(default_factory=GloveCalibration)
