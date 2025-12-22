#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

"""
PSI Glove Python SDK

提供PSI手套设备的Python驱动接口，支持串口通信、数据解析和平滑处理。

主要类:
    - PSIGloveController: 主控制器类
    - SerialInterface: 串口通信接口
    - StatusMessage: 状态数据结构
    - CommunicationInterface: 通信接口抽象基类

基本使用:
    >>> from psi_glove_sdk import PSIGloveController, SerialInterface
    >>> serial = SerialInterface("/dev/ttyUSB0", 115200)
    >>> controller = PSIGloveController(serial)
    >>> controller.connect()
    >>> status = controller.loop()
"""

__version__ = "1.0.0"
__author__ = "PSI Robot Team"
__license__ = "Apache 2.0"

# 导入核心类
from .types import (
    JointType,
    FingerType,
    RequestType,
    RequestMessage,
    StatusMessage,
    CalibrationParams,
    FingerCalibration,
    GloveCalibration,
    Config,
)

from .communication_interface import CommunicationInterface

from .serial_interface import SerialInterface

from .psi_glove_controller import PSIGloveController

# 公开的API
__all__ = [
    # 核心类
    "PSIGloveController",
    "SerialInterface",
    "CommunicationInterface",
    
    # 数据类型
    "StatusMessage",
    "RequestMessage",
    "Config",
    
    # 枚举
    "JointType",
    "FingerType",
    "RequestType",
    
    # 配置类
    "CalibrationParams",
    "FingerCalibration",
    "GloveCalibration",
    
    # 版本信息
    "__version__",
]
