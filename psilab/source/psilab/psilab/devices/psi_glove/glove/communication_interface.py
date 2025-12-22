#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

"""
PSI手套通信接口抽象基类

定义了与PSI手套设备通信的标准接口规范。
支持多种通信方式（串口、TCP/IP等）的统一抽象。
"""

from abc import ABC, abstractmethod
from typing import Optional

from .types import RequestMessage


class CommunicationInterface(ABC):
    """
    抽象通信接口基类
    
    定义了与PSI手套设备通信的标准接口。采用上下文管理器协议，
    支持with语句自动管理连接生命周期。
    
    使用示例:
        >>> class MyInterface(CommunicationInterface):
        ...     def connect(self) -> bool:
        ...         # 实现连接逻辑
        ...         pass
        ...     # 实现其他抽象方法
        ...
        >>> with MyInterface() as interface:
        ...     response = interface.send_and_receive(message)
    """
    
    def __enter__(self):
        """进入上下文管理器，自动连接"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，自动断开连接"""
        self.disconnect()
        return False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到设备
        
        建立与PSI手套设备的通信连接。具体行为由子类实现
        （如打开串口、建立TCP连接等）。
        
        Returns:
            bool: 连接成功返回True，否则返回False
            
        Note:
            此方法应该是幂等的，重复调用不应产生副作用
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与设备的连接
        
        关闭与PSI手套设备的通信连接，释放相关资源。
        
        Note:
            此方法应该是幂等的，重复调用不应产生副作用
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        Returns:
            bool: 已连接返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def send_and_receive(self, message: RequestMessage) -> Optional[bytes]:
        """
        发送请求并接收响应
        
        发送Modbus请求消息到PSI手套，并等待接收响应数据。
        此方法可能阻塞直到收到响应或超时。
        
        Args:
            message: 要发送的请求消息
            
        Returns:
            Optional[bytes]: 成功时返回响应数据的字节序列，失败返回None
            
        Note:
            线程安全性由子类实现决定
        """
        pass
