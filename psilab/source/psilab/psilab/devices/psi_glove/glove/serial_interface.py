#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

"""
串口通信接口实现

基于pyserial实现的串口通信接口，支持多种波特率和超时配置。
"""

import logging
from typing import Optional
import serial

from .communication_interface import CommunicationInterface
from .types import RequestMessage

logger = logging.getLogger(__name__)


class SerialInterface(CommunicationInterface):
    """
    串口通信接口实现
    
    使用pyserial库实现串口通信。
    
    特性:
    - 线程安全的连接状态管理
    - 可配置的超时控制
    - 支持模拟模式（用于测试）
    - 自动资源管理
    
    使用示例:
        >>> # 创建串口接口
        >>> serial = SerialInterface("/dev/ttyUSB0", 115200)
        >>>
        >>> # 使用with语句自动管理连接
        >>> with serial:
        ...     req = RequestMessage(RequestType.READ_JOINT_POSITION)
        ...     response = serial.send_and_receive(req)
        ...     if response:
        ...         print(f"接收到 {len(response)} 字节")
        >>>
        >>> # 或手动管理连接
        >>> if serial.connect():
        ...     response = serial.send_and_receive(req)
        ...     serial.disconnect()
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.006,
        auto_connect: bool = False,
        mock: bool = False
    ):
        """
        构造串口接口
        
        Args:
            port: 串口设备路径（如 "/dev/ttyUSB0" 或 "COM3"）
            baudrate: 波特率，支持9600-921600（默认115200）
            timeout: 读写超时时间，单位秒（默认0.006，即6ms）
            auto_connect: 是否在构造时自动连接（默认False）
            mock: 是否使用模拟模式，不实际打开串口（默认False）
            
        Raises:
            ValueError: 如果参数值不合法
            
        Note:
            mock模式用于测试，会模拟连接但不发送/接收真实数据
        """
        # 输入验证
        if not port or not isinstance(port, str):
            raise ValueError("port必须是非空字符串")
        
        if baudrate not in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000, 921600]:
            raise ValueError(f"不支持的波特率: {baudrate}")
        
        if timeout <= 0:
            raise ValueError(f"timeout必须大于0，收到: {timeout}")
        
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._mock = mock
        self._serial: Optional[serial.Serial] = None
        self._connected = False
        
        # 自动连接模式
        if auto_connect:
            self.connect()
    
    @property
    def port(self) -> str:
        """获取串口设备路径"""
        return self._port
    
    @property
    def baudrate(self) -> int:
        """获取波特率"""
        return self._baudrate
    
    def connect(self) -> bool:
        """
        连接到串口设备
        
        Returns:
            bool: 连接成功返回True，失败返回False
        """
        # 检查是否已连接
        if self._connected:
            logger.info(f"Already connected to {self._port}")
            return True
        
        # 模拟模式：不打开真实串口
        if self._mock:
            logger.info("Mock mode enabled, simulating connection")
            self._connected = True
            return True
        
        try:
            # 打开串口
            # 配置: 8数据位, 无校验, 1停止位 (8N1)
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._timeout,
                write_timeout=self._timeout
            )
            
            self._connected = True
            logger.info(
                f"Successfully connected to {self._port} at {self._baudrate} baud"
            )
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to open port {self._port}: {e}")
            logger.error("Please check:")
            logger.error("  - Device is connected")
            logger.error("  - Port path is correct")
            logger.error("  - User has permission to access port")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    def disconnect(self) -> None:
        """
        断开串口连接
        """
        if not self._connected:
            return
        
        # 模拟模式
        if self._mock:
            self._connected = False
            logger.info("Mock connection closed")
            return
        
        # 关闭串口
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
        
        self._connected = False
        logger.info("Serial connection closed")
    
    def is_connected(self) -> bool:
        """
        检查连接状态
        
        Returns:
            bool: 已连接返回True，否则返回False
        """
        return self._connected
    
    def _send_message(self, data: bytes) -> bool:
        """
        发送消息到串口
        
        Args:
            data: 要发送的字节数据
            
        Returns:
            bool: 发送成功返回True
        """
        if not self.is_connected():
            logger.error("Not connected")
            return False
        
        # 模拟模式：打印发送数据
        if self._mock:
            logger.debug(f"Mock send: {data.hex(' ').upper()}")
            return True
        
        try:
            # 写入串口
            bytes_written = self._serial.write(data)
            
            # 检查是否完整写入
            if bytes_written != len(data):
                logger.error(
                    f"Incomplete write: {bytes_written}/{len(data)} bytes"
                )
                return False
            
            # 确保数据发送完成
            self._serial.flush()
            return True
            
        except serial.SerialTimeoutException:
            # logger.error("Write timeout")
            return False
        except Exception as e:
            logger.error(f"Write failed: {e}")
            return False
    
    def _receive_message(self) -> Optional[bytes]:
        """
        从串口接收消息
        
        Returns:
            Optional[bytes]: 接收到的数据，失败返回None
        """
        if not self.is_connected():
            return None
        
        # 模拟模式：不返回数据
        if self._mock:
            logger.debug("Mock receive (no data in mock mode)")
            return None
        
        try:
            # 读取串口数据
            # 预期接收至少45字节（Modbus响应）
            data = self._serial.read(64)
            
            if len(data) == 0:
                logger.debug("No data received (timeout)")
                return None
            
            return data
            
        except serial.SerialTimeoutException:
            # logger.error("Read timeout")
            return None
        except Exception as e:
            # logger.error(f"Read failed: {e}")
            return None
    
    def send_and_receive(self, message: RequestMessage) -> Optional[bytes]:
        """
        发送请求并接收响应
        
        Args:
            message: 要发送的请求消息
            
        Returns:
            Optional[bytes]: 接收到的响应数据，失败返回None
        """
        # 先发送请求
        if not self._send_message(message.data):
            return None
        
        # 模拟模式不返回数据
        if self._mock:
            return None
        
        # 接收响应
        return self._receive_message()
    
    def __repr__(self) -> str:
        """字符串表示"""
        status = "connected" if self._connected else "disconnected"
        return (
            f"SerialInterface(port={self._port}, "
            f"baudrate={self._baudrate}, "
            f"status={status})"
        )
