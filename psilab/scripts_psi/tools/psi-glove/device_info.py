# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-13-08
# Vesion: 1.0


"""
Glove SDK Calibration Scripts
"""
""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

import os
from typing import Optional


import math
from psilab.devices.configs.psi_glove_cfg import PSIGLOVE_PSI_DC_02_CFG
from psilab.devices.psi_glove.glove.psi_glove_controller import PSIGloveController, StatusMessage
from psilab.devices.psi_glove.glove.serial_interface import SerialInterface

# glove degrees of freedom
glove_dof = 21 
# min/max value of glove ADC
glove_left_adc_min = [ math.inf for i in range(glove_dof)]
glove_left_adc_max = [ -math.inf for i in range(glove_dof)]
glove_right_adc_min = [ math.inf for i in range(glove_dof)]
glove_right_adc_max = [ -math.inf for i in range(glove_dof)]

# initiallize glove
try:
    # 1. 创建串口接口
    # 参数: 串口路径, 波特率, 超时时间(6ms), 自动连接, 模拟模式
    glove_left_serial = SerialInterface(
        port=PSIGLOVE_PSI_DC_02_CFG.glove_port_left,
        baudrate=PSIGLOVE_PSI_DC_02_CFG.glove_baudrate,
        timeout=0.006,
        auto_connect=False,
        mock=False
    )
    glove_right_serial = SerialInterface(
        port=PSIGLOVE_PSI_DC_02_CFG.glove_port_right,
        baudrate=PSIGLOVE_PSI_DC_02_CFG.glove_baudrate,
        timeout=0.006,
        auto_connect=False,
        mock=False
    )

    # 2. 创建控制器
    # 参数: 通信接口, 平滑窗口大小(10个样本)
    glove_left_controller = PSIGloveController(
        communication_interface=glove_left_serial,
        smoothing_window_size=5
    )
    glove_right_controller = PSIGloveController(
        communication_interface=glove_right_serial,
        smoothing_window_size=5
    )

    # 3. 连接设备
    print("[INFO] Psi Glove is connecting...")
    if not glove_left_controller.connect():
        print("[ERROR] Can not connect Psi Glove Left")
    if not glove_right_controller.connect():
        print("[ERROR] Can not connect Psi Glove Right")
    print("[INFO] Psi Glove connect success...")
    print()
except Exception as e:
    print(f"[ERROR]: {e}")

# update
while True:
    # update glove left adc 
    glove_left_adc: Optional[StatusMessage] = glove_left_controller.loop()
    # 
    if glove_left_adc:
        # convert to list
        glove_left_adc_list : list[int] = glove_left_adc.to_list() # type: ignore
        # update min/max value
        for i in range(glove_dof):
            glove_left_adc_min[i] = min(glove_left_adc_min[i], glove_left_adc_list[i])
            glove_left_adc_max[i] = max(glove_left_adc_max[i], glove_left_adc_list[i])
    # glove right
    glove_right_adc: Optional[StatusMessage] = glove_right_controller.loop()
    #
    if glove_right_adc:
        # convert to list
        glove_right_adc_list : list[int] = glove_right_adc.to_list() # type: ignore
        # update min/max value
        for i in range(glove_dof):
            glove_right_adc_min[i] = min(glove_right_adc_min[i], glove_right_adc_list[i])
            glove_right_adc_max[i] = max(glove_right_adc_max[i], glove_right_adc_list[i])
    # log
    os.system("clear")
    print(f"glove_left_adc_min: {glove_left_adc_min}") # type: ignore
    print(f"glove_left_adc_max: {glove_left_adc_max}") # type: ignore
    print(f"glove_right_adc_min: {glove_right_adc_min}") # type: ignore
    print(f"glove_right_adc_max: {glove_right_adc_max}") # type: ignore
