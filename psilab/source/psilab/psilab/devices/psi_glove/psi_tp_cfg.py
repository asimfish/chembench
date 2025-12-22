# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" Common Modules  """ 
import numpy

""" OpenTelevision """
from dex_retargeting.retargeting_config import RetargetingConfig

""" IsaacLab Modules  """ 
from isaaclab.utils import configclass


from psilab.devices.teleop_base import TeleOperateDeviceCfgBase

@configclass
class PsiTpCfg(TeleOperateDeviceCfgBase):
    # 数据设备类型，用于tensor
    device: str = MISSING  # type: ignore
    # 根据tracker serial区分左右手
    tracker_serial_left : str = None  # type: ignore
    tracker_serial_right : str = None  # type: ignore
    # tracker delta position 缩放系数
    delta_pos_scale : list[float] = None  # type: ignore
    # 手套端口
    glove_port_left : str = None # type: ignore
    glove_port_right : str = None # type: ignore
    # 手套端口波特率
    glove_baudrate : int = MISSING # type: ignore
    # 从手套中提取的关节信息索引（展平数据）
    glove_index : list[int] = MISSING # type: ignore
    # 手套返回的ADC值范围，用来归一化
    glove_left_adc_max : list[int] = MISSING # type: ignore
    glove_left_adc_min : list[int] = MISSING # type: ignore
    glove_right_adc_max : list[int] = MISSING # type: ignore
    glove_right_adc_min : list[int] = MISSING # type: ignore

    # Tracker 原始坐标系为右手系，Y轴向上，X轴正方向为左前(45度)，Z轴正方向为右前(45度)，如下：
    # 最新
    # 
    #         Y 
    #     X\  |  / Z
    #       \ | /
    #        \|/
    #
    #       Y |  
    #         | 
    #  Z __ __|
    #        /
    #       /
    #    X /
    # isaaclab中坐标系为右手系，X轴向前，Y轴向左，Z轴向上，如下：
    #       Z |  / X
    #         | /
    #  Y __ __|/
    #      
    grd_yup2grd_zup = numpy.array([[-1, 0, 0.0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
    # grd_yup2grd_zup = numpy.array([[0.7071068, 0, -0.7071068, 0],
    #                                 [-0.7071068, 0, -0.7071068, 0],
    #                                 [0, 1, 0, 0],
    #                                 [0, 0, 0, 1]])
    # # traker实际位置比手掌要高0.11米
    # tracker2wrist_mat : numpy.array = MISSING
    
    # Tracker世界坐标系到仿真世界坐标系的四元数,grd_yup2grd_zup的四元数形式
    tracker2world_quat : numpy.array = MISSING

    # 双臂末端初始位姿,应为手面平行于世界坐标系XY平面,手掌朝向世界坐标系Z轴负方向
    eef_left_quat_init : numpy.array = MISSING
    eef_right_quat_init : numpy.array = MISSING