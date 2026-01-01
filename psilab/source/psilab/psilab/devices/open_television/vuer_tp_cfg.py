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
class VuerTpCfg(TeleOperateDeviceCfgBase):
    """Configuration parameters for vuer open television """ 
    #
    device: str = MISSING  # type: ignore
    # 
    # 图像分辨率,height,width
    resolution:tuple[int, int] = MISSING    # type: ignore
    # 头部相对于Root的偏移, 大地坐标系，xyz, unit: meter
    head_pos:tuple[float, float, float]= MISSING # type: ignore
    # 左眼相对于头部偏移, 大地坐标系，xyz, unit: meter
    eye_left_offset:tuple[float, float, float]= MISSING # type: ignore
    # 右眼相对于头部偏移, 大地坐标系，xyz, unit: meter
    eye_right_offset:tuple[float, float, float]= MISSING # type: ignore
    # Todo: 手部偏移弥补操作空间大于人双臂实际空间，后续改为缩放功能
    # 左手相对原始位置的偏移，, 大地坐标系，xyz, unit: meter
    hand_left_offset:tuple[float, float, float]= MISSING # type: ignore
    # 右手相对原始位置的偏移, 大地坐标系，xyz, unit: meter
    hand_right_offset:tuple[float, float, float]= MISSING # type: ignore

    # 大拇指、食指、中指、无名指、小拇指指尖在WebXR中的索引
    tip_indices = [4, 9, 14, 19, 24]

    # open television原始齐次矩阵，用作坐标系基变换
    # WebXR原始坐标系为右手系，Y轴向上，X轴向右，Z轴向后，如下：
    #       Y | 
    #         |
    #         |__ __X
    #        /
    #     Z / 
    #
    # isaaclab中坐标系为右手系，X轴向前，Y轴向左，Z轴向上，如下：
    #       Z |  / X
    #         | /
    #  Y __ __|/
    #      
    grd_yup2grd_zup = numpy.array([[0, 0, -1, 0],
                                [-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])

    # 双手腕部从USD模型坐标系到WebXR腕部坐标系的转换
    hand_vr2usd:numpy.array = MISSING

    # 双手手指从USD模型坐标系到WebXR腕部坐标系的转换
    finger_vr2usd:numpy.array = MISSING

    # WebXR中双眼默认姿态为(roll:0,pitch:0,yaw:0)，不便于操作
    # 因此调整默认姿态，便于观察和操作
    head_optimize:numpy.array = MISSING

    # open television 所需证书
    cert_file:str = MISSING    # type: ignore
    key_file:str = MISSING # type: ignore

    # 手部retarget配置文件
    left_hand_retarget_cfg:RetargetingConfig = MISSING # type: ignore
    right_hand_retarget_cfg:RetargetingConfig = MISSING # type: ignore

    # 
    hand_retarget_indexs : list[int] = MISSING  # type: ignore
    hand_scale : list[float] = MISSING # type: ignore


    

    