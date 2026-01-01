# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class DiffIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for differential inverse kinematics controller of psilab."""

    joint_name:list[str] = MISSING # type: ignore
    """ The name of joints which be controlled by ik controller """

    eef_link_name:str = MISSING # type: ignore
    """ The name of end effector link of ik controller """



