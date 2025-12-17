# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" IsaacLab Modules  """ 
from isaaclab.utils import configclass

""" PsiLab Modules  """ 
from psilab.envs.rl_env_cfg import RLEnvCfg
# from psilab.devices.vuer_tp_cfg import VuerTpCfg
from psilab.devices.teleop_base import TeleOperateDeviceCfgBase

@configclass
class TPEnvCfg(RLEnvCfg):
    """Configuration for an tele operation environment.
    """

    device_type : Literal["vuer", "psi-glove"]  = MISSING # type: ignore

    device_cfg : TeleOperateDeviceCfgBase = MISSING # type: ignore
    # device_cfg : VuerTpCfg = MISSING  
    # """ Vuer device config. """

