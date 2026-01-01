# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-09-11
# Vesion: 1.0


""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random import PositionRandomCfg,OrientationRandomCfg

@configclass
class PrimRandomCfg():
    """Configuration for rigid random options."""

    position: PositionRandomCfg | None = None

    orientation: OrientationRandomCfg | None = None

    position_initial: list[float] = [0.0,0.0,0.0]
    
    orientation_initial: list[float] = [1.0,0.0,0.0,0.0] # [w,x,y,z]

    