# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-28
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" Common Modules  """ 
import numpy

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random import PositionRandomCfg,OrientationRandomCfg
from psilab.random.rigid_physic_material_random_cfg import RigidPhysicMaterialRandomCfg
from psilab.random.visual_material_random_cfg import VisualMaterialRandomCfg
from psilab.random.joint_random_cfg import JointRandomCfg
from psilab.random.prim_random_cfg import PrimRandomCfg
@configclass
class RobotRandomCfg():
    """Configuration for mass random options of a rigid body.
    """
    
    position: PositionRandomCfg | None = None

    orientation: OrientationRandomCfg | None = None
    
    visual_material : VisualMaterialRandomCfg | None  =  None # type: ignore
    """The material random config."""

    physics_material : RigidPhysicMaterialRandomCfg | None =  None # type: ignore
    """The physics material random config."""
    
    joint : JointRandomCfg | None =  None # type: ignore
    """The joint random config."""

    prim : dict[str,PrimRandomCfg]| None =  None # type: ignore

