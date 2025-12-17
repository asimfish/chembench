# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random.articulated_random_cfg import ArticulatedRandomCfg
from psilab.random.rigid_random_cfg import RigidRandomCfg
from psilab.random.light_random_cfg import LightRandomCfg
from psilab.random.robot_random_cfg import RobotRandomCfg

@configclass
class RandomCfg():
    """Configuration for random options."""

    global_light_cfg: None | LightRandomCfg = None # type: ignore
    """The global light random configuration."""

    local_lights_cfg : None | dict[str, LightRandomCfg] = None # type: ignore
    """The local lights random configuration."""

    robots_cfg : None | dict[str, RobotRandomCfg] = None # type: ignore
    """The robots random configuration."""

    rigid_objects_cfg : None | dict[str, RigidRandomCfg] = None # type: ignore
    """The rigid objects random configuration."""

    articulated_objects_cfg : None | dict[str, ArticulatedRandomCfg] = None # type: ignore
    """The articulated objects random configuration."""

    # deformable_objects_cfg : dict[str, DeformableObjectCfg] = MISSING # type: ignore
    #"""The deformable objects random configuration."""


