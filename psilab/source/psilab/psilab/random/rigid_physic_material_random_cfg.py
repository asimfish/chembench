# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-28
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal



""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class RigidPhysicMaterialRandomCfg():
    """Configuration for material random options(Only for MDL Material)."""

    enable : bool = MISSING # type: ignore
    
    random_type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    static_friction_range : list[float,float] =  MISSING # type: ignore
    """The static friction range, which is [0.0,naf].
    - ``static_friction_range[0]``: The minimum value of the static friction.
    - ``static_friction_range[1]``: The maximum value of the static friction.
    """

    static_friction_list : list[float] = MISSING # type: ignore
    """The static friction list."""

    dynamic_friction_range : list[float,float] =  MISSING # type: ignore
    """The dynamic friction range, which is [0.0,naf].
    - ``dynamic_friction_range[0]``: The minimum value of the dynamic friction range.
    - ``dynamic_friction_range[1]``: The maximum value of the dynamic friction range.
    """

    dynamic_friction_list : list[float] = MISSING # type: ignore
    """The dynamic friction list."""

    restitution_range : list[float,float] =  MISSING # type: ignore
    """The restitution range, which is [0.0,1.0].
    - ``restitution_range[0]``: The minimum value of the restitution range.
    - ``restitution_range[1]``: The maximum value of the restitution range.
    """

    restitution_list : list[float] = MISSING # type: ignore
    """The restitution list."""

    material_path : list[str] = None # type: ignore
    """The physics material path of object(relative)."""

    