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


@configclass
class MassRandomCfg():
    """Configuration for mass random options of prim.
    
    Tips:
        SceneCfg::replicate_physics should be False while mass random is True for rigidbody objects, otherwise will cause error.
    """
    
    enable : bool =  MISSING # type: ignore
    """Whether enable mass random or not. """

    type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for mass random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    mass_range : None | list[float] =  None # type: ignore
    """The mass of the prim (in kg) range which is [mass_min,mass_max].
    """

    mass_list : None | list[float] =  None
    """The mass of the prim (in kg) list."""

    density_range : None | list[float] =  None # type: ignore
    """The density of the prim (in kg/m^3) range which is [density_min,density_max].
    """

    density_list : None | list[float] =  None
    """The density of the prim (in kg/m^3) list."""

    prim_path : None | list[str] =  None
    """The prim path list(relative)."""