# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-10
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal
from typing import Optional

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 

@configclass
class OrientationRandomCfg():
    """Configuration for orientation random options(Eular angle). 

    Notes: the orientation in this config is relative position from initial position of objects.
    
    """

    enable : list[bool] =  MISSING # type: ignore
    """Whether enable orientation or not. The order is Roll, Pitch, Yaw."""

    type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """
    eular_base : list[list[float]] | None =  None # type: ignore

    eular_range : list[list[float]] | None =  None # type: ignore
    """The position offset range which is [offset_x_max,offset_y_max,offset_z_max].
    """

    eular_list : list[list[list[float]]] | None=  None # type: ignore
    """The position offset list which is [offset_x,offset_y,offset_z],...]."""

    # _quaternion: Optional[list[list[float]]] =  None

    height_offset : list[float] | None = None

    