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


@configclass
class PositionRandomCfg():
    """Configuration for position random options. 

    Notes: the position in this config is relative position from initial position of objects.
    
    """

    enable : list[bool] =  MISSING # type: ignore
    """Whether enable position or not. The order is X,Y,Z."""

    type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    offset_range : None | list[float,float,float] =  None # type: ignore
    """The position offset range which is [offset_x_max,offset_y_max,offset_z_max].
    """

    offset_list : Optional[list[list[float]]] =  None
    """The position offset list which is [offset_x,offset_y,offset_z],...]."""

    