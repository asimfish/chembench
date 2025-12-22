# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" Common Modules  """ 
import numpy

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class LightRandomCfg():
    """Configuration for light random options."""

    random_type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``True``: parameters are random within the given range.
    - ``False``: parameters are random selected from the given list.
    """

    random_intensity : bool =  MISSING # type: ignore
    """Whether the intensity of light is random."""

    random_color : bool =  MISSING # type: ignore
    """Whether the color of light is random."""

    intensity_range : None | list[int,int] =  None # type: ignore
    """The intensity range which is [lower_limit, upper_limit]."""

    color_range : list[list[numpy.uint8,numpy.uint8,numpy.uint8]] | None =  None # type: ignore
    """The color range.
    - Notes: \n
    [[red_lower_limit, green_lower_limit,blue_lower_limit],[red_upper_limit, green_upper_limit, blue_upper_limit]]
    """

    intensity_list : None | list[int] =  None # type: ignore
    """The intensity list."""

    color_list : None | list[numpy.uint8,numpy.uint8,numpy.uint8] =  None # type: ignore
    """The color list(RGB)."""


    