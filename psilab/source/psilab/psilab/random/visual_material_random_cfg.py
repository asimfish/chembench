# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-28
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" Common Modules"""
import numpy

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class VisualMaterialRandomCfg():
    """Configuration for material random options(Only for MDL Material)."""

    enable : bool = MISSING # type: ignore
    
    random_type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    material_type : Literal["color", "texture", "colored_texture"]  = MISSING # type: ignore
    """Configuration for material type.
    - ``color``: the material is show as rgb color.
    - ``texture``: the material is show as texture map.
    - ``colored_texture``: the material is show as texture map with rgb color.
    """

    color_range : list[list[numpy.uint8,numpy.uint8,numpy.uint8]] =  MISSING # type: ignore
    """The RGB range.
    - ``color_range[0]``: The minimum value of the RGB color range.
    - ``color_range[1]``: The maximum value of the RGB color range.
    """

    color_list : list[numpy.uint8,numpy.uint8,numpy.uint8] = MISSING # type: ignore
    """The RGB list."""

    texture_list : list[str] = MISSING # type: ignore
    """The texture list."""

    roughness_range : list[float,float] = MISSING # type: ignore
    """
    The range of roughness which is in range [0.0,1.0].
    The roughness parameter determines roughness of the specular reflections. 
    Higher roughness values lead to a more “powdery” look
    - ``roughness_range[0]``: The minimum value of the roughness range.
    - ``roughness_range[1]``: The maximum value of the roughness range.
    """

    roughness_list : list[float] = MISSING # type: ignore
    """
    The list of roughness which is in range [0.0,1.0].
    The roughness parameter determines roughness of the specular reflections. 
    Higher roughness values lead to a more “powdery” look
    """

    metalness_range : list[float,float] = MISSING # type: ignore
    """
    The range of metalness which is in range [0.0,1.0].
    If metalness parameter is 1.0, reflection will be colored and independent of view direction. If it is 0.0, reflection will be white and direction dependent
    - ``metalness_range[0]``: The minimum value of the metalness range.
    - ``metalness_range[1]``: The maximum value of the metalness range.
    """

    metalness_list : list[float] = MISSING # type: ignore
    """
    The list of metalness which is in range [0.0,1.0].
    If metalness parameter is 1.0, reflection will be colored and independent of view direction. If it is 0.0, reflection will be white and direction dependent
    - ``metalness_range[0]``: The minimum value of the metalness range.
    - ``metalness_range[1]``: The maximum value of the metalness range.
    """

    specular_range : list[float,float] = MISSING # type: ignore
    """
    The range of specular reflectivity which is in range [0.0,1.0].
    - ``specular_range[0]``: The minimum value of the specular range.
    - ``specular_range[1]``: The maximum value of the specular range.
    """

    specular_list : list[float] = MISSING # type: ignore
    """
    The list of specular reflectivity which is in range [0.0,1.0].
    - ``specular_range[0]``: The minimum value of the specular range.
    - ``specular_range[1]``: The maximum value of the specular range.
    """
    
    shader_path : list[str] = MISSING # type: ignore
    """The shader path of object material(relative)."""
