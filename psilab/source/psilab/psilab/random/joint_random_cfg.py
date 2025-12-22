# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-23
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal
from typing import Optional

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class JointRandomCfg():
    """Configuration for revolute and prismatic joint random options. 
    """

    enable : bool =  MISSING # type: ignore
    """Whether enable position or not. The order is X,Y,Z."""

    type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    joint_names : list[str] = MISSING # type: ignore

    position_range : None | list[list[float]] =  None # type: ignore
    """The joint position range which is [[joint1_position_min,joint1_position_max],[joint2_position_min,joint2_position_max],...], th order is same as joint names.
    """

    position_list : None | list[list[float]] =  None # type: ignore
    """The joint position list."""

    damping_range : None | list[list[float]] =  None # type: ignore
    """The joint damping range which is [[joint1_damping_min,joint1_damping_max],[joint2_damping_min,joint2_damping_max],...], th order is same as joint names.
    """

    damping_list : None | list[list[float]] =  None # type: ignore
    """The joint damping list."""

    stiffness_range : None | list[list[float]] =  None # type: ignore
    """The joint stiffness range which is [[joint1_stiffness_min,joint1_stiffness_max],[joint2_stiffness_min,joint2_stiffness_max],...], th order is same as joint names.
    """

    stiffness_list : None | list[list[float]] =  None # type: ignore
    """The joint stiffness list."""

    friction_range : None | list[list[float]] =  None # type: ignore
    """The joint friction range which is [[joint1_friction_min,joint1_friction_max],[joint2_friction_min,joint2_friction_max],...], th order is same as joint names.
    """

    friction_list : None | list[list[float]] =  None # type: ignore
    """The joint friction list."""

    armature_range : None | list[list[float]] =  None # type: ignore
    """The joint armature range which is [[joint1_armature_min,joint1_armature_max],[joint2_armature_min,joint2_armature_max],...], th order is same as joint names.
    """

    armature_list : None | list[list[float]] =  None # type: ignore
    """The joint armature list."""


    