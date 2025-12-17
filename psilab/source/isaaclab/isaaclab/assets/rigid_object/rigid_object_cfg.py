# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .rigid_object import RigidObject
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    class_type: type = RigidObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""

    # Feature: add physics material config for rigid object
    # Author: Feng Yunduo, Data 2025-05-28,Start
    physics_material:RigidBodyMaterialCfg = None # type: ignore
    # Author: Feng Yunduo, Data 2025-05-28,End

    # Feature: add auto compute position offset in Z Axis config for rigid object
    # Author: Feng Yunduo, Data 2025-10-24,Start
    height_offset : float | None = None
    enable_height_offset : bool = False
    # Author: Feng Yunduo, Data 2025-10-24,End