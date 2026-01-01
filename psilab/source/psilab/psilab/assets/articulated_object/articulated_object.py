# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-22
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence

""" Common Modules  """ 
import torch
import re

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.sensors.camera import CameraCfg,Camera,TiledCamera
from isaaclab.utils.math import ( 
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)
from isaaclab.sim.utils import bind_physics_material

""" PsiLab Modules  """ 

if TYPE_CHECKING:
    from .articulated_object_cfg import ArticulatedObjectCfg

class ArticulatedObject(Articulation):
    """An articulatied object class.
    """
    cfg: ArticulatedObjectCfg


    def __init__(self, cfg: ArticulatedObjectCfg):
        """Initialize the Robot.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    def _initialize_impl(self):

        super()._initialize_impl()
        # add physics material 
        if self.cfg.physics_material is not None:
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", self.cfg.prim_path) is None
            # multi envs
            if asset_path_is_regex:
                prim_paths = sim_utils.find_matching_prim_paths(self.cfg.prim_path)
                # 
                for prim_path in prim_paths:
                    physics_material_path = f"{prim_path}/physics_material"
                    self.cfg.physics_material.func(physics_material_path, self.cfg.physics_material)
                    # bind the physics material to the scene
                    bind_physics_material(prim_path, physics_material_path)
            # single env
            else:
                physics_material_path = f"{self.cfg.prim_path}/physics_material"
                self.cfg.physics_material.func(physics_material_path, self.cfg.physics_material)
                # bind the physics material to the scene
                bind_physics_material(self.cfg.prim_path, physics_material_path)

    def reset(self, env_ids: Sequence[int] | None = None):
        """
            Reset robot, include joint, actuator, ik controllers
        """
        # reset articulation, which will reset actuator
        super().reset()

        if env_ids is None:
            env_ids = self._ALL_INDICES # type: ignore

        # reset all joint state and target to default state
        self.set_joint_position_target(self.data.default_joint_pos.clone()[env_ids,:],env_ids=env_ids)
        self.write_joint_state_to_sim(
            position=self.data.default_joint_pos.clone()[env_ids,:],
            velocity=torch.zeros((len(env_ids),self.num_joints),device=self.device), # type: ignore
            env_ids=env_ids
        )
        self.write_data_to_sim()   
