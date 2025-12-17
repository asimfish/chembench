# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-22
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

""" Common Modules  """ 
import torch
import re


import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics,PhysxSchema


""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.assets.asset_base import AssetBase

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
    from .particle_cloth_cfg import ParticleClothCfg

from psilab.assets.particle_cloth.particle_cloth_data import ParticleClothData

class ParticleCloth(AssetBase):
    """An articulatied object class.
    """
    cfg: ParticleClothCfg


    def __init__(self, cfg: ParticleClothCfg):
        """Initialize the Robot.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    def _initialize_impl(self):

        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString
        # find particle cloth root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxAutoParticleClothAPI)
        )

        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a particle cloth when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'PhysxSchema PhysxAutoParticleClothAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single particle cloth when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one particle cloth in the prim path tree."
            )


        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid objects. These are"
                    f" located at: '{articulation_prims}' under '{template_prim_path}'. Please disable the articulation"
                    " root in the USD or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_particle_cloth_view(
            root_prim_path_expr.replace(".*", "*")
        )

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create particle cloth at: {self.cfg.prim_path}. Please check PhysX logs.")
        
        # container for data access
        self._data = ParticleClothData(self._root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)
        pass

    def reset(self, env_ids: Sequence[int] | None = None):
        """
            Reset robot, include joint, actuator, ik controllers
        """
        pass
        self.write_positions_to_sim(self._data.default_positions,env_ids)
        # reset articulation, which will reset actuator
        # super().reset()

        # if env_ids is None:
        #     env_ids = self._ALL_INDICES # type: ignore

        # # reset all joint state and target to default state
        # self.set_joint_position_target(self.data.default_joint_pos.clone()[env_ids,:],env_ids=env_ids)
        # self.write_joint_state_to_sim(
        #     position=self.data.default_joint_pos.clone()[env_ids,:],
        #     velocity=torch.zeros((len(env_ids),self.num_joints),device=self.device), # type: ignore
        #     env_ids=env_ids
        # )
        # self.write_data_to_sim()   

    @property
    def data(self) -> Any:
        """Data related to the asset."""
        return self._data

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)
    
    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # # external forces and torques
        # self.has_external_wrench = False
        # self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        # self._external_torque_b = torch.zeros_like(self._external_force_b)

        # # set information about rigid body into data
        # self._data.body_names = self.body_names
        # self._data.default_mass = self.root_physx_view.get_masses().clone()
        # self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # pass
        
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        # default_root_state = (
        #     tuple(self.cfg.init_state.pos)
        #     + tuple(self.cfg.init_state.rot)
        # )
        # default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_positions = self._data.positions
    
    def write_positions_to_sim(self,positions:torch.tensor,env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self._ALL_INDICES # type: ignore
        # set into simulation
        self.root_physx_view.set_positions(positions, indices=env_ids)


    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def root_physx_view(self) -> physx.ParticleClothView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view