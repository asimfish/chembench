# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
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
from psilab.controllers.differential_ik import DiffIKController

if TYPE_CHECKING:
    from .robot_base_cfg import RobotBaseCfg

class RobotBase(Articulation):
    """An robot asset class.
    An robot is an articulation with some ik controllers which can be easily controled by device and policy input 
    """
    cfg: RobotBaseCfg

    num_envs : int = 1
    cameras : dict[str,Camera] = None # type: ignore
    tiled_cameras : dict[str,TiledCamera] = None # type: ignore
    ik_controllers : dict[str,DiffIKController] = None # type: ignore
    eef_links : dict[str,int] = None # type: ignore
    
    # 速度控制相关
    vel_target : dict[str,torch.Tensor] = None  # type: ignore
    enable_velocity_control : bool = False  # 是否启用速度控制
    sim_dt : float = 1.0 / 40.0  # 仿真时间步长

    def __init__(self, cfg: RobotBaseCfg):
        """Initialize the Robot.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        self.cameras = {}
        self.tiled_cameras = {}
        self.ik_controllers = {}
        self.eef_links = {}
        self.pos_target = {}
        self.vel_target = {}  # 速度目标
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

    def _initialize_impl(self):

        super()._initialize_impl()
        #
        if self.cfg.diff_ik_controllers is not None:
            for ik_name,ik_cfg in self.cfg.diff_ik_controllers.items():
                #
                self.ik_controllers[ik_name] = DiffIKController(ik_cfg, num_envs=self.data.root_com_state_w.shape[0], device=self.device)
                #
                self.ik_controllers[ik_name].initialize_impl(self)
                # get default joint position
                joint_index = self.find_joints(ik_cfg.joint_name,preserve_order=True)[0]
                self.pos_target[ik_name] = self.data.default_joint_pos[:,joint_index]
                pass
            #
            for eef_name,eef_link_name in self.cfg.eef_links.items():
                eef_index = self.find_bodies(eef_link_name)[0][0]
                self.eef_links[eef_name] = eef_index

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
        # print(self.data.joint_pos_target[0,:])
        #
        if self.cfg.diff_ik_controllers:
            for ik_name,ik_cfg in self.cfg.diff_ik_controllers.items():
                self.ik_controllers[ik_name].reset(self,env_ids)
                # get default joint position
                joint_index = self.find_joints(ik_cfg.joint_name,preserve_order=True)[0]
                self.pos_target[ik_name] = self.data.default_joint_pos[:,joint_index]
            # self.ik_step()

        # self.data.root_link_pos_w
 
    def step(self):
        """
        Step for each simulation step, include all computation and set varibale values to data
        
        支持两种控制模式：
        - enable_velocity_control = False: 仅位置控制
        - enable_velocity_control = True: 位置 + 速度控制（更平滑）
        """
        # only update joint position target according to ik result
        if self.ik_controllers:
            for name, controller in self.ik_controllers.items():
                # 设置位置目标
                self.set_joint_position_target(self.pos_target[name], controller.joint_index)
                
                # 如果启用速度控制，同时设置速度目标
                if self.enable_velocity_control:
                    # 计算速度目标：(目标位置 - 当前位置) / dt
                    current_pos = self.data.joint_pos[:, controller.joint_index]
                    self.vel_target[name] = (self.pos_target[name] - current_pos) / self.sim_dt
                    self.set_joint_velocity_target(self.vel_target[name], controller.joint_index)



    def ik_step(self):
        # get root stae 
        root_pose_w = self.data.root_state_w[:, 0:7]
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        # traverse ik controllers
        if self.ik_controllers:
            for name,controller in self.ik_controllers.items():
                jacobian = self.root_physx_view.get_jacobians()[:,controller.eef_jacobian_index, :, controller.joint_index]
                #
                ee_pose_w = self.data.body_link_state_w[:, controller.eef_link_index, :7]
                #
                jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
                jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
                #
                joint_pos = self.data.joint_pos[:, controller.joint_index]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                # compute the joint commands
                self.pos_target[name] = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    
    def set_ik_command(self,command: dict[str,torch.Tensor]):
        """
            Set command for all ik controllers
        """
        ik_command_keys = list[str](command.keys())
        for ik_name in self.cfg.diff_ik_controllers.keys():
            if ik_name in ik_command_keys:
                self.ik_controllers[ik_name].set_command(command[ik_name])

