# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

from __future__ import annotations
import torch

from typing import TYPE_CHECKING


from isaaclab.assets.articulation import Articulation

# from omni.isaac.lab.utils.math import (  # isort:skip
#     compute_pose_error,
#     matrix_from_quat,
#     quat_inv,
#     random_yaw_orientation,
#     subtract_frame_transforms,
#     quat_apply,
#     quat_mul
# )
# from collections.abc import Sequence
# from omni.isaac.lab.sensors.camera import CameraCfg,Camera

# from omni.isaac.lab.controllers.joint_impedance import JointImpedanceController
# from omni.isaac.lab.controllers.differential_ik import DifferentialIKController


if TYPE_CHECKING:
    from .manipulation_cfg import ManipulationCfg



class Manipulation(Articulation):

    """An robot asset class.

    An robot is an articulation with some ik controllers which can be easily controled by device and policy input 

    """
    cfg: ManipulationCfg

    
    ik_controllers: dict[str,DifferentialIKController] = None # type: ignore

    # eef_link_idx,eef_jacobi_idx,arm_joint_idxs
    ik_data: dict[str,dict] = None # type: ignore
    # joint target position from ik controller
    ik_joint_pos_target: dict[str,torch.Tensor] = None # type: ignore

    # eef_joint_idxs
    eff_data:dict[str,dict] = None # type: ignore
    # joint target position for effector, eg: inspired hand
    eff_joint_pos_target: dict[str,torch.Tensor] = None # type: ignore

    cameras: dict[str,Camera] = None # type: ignore

    real_joint_index: dict[str:list[int]] = None # type: ignore
    virtual_joint_index: dict[str:list[int]] = None # type: ignore
    real_joint_index_eef: dict[str:list[int]] = None # type: ignore
    virtual_joint_index_eef: dict[str:list[int]] = None # type: ignore

    def __init__(self, cfg: RobotCfg):
        """Initialize the Robot.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        
        self.cameras = dict()

    def reset(self, env_ids: Sequence[int] | None = None):
        """
            Reset robot and ik controller state
        """
        # reset articulation
        super().reset()
        # reset all joint state
        joint_pos = self.data.default_joint_pos.clone()
        joint_vel = self.data.default_joint_vel.clone()
        self.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset ik controller
        for ik_name in self.cfg.ik_cfg.keys():
            self.ik_controllers[ik_name].reset()
            self.ik_controllers[ik_name].set_command(
                torch.tensor(self.cfg.init_state.eef_state[ik_name],device="cuda:0"))

    def step(self):
        """
        Step for each simulation step, include all computation and set varibale values to data
        """
        if self.cfg.control_type=="ik":
            # ik控制器计算
            self.ik_compute()
            # 遍历所有ik控制器
            for ik_name in self.cfg.ik_cfg.keys():
                self.set_joint_position_target(self.ik_joint_pos_target[ik_name],self.ik_data[ik_name]["arm_joint_idxs"])
            # 遍历所有末端执行机构
            for eff_name in self.cfg.eff_name:
                self.set_joint_position_target(self.eff_joint_pos_target[eff_name],self.eff_data[eff_name]["eef_joint_idxs"])

    def set_ik_command(self, ik_command: dict[str,torch.Tensor]):
        """
            Set command for all ik controllers
        """
        ik_command_key = list(ik_command.keys())
        for ik_name in self.cfg.ik_cfg.keys():
            if ik_name in ik_command_key:
                self.ik_controllers[ik_name].set_command(ik_command[ik_name])

    def ik_compute(self):
        """
            ik controllers compute
        """
        self.ik_joint_pos_target = {}
        # get root stae 
        root_pose_w = self.data.root_link_state_w[:, 0:7]
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        # 遍历所有ik控制器
        for ik_name in self.cfg.ik_cfg.keys():
            jacobian = self.root_physx_view.get_jacobians()[:, self.ik_data[ik_name]["eef_jacobi_idx"], :, self.ik_data[ik_name]["arm_joint_idxs"]]
            ee_pose_w = self.data.body_link_state_w[:, self.ik_data[ik_name]["eef_link_idx"], 0:7]
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            joint_pos = self.data.joint_pos[:, self.ik_data[ik_name]["arm_joint_idxs"]]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            # compute the joint commands
            self.ik_joint_pos_target[ik_name] = self.ik_controllers[ik_name].compute(ee_pos_b, ee_quat_b,jacobian, joint_pos)

    def set_eff_command(self,eff_command: dict[str,torch.Tensor]):
        """
            Set command for all effector
        """
        self.eff_joint_pos_target = {}
        effector_command_key = eff_command
        for eff_name in self.cfg.eff_name:
            if eff_name in effector_command_key:
                self.eff_joint_pos_target[eff_name] = eff_command[eff_name]
                # 根据JointMap手动修改虚拟关节
                if eff_name in self.cfg.real_virtual_joint_map.keys():
                    # limit
                    # aa = self.real_joint_index[eff_name]
                    # bb = self.data.joint_limits[0][aa]
                    lower_limit_real_joint = self.data.joint_limits[0,self.real_joint_index[eff_name],0]
                    upper_limit_real_joint = self.data.joint_limits[0,self.real_joint_index[eff_name],1]
                    lower_limit_virtual_joint = self.data.joint_limits[0,self.virtual_joint_index[eff_name],0]
                    upper_limit_virtual_joint = self.data.joint_limits[0,self.virtual_joint_index[eff_name],1]
                    # 
                    target_norm = norm(self.eff_joint_pos_target[eff_name][self.real_joint_index_eef[eff_name]],lower_limit_real_joint,upper_limit_real_joint)
                    self.eff_joint_pos_target[eff_name][self.virtual_joint_index_eef[eff_name]] = scale(target_norm,lower_limit_virtual_joint,upper_limit_virtual_joint)
                    # (self.eff_joint_pos_target[eff_name][self.real_joint_index_eef[eff_name]] -lower_limit_real_joint) / (upper_limit_real_joint - lower_limit_real_joint)
                    # self.eff_joint_pos_target[eff_name][self.virtual_joint_index_eef[eff_name]] = target_norm * (upper_limit_virtual_joint - lower_limit_virtual_joint) + lower_limit_virtual_joint

    
    def _initialize_impl(self):

        super()._initialize_impl()

        # initiallize robot ik data, must be here as physics view only created after impl initiallize
        self.ik_data = {}
        self.ik_controllers={}
        for ik_name in list(self.cfg.ik_cfg.keys()):
            # date used to compute
            self.ik_data[ik_name]={}
            # Obtain the frame index of the end-effector，获取末端执行器索引
            self.ik_data[ik_name]["eef_link_idx"] = self.find_bodies(self.cfg.ik_eef_name[ik_name])[0][0]
            # 索引，根据此索引获取机械臂对应的雅可比矩阵
            self.ik_data[ik_name]["eef_jacobi_idx"] = self.ik_data[ik_name]["eef_link_idx"]  - 1
            # Obtain joint indices，获取机械臂的所有关节索引
            self.ik_data[ik_name]["arm_joint_idxs"]  = self.find_joints(self.cfg.ik_joint_name[ik_name])[0]
            # create IK Controller
            self.ik_controllers[ik_name] = DifferentialIKController(self.cfg.ik_cfg[ik_name], num_envs=1, device=self.device)
            # 设置IK初始状态
            self.ik_controllers[ik_name].reset()
            self.ik_controllers[ik_name].set_command(
                torch.tensor(self.cfg.init_state.eef_state[ik_name],device=self.device))

        # initiallize robot effector data,
        self.eff_data = {}
        for eff_name in self.cfg.eff_name:
            self.eff_data[eff_name] = {}
            # Obtain joint indices，获取执行机构的所有关节索引
            self.eff_data[eff_name]["eef_joint_idxs"]  = self.find_joints(self.cfg.eff_joint_name[eff_name])[0]

        # initiallize joint index map
        self.real_joint_index={}
        self.virtual_joint_index={}
        self.real_joint_index_eef={}
        self.virtual_joint_index_eef={}
        for eff_name in self.cfg.eff_name:
            if eff_name in self.cfg.real_virtual_joint_map.keys():
                # aa = list(self.cfg.real_virtual_joint_map[eff_name].keys())
                # bb = self.find_joints(aa) 
                self.real_joint_index[eff_name] = self.find_joints(self.cfg.real_virtual_joint_map[eff_name].keys())[0]# type: ignore
                self.virtual_joint_index[eff_name] = self.find_joints(self.cfg.real_virtual_joint_map[eff_name].values())[0] # type: ignore
                self.real_joint_index_eef[eff_name] = [self.eff_data[eff_name]["eef_joint_idxs"].index(idx) for idx in self.real_joint_index[eff_name]]
                self.virtual_joint_index_eef[eff_name] = [self.eff_data[eff_name]["eef_joint_idxs"].index(idx) for idx in self.virtual_joint_index[eff_name]]

        pass
                

                
# 将数据根据上下限制归一化至 [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

# 将数据根据上下限制进行反归一化，由[-1,1]->[lower,upper]
@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

