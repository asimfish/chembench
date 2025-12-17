# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from collections.abc import Sequence

import torch
""" IsaacLab Modules  """ 
from  isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.utils.math import apply_delta_pose, compute_pose_error

""" PsiLab Modules  """ 
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg


class DiffIKController(DifferentialIKController):
    """psilab differential inverse kinematics controller."""

    eef_link_index:int = None # type: ignore
    """ The end effector link index of ik controller among all robot link """

    eef_jacobian_index:int = None # type: ignore
    """ The end effector link jacobian index """

    joint_index: list[int] = None # type: ignore
    """ The index of joints which be controlled by ik controller """

    # 关节增量限制参数
    max_delta_joint_pos: float = 0.1  # 每步最大关节变化量 (rad)

    def __init__(self, cfg: DiffIKControllerCfg, num_envs: int, device: str):
        super().__init__(cfg,num_envs,device)
        # overwrite cfg
        self.cfg = cfg

    def initialize_impl(self,robot): # type: ignore

        # 
        self.joint_index = robot.find_joints(self.cfg.joint_name)[0]
        # 
        self.eef_link_index = robot.find_bodies(self.cfg.eef_link_name)[0][0]
        # 
        self.eef_jacobian_index = self.eef_link_index - 1 
        # 
        
        
    def reset(self,robot,env_ids: Sequence[int] | None = None):
        #
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self._device) # type: ignore

            
        super().reset()
        # 
        self.eef_pose_init = robot.data.body_link_state_w[env_ids,self.eef_link_index,:7]
        # print(robot.data.body_link_state_w[0,16,:7])
        # print(robot.data.body_link_state_w[0,17,:7])
        # 
        self.eef_pose_init[:,:3] -= robot.data.root_link_pos_w[env_ids,:3]
        #
        self.set_command(self.eef_pose_init,env_ids) # type: ignore

    def set_command(
        self, command: torch.Tensor, env_ids: Sequence[int] | None = None, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self._device) # type: ignore
                
        # store command
        self._command[env_ids,:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[env_ids,:] = ee_pos + self._command[env_ids,:]
                self.ee_quat_des[env_ids,:] = ee_quat
            else:
                self.ee_pos_des[env_ids,:] = self._command[env_ids,:]
                self.ee_quat_des[env_ids,:] = ee_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.ee_pos_des[env_ids,:], self.ee_quat_des[env_ids,:] = apply_delta_pose(ee_pos, ee_quat, self._command[env_ids,:])
            else:
                self.ee_pos_des[env_ids,:] = self._command[env_ids, 0:3]
                self.ee_quat_des[env_ids,:] = self._command[env_ids, 3:7]

    def compute(
            self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor,env_ids: Sequence[int] | None = None
        ) -> torch.Tensor:
            """Computes the target joint positions that will yield the desired end effector pose.

            Args:
                ee_pos: The current end-effector position in shape (N, 3).
                ee_quat: The current end-effector orientation in shape (N, 4).
                jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
                joint_pos: The current joint positions in shape (N, num_joints).

            Returns:
                The target joint positions commands in shape (N, num_joints).
            """

            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self._device) # type: ignore

            # compute the delta in joint-space
            if "position" in self.cfg.command_type:
                position_error = self.ee_pos_des[env_ids,:] - ee_pos
                jacobian_pos = jacobian[env_ids, 0:3]
                delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
            else:
                position_error, axis_angle_error = compute_pose_error(
                    ee_pos, ee_quat, self.ee_pos_des[env_ids,:], self.ee_quat_des[env_ids,:], rot_error_type="axis_angle"
                )
                pose_error = torch.cat((position_error, axis_angle_error), dim=1)
                delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
            
            # # 正则化：限制关节增量最大值，避免跳跃
            # delta_joint_pos = torch.clamp(delta_joint_pos, -self.max_delta_joint_pos, self.max_delta_joint_pos)
            
            # return the desired joint positions
            return joint_pos + delta_joint_pos