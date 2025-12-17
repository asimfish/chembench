# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any

""" Common Modules  """
import torch
from datetime import datetime

""" Omniverse Modules  """ 
import omni.kit.app

""" IsaacLab Modules  """ 
from isaaclab.envs.common import VecEnvStepReturn

""" PsiLab Modules  """ 
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.envs.rl_env import RLEnv
from psilab.utils.data_collect_utils import create_data_buffer

class ILEnv(RLEnv):
    """The imitation learning basic environment class."""

    def __init__(self, cfg: ILEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)
        #
        self.cfg = cfg
        # fake state
        self._obs_zero = {
            "policy":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device), # type: ignore
            "critic":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device) # type: ignore
        }
        self._reward_zero = torch.zeros(self.num_envs,device=self.device) # type: ignore
        self._reset_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)
        self._dones_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        
        self.episode_length_buf += 1
        #
        # return observations, rewards, resets and extras
        return self._obs_zero, self._reward_zero, self._reset_zero, self.reset_time_outs, dict()

    def sim_step(self, render: bool = True):

        
        self._sim_step_counter+=1
        # robot step to compute ik and ..., to set joint target
        for robot in self.scene.robots.values():
            robot.step()
        
        # set actions into simulator
        self.scene.write_data_to_sim()

        # simulate
        self.sim.step(render=render)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
        #     self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # reset 
        super().reset()


    """
    Functions for RL env which is useless in IL env
    """
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        return self._obs_zero

    def _get_rewards(self) -> torch.Tensor:
       return self._reward_zero

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reset_zero,self._dones_zero

