# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any
import builtins

""" Common Modules  """ 
import torch
from datetime import datetime
""" Omniverse Modules  """ 
import omni.kit.app
import omni.log
""" IsaacSim Modules  """ 
import isaacsim.core.utils.torch as torch_utils


""" IsaacLab Modules  """ 
from isaaclab.sim import SimulationContext
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.timer import Timer
from isaaclab.envs.ui import ViewportCameraController

""" PsiLab Modules  """ 
from psilab.scene import Scene
from psilab.envs.ct_env_cfg import CTEnvCfg
from psilab.envs.rl_env import RLEnv
from psilab.utils.data_collect_utils import create_data_buffer



class CTEnv():
    """The custom environment class."""

    def __init__(self, cfg: CTEnvCfg, render_mode: str | None = None, **kwargs):

        #
        self.cfg = cfg
        
        # store the render mode
        self.render_mode = render_mode
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):

            self.scene : Scene = Scene(self.cfg.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer) # type: ignore
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False: # type: ignore
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.cfg.sim.dt)
        #
        self.scene.reset()
        
    def step(self):
        
        # robot step to compute ik and ..., to set joint target
        for robot in self.scene.robots.values():
            robot.step()
        
        # set actions into simulator
        self.scene.write_data_to_sim()


        self.sim.step(render=True)

        self.scene.update(self.scene.physics_dt)


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        
        self.scene.reset()
        # reset scene
        # self.scene.reset()
        # # 
        # self._vuer.reset()
       
    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)
