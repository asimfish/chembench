# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")

parser.add_argument("--task", type=str, default="", help="Name of the task.")




""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# store args befor create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" Common Modules  """ 
import sys

""" Isaac Lab Modules  """ 
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401

""" Psi RL Modules  """ 
# import psilab.tasks # noqa: F401
import psilab_tasks
from psilab.scene.sence_cfg import SceneCfg
from psilab.envs.rp_env import RPEnv
from psilab.envs.ct_env_cfg import CTEnvCfg
from psilab.envs.ct_env import CTEnv

from psilab.utils.config_utils import scene_cfg
from psilab_tasks.utils import parse_scene_cfg,parse_rp_env_cfg
from psilab.utils.gym_utils import make

# env_cfg = # parse argumanets for isaac lab rl env config
env_cfg= parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
)


# parse argumanets for psi lab scene config
env_cfg.scene = parse_scene_cfg(
    args_cli.task, 
    False,
    None,
    None,
    1,
)


# create env
env : CTEnv = make(args_cli.task, cfg=env_cfg)

# reset env before loop
env.reset()

step = 0
step_max = 100
while(True):
    # sim reset
    if step % step_max == 0:
        env.reset()

    env.step()
    step+=1
