# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script provides a demonstration of how to configure and randomize a robot and scene 
    using a JSON configuration file in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python scene_json_cfg.py

===============================================================================
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates random demo")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import numpy

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext,SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.envs.common import ViewerCfg

""" Psi Lab Modules  """ 
from psilab import SCRIPT_DIR
from psilab.scene.sence import Scene
from psilab_tasks.utils.parse_cfg import parse_scene_cfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(2,0.0,1.2),
    lookat=(-10.0, 0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=1,
    enable_scene_query_support=True,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
    ),

    render=RenderCfg(),
)

# parse scene confif from json file
scene_cfg = parse_scene_cfg(
    enable_json=True,
    json_file=SCRIPT_DIR +"/demos/scene_json_cfg/room_cfg.json",
    num_envs=1
)

# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(scene_cfg)
#
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
#
step = 0
step_max = 300

# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


