# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to load a particle cloth object in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python particle_cloth.py

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
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.envs.common import ViewerCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene import Scene,SceneCfg
from psilab.assets import ParticleClothCfg,DomeLightCfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(2.75,0.0,0.3),
    lookat=(-15.0,0.0,0.3)
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

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 10, 
        env_spacing=3.0, 
        replicate_physics=False,

        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # static object
        static_objects_cfg = {
            "ground" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Ground", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/grid/default_environment.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                )
            ),
        },
       
        # particle
        particle_cloths_cfg ={
            "jacket": ParticleClothCfg(
                prim_path="/World/envs/env_[0-9]+/Cloth",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/clothes/jacket/jacket.usd",
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state = AssetBaseCfg().InitialStateCfg(
                    pos=(0.15, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                ),
            )
        },
        
    )


# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(SCENE_CFG)
#
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
#
step = 0
step_max = 500
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    # get all positions of particles in the cloth
    particle_positions = scene.particle_cloths["jacket"].data.positions
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


