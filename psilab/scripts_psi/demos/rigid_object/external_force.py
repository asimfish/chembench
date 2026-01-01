# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to apply external force and torque to a rigid object in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python external_force.py

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
import torch

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    SimulationContext,SimulationCfg,PhysxCfg,RenderCfg,
    RigidBodyPropertiesCfg,MassPropertiesCfg,RigidBodyMaterialCfg)
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg

""" Psi Lab Modules  """ 
from psilab.scene.sence import Scene,SceneCfg
from psilab import PSILAB_USD_ASSET_DIR
from psilab.assets.light.light_cfg import DomeLightCfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(6.75,0.0,1.2),
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
        env_spacing=1.5, 
        replicate_physics=True,
  
        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # rigid objects
        rigid_objects_cfg ={      
            "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/drink/B36/B36.usd",
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    ),
                    mass_props=MassPropertiesCfg(
                        mass=0.5
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.2,0.0,0.85),
                    rot= (1,0, 0.0, 0.0)
                    # rot= (0.707, 0.707, 0.0, 0.0)
                ),
                physics_material= RigidBodyMaterialCfg(
                    static_friction=1.4,
                    dynamic_friction=1.4,
                    restitution=0.0
                )
            ),
            
            "table" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Table", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.15, 0.0, 0.0), 
                    rot= (0.707, 0.0, 0.0, 0.707)
                )
            ),  
        },

        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
                "force": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/arrow_x.usd",
                    scale=(0.05,0.05,0.3),
                )
            },
        ),    
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
step_max = 1000
#
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
        step = 0
        # print("reset")

    if step == 500: 
        # print("push")
        forces = 5 * torch.rand((scene.num_envs,scene.rigid_objects["bottle"].num_bodies,3))
        # forces = torch.tensor([0.0,0.0,10.0]).unsqueeze(0).unsqueeze(0).repeat(scene.num_envs,scene.rigid_objects["bottle"].num_bodies,1)
        scene.rigid_objects["bottle"].set_external_force_and_torque(
            forces = forces,
            torques=torch.zeros((scene.num_envs,scene.rigid_objects["bottle"].num_bodies,3)),
        )

        # normlize forces
        forces = forces /  torch.norm(forces,p=2,dim=2).unsqueeze(1)
        # arrow initial unit vector
        unit_vector = torch.tensor([1.0,0.0,0.0]).unsqueeze(0).unsqueeze(0).repeat(scene.num_envs,scene.rigid_objects["bottle"].num_bodies,1)
        # compute quaternion
        cross = torch.cross(unit_vector,forces , dim=-1).squeeze(1)
        dot = torch.sum(unit_vector * forces, dim=-1)
        qw = torch.sqrt((1 + dot) / 2)
        qxyz = cross / (2 * qw + 1e-10)
        quaternion = torch.cat([qw, qxyz], dim=1)
        # refresh marker
        scene.visualizer.visualize(scene.rigid_objects["bottle"].data.root_link_pos_w, quaternion)
    #

    scene.rigid_objects["bottle"].write_data_to_sim()
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


