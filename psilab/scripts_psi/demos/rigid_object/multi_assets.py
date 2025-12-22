# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to spawn multiple assets in multiple environments in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python multi_assets.py

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
from isaaclab.sim import SimulationContext,SimulationCfg,PhysxCfg,RenderCfg,RigidBodyPropertiesCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.common import ViewerCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets import DomeLightCfg
from psilab.random import (
    RandomCfg,RigidRandomCfg,PositionRandomCfg,OrientationRandomCfg)


# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(3.75,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=1,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        enable_ccd=True,
        gpu_found_lost_pairs_capacity = 137401003
    ),
    render=RenderCfg(),
)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 20, 
        env_spacing=1.5, 
        replicate_physics=False,

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

            "target" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target",
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path={
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A4/A4.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A5/A5.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A10/A10.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A18/A18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A38/A38.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/drink/B31/B31.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/condiment/C21/C21.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/kitchen_supplies/E18/E18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/clean_household_necessities/F16/F16.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                    },
                    random_choice=False,
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        solver_position_iteration_count=255,
                    ),
                ),
                
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0,-0.105,0.9),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),

            "obstacle1" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Obstacle1",
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path={
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A4/A4.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A5/A5.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A10/A10.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A18/A18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A38/A38.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/drink/B31/B31.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/condiment/C21/C21.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/kitchen_supplies/E18/E18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/clean_household_necessities/F16/F16.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                    },
                    random_choice=False,
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        solver_position_iteration_count=255,
                    ),
                ),
                
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0,-0.105,0.78),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),

            "obstacle2" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Obstacle2",
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path={
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A4/A4.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A5/A5.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A10/A10.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A18/A18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/packaged_foods/A38/A38.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/drink/B31/B31.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/condiment/C21/C21.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/kitchen_supplies/E18/E18.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/clean_household_necessities/F16/F16.usd":[[1.3*i for i in range(1,3)],[0.7*i for i in range(1,3)],[1.0*i for i in range(1,3)]],
                    },
                    random_choice=False,
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        solver_position_iteration_count=255,
                    ),
                ),
                
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0,-0.105,0.78),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
        },

        # 
        random = RandomCfg(
            rigid_objects_cfg = {
            "target": RigidRandomCfg(
                position= PositionRandomCfg(
                        enable=[True,True,True],
                        type="range",
                        offset_range=[0.12,0.155,0.0],
                        offset_list=[
                            [0.1,0.0,0.0],
                            [0.0,0.1,0.0],
                            [-0.1,0.0,0.0],
                            [0.0,-0.1,0.0],
                        ],
                    ),
                orientation=OrientationRandomCfg(
                    enable=[False,False,False],
                    type="range",
                    eular_base=[
                        [0.0,0.0,0.0],
                    ],
                    height_offset=[
                        0.0,
                    ],
                    eular_range=[
                        [0.0,0.0,numpy.pi],

                    ],
                    eular_list=[
                        [
                            [0.0,0.0,0.5 * numpy.pi],
                            [0.0,0.0,-0.5 * numpy.pi],
                            [0.0,0.0,1.5 * numpy.pi],
                            [0.0,0.0,-1.5 * numpy.pi],
                        ], 
                    ],
                ),
                visual_material = None,                                      
                physics_material= None

            ),
            "obstacle1": RigidRandomCfg(
                position= PositionRandomCfg(
                        enable=[True,True,True],
                        type="range",
                        offset_range=[0.12,0.155,0.0],
                        offset_list=[
                            [0.1,0.0,0.0],
                            [0.0,0.1,0.0],
                            [-0.1,0.0,0.0],
                            [0.0,-0.1,0.0],
                        ],
                    ),
                orientation=OrientationRandomCfg(
                    enable=[False,False,False],
                    type="range",
                    eular_base=[
                        [0.0,0.0,0.0],
                    ],
                    height_offset=[
                        0.0,
                    ],
                    eular_range=[
                        [0.0,0.0,numpy.pi],

                    ],
                    eular_list=[
                        [
                            [0.0,0.0,0.5 * numpy.pi],
                            [0.0,0.0,-0.5 * numpy.pi],
                            [0.0,0.0,1.5 * numpy.pi],
                            [0.0,0.0,-1.5 * numpy.pi],
                        ], 
                    ],
                ),
                visual_material = None,                                      
                physics_material= None
            ),
            "obstacle2": RigidRandomCfg(
                position= PositionRandomCfg(
                        enable=[True,True,True],
                        type="range",
                        offset_range=[0.12,0.155,0.0],
                        offset_list=[
                            [0.1,0.0,0.0],
                            [0.0,0.1,0.0],
                            [-0.1,0.0,0.0],
                            [0.0,-0.1,0.0],
                        ],
                    ),
                orientation=OrientationRandomCfg(
                    enable=[False,False,False],
                    type="range",
                    eular_base=[
                        [0.0,0.0,0.0],
                    ],
                    height_offset=[
                        0.0,
                    ],
                    eular_range=[
                        [0.0,0.0,numpy.pi],

                    ],
                    eular_list=[
                        [
                            [0.0,0.0,0.5 * numpy.pi],
                            [0.0,0.0,-0.5 * numpy.pi],
                            [0.0,0.0,1.5 * numpy.pi],
                            [0.0,0.0,-1.5 * numpy.pi],
                        ], 
                    ],
                ),
                visual_material = None,                                      
                physics_material= None
            ),
            },
            #
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
step_max = 500
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


