# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script is used to demonstrate how to randomize and load articulated objects
    in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python articulated_object.py

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
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_TEXTURE_ASSET_DIR
from psilab.scene import Scene,SceneCfg
from psilab.assets import ArticulatedObjectCfg,DomeLightCfg
from psilab.random import (
    RandomCfg,
    ArticulatedRandomCfg,
    VisualMaterialRandomCfg,
    RigidPhysicMaterialRandomCfg,
    PositionRandomCfg,
    OrientationRandomCfg,
    JointRandomCfg,
    MassRandomCfg,
)



# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(10.75,0.0,1.2),
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
        replicate_physics=True,

        
        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        
        # articulated objects
        articulated_objects_cfg={
            "door": ArticulatedObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Door",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/articulated_objects/door/willow_door/WillowDoor.usd",
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state = ArticulatedObjectCfg.InitialStateCfg(
                    pos=(0.15, 0.0, 0.0), 
                    rot= (0.0, 0.0, 0.0, 1.0)
                ),
                actuators={}
            )
        },
        
        # 
        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            articulated_objects_cfg = {
                "door": ArticulatedRandomCfg(
                    mass=MassRandomCfg(
                        enable=True,
                        type="range",
                        mass_range=[0,100],
                        mass_list=[],
                        density_range=None,
                        density_list=None,
                        prim_path=[
                            "/door",
                        ]
                    ),
                    position= PositionRandomCfg(
                        enable=[True,True,True],
                        type="range",
                        offset_range=[0.1,0.1,0.0],
                        offset_list=[
                            [0.1,0.0,0.0],
                            [0.0,0.1,0.0],
                            [-0.1,0.0,0.0],
                            [0.0,-0.1,0.0],
                        ],
                    ),
                    orientation=OrientationRandomCfg(
                        enable=[True,True,True],
                        type="range",
                        eular_base=[
                            [0.0,0.0,numpy.pi],
                        ],
                        height_offset=[
                            0.0,
                            -0.0668
                        ],
                        eular_range=[
                            [0.0,0.0,15 * numpy.pi / 180.0 ]

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
                    visual_material = VisualMaterialRandomCfg(
                        enable=True,
                        shader_path=[
                            "/Looks/material_1/Shader",
                            "/Looks/material_2/Shader",
                            "/Looks/material_3/Shader",
                            "/Looks/material_4/Shader",
                            "/Looks/material_5/Shader",
                        ],
                        random_type="range",
                        material_type = "colored_texture",
                        color_range=[
                            [0,0,0],
                            [255,255,255]
                        ], # type: ignore
                        color_list = [
                            [0,32,54],
                            [231,65,0],
                            [21,123,10],
                        ], # type: ignore
                        roughness_range=[0.0,1.0],
                        roughness_list=[0.0,0.5,1.0],
                        metalness_range=[0.0,1.0],
                        metalness_list=[0.0,0.5,1.0],
                        specular_range=[0.0,1.0],
                        specular_list=[0.0,0.5,1.0],
                        texture_list =[
                            PSILAB_USD_ASSET_DIR + "/articulated_objects/door/willow_door/Textures/T_5ec51feb7d6a630001a94e47_color.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092148.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092142.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092135.jpg",
                        ]
                    ),                                      
                    physics_material= RigidPhysicMaterialRandomCfg(
                        enable=False,
                        random_type="range",
                        static_friction_range=[0.0,1.0],
                        static_friction_list=[],
                        dynamic_friction_range=[0.0,2.0],
                        dynamic_friction_list=[],
                        restitution_range=[0.0,1.0],
                        restitution_list=[]
                    ),
                    joint=JointRandomCfg(
                        enable=True,
                        type="range",
                        joint_names = ["joint_door"],
                        position_range=[[0,1.57]],
                        position_list=[[0,0.1,0.2,0.3]],
                        damping_range=[[0,1000000]],
                        damping_list=[[0,100,10000,1000000]],
                        stiffness_range=[[0,1000000]],
                        stiffness_list=[[0,100,10000,1000000]],
                        friction_range=[[0,10.0]],
                        friction_list=[[0,0.5,1.0,1.5,2.0,2.5,3.0]],
                        armature_range=None,
                        armature_list=None

                    )
                    
                )
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
step_max = 100
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()

    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


