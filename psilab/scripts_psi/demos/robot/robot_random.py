# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to randomize a robot in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python robot_random.py

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
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import numpy
import torch

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    SimulationContext,SimulationCfg,PhysxCfg,RenderCfg,
    RigidBodyPropertiesCfg,RigidBodyMaterialCfg)
from isaaclab.envs.common import ViewerCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_TEXTURE_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets.light import DomeLightCfg
from psilab.assets.robot import RobotBaseCfg
from psilab.random import (
    RandomCfg,
    VisualMaterialRandomCfg,
    RigidPhysicMaterialRandomCfg,
    PositionRandomCfg,
    OrientationRandomCfg,
    RobotRandomCfg,
    JointRandomCfg,
    PrimRandomCfg
)

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(3.75,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=2,
    enable_scene_query_support=True,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        gpu_max_rigid_patch_count = 4096 * 4096,
    ),
    render=RenderCfg(),
)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 2, 
        env_spacing=2.5, 
        replicate_physics=True,

        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=100.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02.usd",
                    activate_contact_sensors = False,

                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "arm1_joint_link1": -0.24,
                        "arm1_joint_link2": -0.64,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 1.35,
                        "arm2_joint_link1": 0.36939895,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.85681702,
                        "arm2_joint_link7": 2.33696087,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_1_2":0.63,
                        "hand1_joint_link_1_3":0.03,
                        "hand1_joint_link_2_1":3.10,
                        "hand1_joint_link_2_2":1.56,
                        "hand1_joint_link_3_1":3.06,
                        "hand1_joint_link_3_2":1.56,
                        "hand1_joint_link_4_1":3.06,
                        "hand1_joint_link_4_2":1.56,
                        "hand1_joint_link_5_1":3.04,
                        "hand1_joint_link_5_2":1.56,
                        "hand2_joint_link_1_1":0.0,
                        "hand2_joint_link_1_2":0.64,
                        "hand2_joint_link_1_3":0.03,
                        "hand2_joint_link_2_1":3.11,
                        "hand2_joint_link_2_2":1.56,
                        "hand2_joint_link_3_1":3.06,
                        "hand2_joint_link_3_2":1.56,
                        "hand2_joint_link_4_1":3.08,
                        "hand2_joint_link_4_2":1.56,
                        "hand2_joint_link_5_1":3.05,
                        "hand2_joint_link_5_2":1.56,
                    }
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0
                ),        
                actuators={
                    "arm1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,

                    ),
                    "arm2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,
                    ),
                    "hand1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand1_joint_link_1_1",
                            "hand1_joint_link_2_1",
                            "hand1_joint_link_3_1",
                            "hand1_joint_link_4_1",
                            "hand1_joint_link_5_1",
                            "hand1_joint_link_1_2",
                            "hand1_joint_link_2_2",
                            "hand1_joint_link_3_2",
                            "hand1_joint_link_4_2",
                            "hand1_joint_link_5_2",
                            "hand1_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                    "hand2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand2_joint_link_1_1",
                            "hand2_joint_link_2_1",
                            "hand2_joint_link_3_1",
                            "hand2_joint_link_4_1",
                            "hand2_joint_link_5_1",
                            "hand2_joint_link_1_2",
                            "hand2_joint_link_2_2",
                            "hand2_joint_link_3_2",
                            "hand2_joint_link_4_2",
                            "hand2_joint_link_5_2",
                            "hand2_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                },
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },
                tiled_cameras={
                    "camera_head_color": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera_head_base/camera_head_color",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                    "camera_chest_color": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera_chest_base/camera_chest_color",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                },
            )
        },

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
        
        #
        random = RandomCfg(
            robots_cfg={
                "robot":RobotRandomCfg(

                    position= PositionRandomCfg(
                        enable=[True,True,False],
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
                        enable=[False,False,True],
                        type="range",
                        eular_base=[
                            [0.0,0.0,0.0],
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
                        enable=False,
                        shader_path=[
                            "/Looks/material/Shader",
                        ],
                        random_type="range",
                        material_type = "texture",
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
                        enable=False,
                        type="range",
                        joint_names=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7",
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7",
                            ],
                        position_range=[
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            [-1.57,1.57],
                            ],
                        position_list=[[0,0.1,0.2,0.3]],
                        damping_range=[
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            [0,1000000],
                            ],
                        damping_list=[[0,100,10000,1000000]],
                        stiffness_range=[
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                            [1000000,10000000],
                        ],
                        stiffness_list=[[0,100,10000,1000000]],
                        friction_range=[
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0],
                            [0,3.0]
                        ],
                        friction_list=[[0,0.5,1.0,1.5,2.0,2.5,3.0]],
                        armature_range=None,
                        armature_list=None
                    ),
                    prim={
                        "camera_chest_base":PrimRandomCfg(
                            position=PositionRandomCfg(
                                enable=[True,True,True],
                                type="range",
                                offset_range=[0.0,0.01,0.01],
                                offset_list=[
                                    [0,0.5,0],
                                    [0.5,0,0]
                                ]
                            ),
                            orientation=OrientationRandomCfg(
                                enable=[True,True,True],
                                type="range",
                                eular_base=[[0.05,0.05,0.05]],
                                eular_range=[[0.87222222,0.87222222,0.87222222]],
                                eular_list=[[[0.0,0.5,0.0],[0.5,0.0,0.0]]],
                                height_offset=[0.0]
                            ),
                            position_initial=[0.079, -0.0005, 1.124],
                            orientation_initial=[1.0,0.0,0.0,0.0]
                        ),
                        "camera_head_base":PrimRandomCfg(
                            position=PositionRandomCfg(
                                enable=[True,True,True],
                                type="range",
                                offset_range=[0.0,0.01,0.01],
                                offset_list=[
                                    [0,0.5,0],
                                    [0.5,0,0]
                                ]
                            ),
                            orientation=OrientationRandomCfg(
                                enable=[True,True,True],
                                type="range",
                                eular_base=[[0.05,0.05,0.05]],
                                eular_range=[[0.87222222,0.87222222,0.87222222]],
                                eular_list=[[[0.0,0.5,0.0],[0.5,0.0,0.0]]],
                                height_offset=[0.0]
                            ),
                            position_initial=[0.19, 0.003, 1.651],
                            orientation_initial=[0.86603,0.0,0.5,0.0]
                        )
                    }
                )
            }
        )
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
step_max = 300
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()

        # random joint positon target
        joint_pos_target = 2.0 * (torch.rand((scene.cfg.num_envs,scene.robots["robot"].num_joints),device=scene.device) - 0.5 )

        # set joint position
        scene.robots["robot"].set_joint_position_target(joint_pos_target.clone())
        scene.robots["robot"].write_data_to_sim()
        
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1