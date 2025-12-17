# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

""" Common Modules  """ 
import numpy
import torch

""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import RigidBodyPropertiesCfg,MassPropertiesCfg,CollisionPropertiesCfg
from isaaclab.sim.spawners import PreviewSurfaceCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_TEXTURE_ASSET_DIR
from psilab.scene import SceneCfg
from psilab.assets.robot import RobotBaseCfg
from psilab.assets.light.light_cfg import DomeLightCfg
from psilab.random import (
    RandomCfg,
    RigidRandomCfg,
    VisualMaterialRandomCfg,
    RigidPhysicMaterialRandomCfg,
    PositionRandomCfg,
    OrientationRandomCfg,
    MassRandomCfg,
    JointRandomCfg,
    ArticulatedRandomCfg
)
from psilab.assets import ArticulatedObjectCfg

BASE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=2.0, 
        replicate_physics=True,
        
        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # local light
        local_lights_cfg={},

        # robot
        robots_cfg = {},
        
        # static object
        static_objects_cfg = {},
        
        # rigid objects
        rigid_objects_cfg ={},
        
        # articulated objects
        articulated_objects_cfg={
           "door": ArticulatedObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Door",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/articulated_objects/door/willow_door/WillowDoor.usd",
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state = ArticulatedObjectCfg.InitialStateCfg(
                    pos=(0.55, -0.7, 0.9), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                ),
                actuators={}
            ) 
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={},
        
        tiled_cameras_cfg = {},

        # contact sensor
        contact_sensors_cfg={},

        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
                "thumb": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "index": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "middle": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "ring": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "pinky": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "lego": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.04, 0.04, 0.04),
                ),
                "middle_point": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),


            },

        ),

        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = None,
            articulated_objects_cfg = {
                "door": ArticulatedRandomCfg(
                    mass=MassRandomCfg(
                        enable=False,
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
                        enable=[False,False,False],
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
                        enable=[False,False,False],
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
                        position_range=[[0,0]],
                        position_list=[[0]],
                        damping_range=[[0,0]],
                        damping_list=[[0]],
                        stiffness_range=[[0,0]],
                        stiffness_list=[[0]],
                        friction_range=[[0,0.1]],
                        friction_list=[[0,0.03]],
                        armature_range=None,
                        armature_list=None

                    )
                    
                )
            },
            #


        )

    )

PSI_DC_01_CFG = BASE_CFG.replace(
    robots_cfg = {
        "robot" : RobotBaseCfg(
            prim_path = "/World/envs/env_[0-9]+/Robot",
            spawn = sim_utils.UsdFileCfg(
                usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_01/PsiRobot_DC_01.usd",
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
            diff_ik_controllers = {},
            eef_links={
                "arm1":"arm1_link7",
                "arm2":"arm2_link7"
            },
            cameras = {},
            tiled_cameras={
                "base_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/base_camera_rgb/base_camera_rgb",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
                "arm2_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/arm2_camera_rgb/arm2_camera_rgb",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
            },
        )
    }
)

PSI_DC_02_CFG = BASE_CFG.replace(
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
                        # "camera_chest_base_joint": 0.0,
                        # "camera_head_base_joint": 0.0,
                        "arm1_joint_link1": -0.24,
                        "arm1_joint_link2": -0.64,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 0.94,
                        "arm2_joint_link1": 0.14,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.55681702,
                        "arm2_joint_link7": 1.871788888888889,
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
                    # "camera": ImplicitActuatorCfg(
                    #     joint_names_expr=[
                    #         "camera_head_base_joint",
                    #         "camera_chest_base_joint"],
                    #     stiffness=None,
                    #     damping=None,

                    # ),
                },
                diff_ik_controllers = {},
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },
                cameras = {},
                tiled_cameras={
                    # "base_camera": TiledCameraCfg(
                    #     prim_path="/World/envs/env_[0-9]+/Robot/base_camera_rgb/base_camera_rgb",
                    #     data_types=["rgb"],
                    #     width=640,
                    #     height=480,
                    #     spawn=None,
                    # ),
                    # "arm2_camera": TiledCameraCfg(
                    #     prim_path="/World/envs/env_[0-9]+/Robot/arm2_camera_rgb/arm2_camera_rgb",
                    #     data_types=["rgb"],
                    #     width=640,
                    #     height=480,
                    #     spawn=None,
                    # ),
                },
            )
    }
)
