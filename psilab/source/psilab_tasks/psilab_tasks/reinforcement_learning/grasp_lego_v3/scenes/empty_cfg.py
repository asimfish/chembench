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
from isaaclab.sim.schemas import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.sim.spawners import PreviewSurfaceCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
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
    MassRandomCfg
)

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

        # static objects ： ground
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

        # rigid objects
        rigid_objects_cfg ={

            "table" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Table", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/table/CubeTable.usd",
                    scale=(1.0, 1.0, 0.78),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.1,0.1,0.1)
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.65, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                )
            ),

            "target" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Lego",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/lego/lego_1x2.usd",
                    activate_contact_sensors = True,
                    scale=(1.0,1.0,1.0),
                    visual_material=None,
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5,-0.105,0.8),
                    rot= (1,0,0,0)
                ),
                enable_height_offset=True
            ),
        },
        
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
            rigid_objects_cfg = {
                "target": RigidRandomCfg(
                    mass=MassRandomCfg(
                        enable=False,
                        type="range",
                        mass_range=[0,1],
                        mass_list=[],
                        density_range=None,
                        density_list=None,
                    ),
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
                        enable=[False,False,True],
                        type="range",
                        eular_base=[
                            [0.0,0.0,0.0],
                            [0.0, 0.5 * numpy.pi, 0.0],
                            [0.0, 1.0 * numpy.pi, 0.0],
                            [0.5 * numpy.pi, 0.0, 0.0],

                        ],
                        eular_range=[
                            [0.0,0.0,numpy.pi],
                            [0.0,0.0,numpy.pi],
                            [0.0,0.0,numpy.pi],
                            [0.0,0.0,numpy.pi],

                        ],
                        eular_list=[
                            [
                                [0.0,0.0,0.5 * numpy.pi],
                                [0.0,0.0,-0.5 * numpy.pi],
                                [0.0,0.0,1.5 * numpy.pi],
                                [0.0,0.0,-1.5 * numpy.pi],
                            ],
                            [
                                [0.0,0.0,0.5 * numpy.pi],
                                [0.0,0.0,-0.5 * numpy.pi],
                                [0.0,0.0,1.5 * numpy.pi],
                                [0.0,0.0,-1.5 * numpy.pi],
                            ],
                            [
                                [0.0,0.0,0.5 * numpy.pi],
                                [0.0,0.0,-0.5 * numpy.pi],
                                [0.0,0.0,1.5 * numpy.pi],
                                [0.0,0.0,-1.5 * numpy.pi],
                            ],
                            [
                                [0.0,0.0,0.5 * numpy.pi],
                                [0.0,0.0,-0.5 * numpy.pi],
                                [0.0,0.0,1.5 * numpy.pi],
                                [0.0,0.0,-1.5 * numpy.pi],
                            ]
                            
                        ],
                    
                    ),
                    visual_material = VisualMaterialRandomCfg(
                        enable=True,
                        shader_path=["/Looks/material/Shader"],
                        random_type="range",
                        material_type = "color",
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
                        texture_list =[]
                    ),
                    physics_material= RigidPhysicMaterialRandomCfg(
                        enable=True,
                        random_type="range",
                        static_friction_range=[0.4,0.6],
                        static_friction_list=[],
                        dynamic_friction_range=[0.4,0.6],
                        dynamic_friction_list=[],
                        restitution_range=[0.4,0.6],
                        restitution_list=[]
                    )
                )
            },

        )

    )


PSI_DC_02_CFG = BASE_CFG.replace(
    robots_cfg = {
        "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_PsiSynHand.usd",
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
                        "arm1_joint_link2": -1.73,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 0.94,
                        "arm2_joint_link1": 0.54,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.85681702,
                        "arm2_joint_link7": 1.871788888888889,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_1_2":0.0,
                        "hand1_joint_link_1_3":0.0,
                        "hand1_joint_link_1_4":0.0,
                        "hand1_joint_link_1_5":0.0,
                        "hand1_joint_link_2_1":0.0,
                        "hand1_joint_link_2_2":0.0,
                        "hand1_joint_link_2_3":0.0,
                        "hand1_joint_link_2_4":0.0,
                        "hand1_joint_link_3_1":0.0,
                        "hand1_joint_link_3_2":0.0,
                        "hand1_joint_link_3_3":0.0,
                        "hand1_joint_link_3_4":0.0,
                        "hand1_joint_link_4_1":0.0,
                        "hand1_joint_link_4_2":0.0,
                        "hand1_joint_link_4_3":0.0,
                        "hand1_joint_link_4_4":0.0,
                        "hand1_joint_link_5_1":0.0,
                        "hand1_joint_link_5_2":0.0,
                        "hand1_joint_link_5_3":0.0,
                        "hand1_joint_link_5_4":0.0,
                        "hand2_joint_link_1_1":0.0,
                        "hand2_joint_link_1_2":0.0,
                        "hand2_joint_link_1_3":0.0,
                        "hand2_joint_link_1_4":0.0,
                        "hand2_joint_link_1_5":0.0,
                        "hand2_joint_link_2_1":0.0,
                        "hand2_joint_link_2_2":0.0,
                        "hand2_joint_link_2_3":0.0,
                        "hand2_joint_link_2_4":0.0,
                        "hand2_joint_link_3_1":0.0,
                        "hand2_joint_link_3_2":0.0,
                        "hand2_joint_link_3_3":0.0,
                        "hand2_joint_link_3_4":0.0,
                        "hand2_joint_link_4_1":0.0,
                        "hand2_joint_link_4_2":0.0,
                        "hand2_joint_link_4_3":0.0,
                        "hand2_joint_link_4_4":0.0,
                        "hand2_joint_link_5_1":0.0,
                        "hand2_joint_link_5_2":0.0,
                        "hand2_joint_link_5_3":0.0,
                        "hand2_joint_link_5_4":0.0,
                        
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
                            "hand1_joint_link_1_2",
                            "hand1_joint_link_1_3",
                            "hand1_joint_link_1_4",
                            "hand1_joint_link_1_5",
                            "hand1_joint_link_2_1",
                            "hand1_joint_link_2_2",
                            "hand1_joint_link_2_3",
                            "hand1_joint_link_2_4",
                            "hand1_joint_link_3_1",
                            "hand1_joint_link_3_2",
                            "hand1_joint_link_3_3",
                            "hand1_joint_link_3_4",
                            "hand1_joint_link_4_1",
                            "hand1_joint_link_4_2",
                            "hand1_joint_link_4_3",
                            "hand1_joint_link_4_4",
                            "hand1_joint_link_5_1",
                            "hand1_joint_link_5_2",
                            "hand1_joint_link_5_3",
                            "hand1_joint_link_5_4"],
                        stiffness=None,
                        damping=None,

                    ),
                    "hand2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand2_joint_link_1_1",
                            "hand2_joint_link_1_2",
                            "hand2_joint_link_1_3",
                            "hand2_joint_link_1_4",
                            "hand2_joint_link_1_5",
                            "hand2_joint_link_2_1",
                            "hand2_joint_link_2_2",
                            "hand2_joint_link_2_3",
                            "hand2_joint_link_2_4",
                            "hand2_joint_link_3_1",
                            "hand2_joint_link_3_2",
                            "hand2_joint_link_3_3",
                            "hand2_joint_link_3_4",
                            "hand2_joint_link_4_1",
                            "hand2_joint_link_4_2",
                            "hand2_joint_link_4_3",
                            "hand2_joint_link_4_4",
                            "hand2_joint_link_5_1",
                            "hand2_joint_link_5_2",
                            "hand2_joint_link_5_3",
                            "hand2_joint_link_5_4"],
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
                    "head_camera_rgb": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera_head_base/camera_head_color",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                    "arm2_camera_rgb": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/falan2/arm2_camera_rgb",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                },
            )
    }
)
