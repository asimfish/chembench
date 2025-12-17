# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

import numpy

""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg,TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence_cfg import SceneCfg
from psilab.assets.robot.robot_base_cfg import RobotBaseCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg
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
        env_spacing=15.0, 
        replicate_physics=True,
        
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
            "room" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Room", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (0.707, 0.707, 0.0, 0.0)
                )
            )
        },
        
        # rigid objects
        rigid_objects_cfg ={

            "table" : RigidObjectCfg(
                    prim_path="/World/envs/env_[0-9]+/Table", 
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
                        scale=(1.0, 1.0, 1.0),
                        visual_material=None,
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = True,
                            solver_position_iteration_count=255
                        )
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.65, 0.0, 0.0), 
                        rot= (0.707, 0.0, 0.0, 0.707)
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
                    rot= (0.707,0,0,0.707)
                ),
                enable_height_offset=True
            ),
          
        },
        
        # random config
        random = RandomCfg(
            rigid_objects_cfg = {
                # "target": RigidRandomCfg(
                #     position= PositionRandomCfg(
                #         enable=[True,True,False],
                #         type="range",
                #         offset_range=[0.05,0.05,0.0],
                #         offset_list=[
                #             [0.1,0.0,0.0],
                #             [0.0,0.0,0.0],
                #             [-0.1,0.0,0.0],
                #             [0.0,-0.1,0.0],
                #         ],
                #     ),
                #     visual_material = VisualMaterialRandomCfg(
                #         enable=False,
                #         shader_path=["/Looks/material/Shader"],
                #         random_type="range",
                #         material_type = "color",
                #         color_range=[
                #             [0,0,0],
                #             [255,255,255]
                #         ], # type: ignore
                #         color_list = [
                #             [0,32,54],
                #             [231,65,0],
                #             [21,123,10],
                #         ], # type: ignore
                #         roughness_range=[0.0,1.0],
                #         roughness_list=[0.0,0.5,1.0],
                #         metalness_range=[0.0,1.0],
                #         metalness_list=[0.0,0.5,1.0],
                #         specular_range=[0.0,1.0],
                #         specular_list=[0.0,0.5,1.0],
                #         texture_list =[]
                #     ),
                    
                # )
            },

        )
    )


PSI_DC_02_CFG = BASE_CFG.replace(
        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02.usd",
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
                        "arm2_joint_link1": 0.24,
                        "arm2_joint_link2": -0.64,
                        "arm2_joint_link3": 1.52,
                        "arm2_joint_link4": -0.81,
                        "arm2_joint_link5": -0.30,
                        "arm2_joint_link6": -1.03,
                        "arm2_joint_link7": -0.36,
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
                        damping=0.5,
                        stiffness=1,
                        

                    ),
                },
                diff_ik_controllers = {
                    "arm1":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7"
                        ],
                        eef_link_name="arm1_link7"
                    ),
                    "arm2":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7"
                        ],
                        eef_link_name="arm2_link7"
                    ),
    
                },
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },  
                tiled_cameras={
                "head_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_head_base/camera_head_color",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
                "arm2_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/falan2/arm2_camera_rgb",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
            },        
            )
        },
     
)


# BASE_IL_CFG = SceneCfg(
        
#         num_envs = 1, 
#         env_spacing=15.0, 
#         replicate_physics=True,
        
#         # global light
#         global_light_cfg = DomeLightCfg(
#             prim_path="/World/Light", 
#             spawn=sim_utils.DomeLightCfg(
#                 intensity=3000.0, 
#                 color=(0.75, 0.75, 0.75)
#             )
#         ),

#         # robot
#         robots_cfg = {
#             "robot" : RobotBaseCfg(
#                 prim_path = "/World/Robot",
#                 spawn = sim_utils.UsdFileCfg(
#                     usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02.usd",
#                     activate_contact_sensors = True,

#                     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                         enabled_self_collisions=True,
#                     ),
#                     rigid_props=RigidBodyPropertiesCfg(
#                         solver_position_iteration_count=255,
#                     )
#                 ),
#                 init_state=ArticulationCfg.InitialStateCfg(
#                     pos=(0.0, 0.0, 0.0),
#                     rot=(1.0,0.0,0.0,0.0),
#                     joint_pos={
#                         "arm1_joint_link1": -0.24,
#                         "arm1_joint_link2": -0.64,
#                         "arm1_joint_link3": -1.52,
#                         "arm1_joint_link4": -0.81,
#                         "arm1_joint_link5": 0.30,
#                         "arm1_joint_link6": -1.03,
#                         "arm1_joint_link7": 1.35,
#                         "arm2_joint_link1": 0.24,
#                         "arm2_joint_link2": -0.64,
#                         "arm2_joint_link3": 1.52,
#                         "arm2_joint_link4": -0.81,
#                         "arm2_joint_link5": -0.30,
#                         "arm2_joint_link6": -1.03,
#                         "arm2_joint_link7": -0.36,
#                         "hand1_joint_link_1_1":0.0,
#                         "hand1_joint_link_1_2":0.63,
#                         "hand1_joint_link_1_3":0.03,
#                         "hand1_joint_link_2_1":3.10,
#                         "hand1_joint_link_2_2":1.56,
#                         "hand1_joint_link_3_1":3.06,
#                         "hand1_joint_link_3_2":1.56,
#                         "hand1_joint_link_4_1":3.06,
#                         "hand1_joint_link_4_2":1.56,
#                         "hand1_joint_link_5_1":3.04,
#                         "hand1_joint_link_5_2":1.56,
#                         "hand2_joint_link_1_1":0.0,
#                         "hand2_joint_link_1_2":0.64,
#                         "hand2_joint_link_1_3":0.03,
#                         "hand2_joint_link_2_1":3.11,
#                         "hand2_joint_link_2_2":1.56,
#                         "hand2_joint_link_3_1":3.06,
#                         "hand2_joint_link_3_2":1.56,
#                         "hand2_joint_link_4_1":3.08,
#                         "hand2_joint_link_4_2":1.56,
#                         "hand2_joint_link_5_1":3.05,
#                         "hand2_joint_link_5_2":1.56,
#                     }
#                 ),
                                    
#                 actuators={
#                     "arm1": ImplicitActuatorCfg(
#                         joint_names_expr=[
#                             "arm1_joint_link1",
#                             "arm1_joint_link2",
#                             "arm1_joint_link3",
#                             "arm1_joint_link4",
#                             "arm1_joint_link5",
#                             "arm1_joint_link6",
#                             "arm1_joint_link7",
#                             ],
#                         stiffness=None,
#                         damping=None,

#                     ),
#                     "arm2": ImplicitActuatorCfg(
#                         joint_names_expr=[
#                             "arm2_joint_link1",
#                             "arm2_joint_link2",
#                             "arm2_joint_link3",
#                             "arm2_joint_link4",
#                             "arm2_joint_link5",
#                             "arm2_joint_link6",
#                             "arm2_joint_link7",
#                             ],
#                         stiffness=None,
#                         damping=None,
#                     ),
#                     "hand1": ImplicitActuatorCfg(
#                         joint_names_expr=[
#                             "hand1_joint_link_1_1",
#                             "hand1_joint_link_2_1",
#                             "hand1_joint_link_3_1",
#                             "hand1_joint_link_4_1",
#                             "hand1_joint_link_5_1",
#                             "hand1_joint_link_1_2",
#                             "hand1_joint_link_2_2",
#                             "hand1_joint_link_3_2",
#                             "hand1_joint_link_4_2",
#                             "hand1_joint_link_5_2",
#                             "hand1_joint_link_1_3"],
#                         stiffness=None,
#                         damping=None,

#                     ),
#                     "hand2": ImplicitActuatorCfg(
#                         joint_names_expr=[
#                             "hand2_joint_link_1_1",
#                             "hand2_joint_link_2_1",
#                             "hand2_joint_link_3_1",
#                             "hand2_joint_link_4_1",
#                             "hand2_joint_link_5_1",
#                             "hand2_joint_link_1_2",
#                             "hand2_joint_link_2_2",
#                             "hand2_joint_link_3_2",
#                             "hand2_joint_link_4_2",
#                             "hand2_joint_link_5_2",
#                             "hand2_joint_link_1_3"],
#                         damping=0.5,
#                         stiffness=1,
                        

#                     ),
#                 },
#                 diff_ik_controllers = {
#                     "arm1":DiffIKControllerCfg(
#                         command_type="pose", 
#                         use_relative_mode=False, 
#                         ik_method="dls",
#                         joint_name=[
#                             "arm1_joint_link1",
#                             "arm1_joint_link2",
#                             "arm1_joint_link3",
#                             "arm1_joint_link4",
#                             "arm1_joint_link5",
#                             "arm1_joint_link6",
#                             "arm1_joint_link7"
#                         ],
#                         eef_link_name="arm1_link7"
#                     ),
#                     "arm2":DiffIKControllerCfg(
#                         command_type="pose", 
#                         use_relative_mode=False, 
#                         ik_method="dls",
#                         joint_name=[
#                             "arm2_joint_link1",
#                             "arm2_joint_link2",
#                             "arm2_joint_link3",
#                             "arm2_joint_link4",
#                             "arm2_joint_link5",
#                             "arm2_joint_link6",
#                             "arm2_joint_link7"
#                         ],
#                         eef_link_name="arm2_link7"
#                     ),
    
#                 },
#                 eef_links={
#                     "arm1":"arm1_link7",
#                     "arm2":"arm2_link7"
#                 },
#                 tiled_cameras={
#                     "base_camera": TiledCameraCfg(
#                         prim_path="/World/Robot/base_camera_rgb/base_camera_rgb",
#                         data_types=["rgb"],
#                         width=640,
#                         height=480,
#                         spawn=None,
#                     ),
#                     # "arm1_camera": TiledCameraCfg(
#                     #     prim_path="/World/Robot/arm1_camera_rgb/arm1_camera_rgb",
#                     #     data_types=["rgb"],
#                     #     width=640,
#                     #     height=480,
#                     #     spawn=None,
#                     # ),
#                     "arm2_camera": TiledCameraCfg(
#                         prim_path="/World/Robot/arm2_camera_rgb/arm2_camera_rgb",
#                         data_types=["rgb"],
#                         width=640,
#                         height=480,
#                         spawn=None,
#                     ),
#                 },            
#             )
#         },
        
#         # static object
#         static_objects_cfg = {
#             "room" : AssetBaseCfg(
#                 prim_path="/World/envs/env_[0-9]+/Room", 
#                 spawn=sim_utils.UsdFileCfg(
#                     usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
#                 ),
#                 init_state = RigidObjectCfg.InitialStateCfg(
#                     pos=(0.0, 0.0, 0.0), 
#                     rot= (0.707, 0.707, 0.0, 0.0)
#                 )
#             )
#         },
        
#         # rigid objects
#         rigid_objects_cfg ={

#             "table" : RigidObjectCfg(
#                     prim_path="/World/envs/env_[0-9]+/Table", 
#                     spawn=sim_utils.UsdFileCfg(
#                         usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
#                         scale=(1.0, 1.0, 1.0),
#                         visual_material=None,
#                         rigid_props=RigidBodyPropertiesCfg(
#                             kinematic_enabled = True,
#                             solver_position_iteration_count=255
#                         )
#                     ),
#                     init_state = RigidObjectCfg.InitialStateCfg(
#                         pos=(0.65, 0.0, 0.0), 
#                         rot= (0.707, 0.0, 0.0, 0.707)
#                     )
#                 ),
#             "target" : RigidObjectCfg(
#                 prim_path="/World/Lego",
#                 spawn=sim_utils.UsdFileCfg(
#                     usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/lego/lego_1x2.usd",
#                     activate_contact_sensors = True,
#                     scale=(1.0,1.0,1.0),
#                     visual_material=None,
#                     mass_props=MassPropertiesCfg(
#                         mass = 0.01
#                     ),
#                     rigid_props=RigidBodyPropertiesCfg(
#                         solver_position_iteration_count=255,
#                     )
#                 ),
#                 init_state=RigidObjectCfg.InitialStateCfg(
#                     pos=(0.5,-0.105,0.8),
#                     rot= (0.707,0,0,0.707)
#                 ),
#                 enable_height_offset=True
#             ),
          
#         },
        
#         # rigid objects
#         deformable_objects_cfg ={},
        
#         # camera sensor
#         cameras_cfg={},
        
#         # contact sensor
#         contact_sensors_cfg={

#         },

#         # random config
#         random = None,
#         # random = RandomCfg(
#         #     rigid_objects_cfg = {
#         #         "bottle": RigidRandomCfg(
#         #             mass=MassRandomCfg(
#         #                 enable=False,
#         #                 type="range",
#         #                 mass_range=[0,1],
#         #                 mass_list=[],
#         #                 density_range=None,
#         #                 density_list=None,
#         #             ),
#         #             position= PositionRandomCfg(
#         #                 enable=[True,True,False],
#         #                 type="range",
#         #                 offset_range=[0.02,0.02,0.0],
#         #                 offset_list=[
#         #                     [0.1,0.0,0.0],
#         #                     [0.0,0.0,0.0],
#         #                     [-0.1,0.0,0.0],
#         #                     [0.0,-0.1,0.0],
#         #                 ],
#         #             ),
#         #             orientation = None,
#         #             visual_material = VisualMaterialRandomCfg(
#         #                 enable=False,
#         #                 shader_path=["/Looks/material/Shader"],
#         #                 random_type="range",
#         #                 material_type = "color",
#         #                 color_range=[
#         #                     [0,0,0],
#         #                     [255,255,255]
#         #                 ], # type: ignore
#         #                 color_list = [
#         #                     [0,32,54],
#         #                     [231,65,0],
#         #                     [21,123,10],
#         #                 ], # type: ignore
#         #                 roughness_range=[0.0,1.0],
#         #                 roughness_list=[0.0,0.5,1.0],
#         #                 metalness_range=[0.0,1.0],
#         #                 metalness_list=[0.0,0.5,1.0],
#         #                 specular_range=[0.0,1.0],
#         #                 specular_list=[0.0,0.5,1.0],
#         #                 texture_list =[]
#         #             ),
#         #             physics_material= RigidPhysicMaterialRandomCfg(
#         #                 enable=False,
#         #                 random_type="range",
#         #                 static_friction_range=[0.2,0.8],
#         #                 static_friction_list=[],
#         #                 dynamic_friction_range=[0.2,0.8],
#         #                 dynamic_friction_list=[],
#         #                 restitution_range=[0.0,0.2],
#         #                 restitution_list=[]
#         #             )
#         #         )
#         #     },

#         # )
    
#     )


