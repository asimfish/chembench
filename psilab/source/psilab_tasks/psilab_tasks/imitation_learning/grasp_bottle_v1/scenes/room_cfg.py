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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence_cfg import SceneCfg
from psilab.assets.robot.robot_base_cfg import RobotBaseCfg
from psilab.assets.articulated_object import ArticulatedObjectCfg
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
                    # usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_real.usd"
                    usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_change_desk.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    # pos=(-1.0, 8, -0.3), 
                    pos=(0.2, 0.0, 0.0), 
                    # pos = (0.0, 0.0, 0.0),
                    # rot= (0.0, 0.0, 0.0, 1.0)
                    rot=(0.7071, 0.0000, 0.0000, 0.7071)

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
                    # init_state = RigidObjectCfg.InitialStateCfg(
                    #     pos=(0.15, 0.0, 0.0), 
                    #     rot= (0.707, 0.0, 0.0, 0.707)
                    # )

                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.65, 0.0, 0.0), 
                        rot= (0.707, 0.0, 0.0, 0.707)
                    )
                ),
            "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    # usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/manipulable_objects/drink/B36/B36.usd",
                    # usd_path="/home/psibot/psi-lab-v2/assets/Laboratory_LW_Assets_20251128/Beaker005/Beaker005.usd",
                    # usd_path="/home/psibot/psi-lab-v2/assets/Laboratory_LW_Assets_20251128/TestTube002/TestTube002.usd",
                    usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    
                    # Beaker002_bp.usd
                    # usd_path="/home/psibot/psi-lab-v2/assets/Laboratory_LW_Assets_20251128/Beaker002/Beaker002_bp.usd",
                    # scale=(0.6,0.6,0.6),
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # pos=(0.0,0.0,0.85),
                    pos=(0.5,-0.105,0.85),
                    # rot= (0.707, 0.707, 0.0, 0.0)
                    rot= (0.0, 0.0, 0.0, 1.0)
                ),
                enable_height_offset=True
            ),
          
        },
        
        # contact sensor
        contact_sensors_cfg={
            "hand2_link_base": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_base",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_1_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_1_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_1_3": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_3",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_2_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_2_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_2_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_2_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_3_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_3_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_3_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_3_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_4_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_4_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_4_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_4_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_5_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_5_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand2_link_5_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_5_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
        },

        # random config
        random = RandomCfg(
            rigid_objects_cfg = {
                "bottle": RigidRandomCfg(
                    # mass=MassRandomCfg(
                    #     enable=False,
                    #     type="range",
                    #     mass_range=[0,1],
                    #     mass_list=[],
                    #     density_range=None,
                    #     density_list=None,
                    # ),
                    position= PositionRandomCfg(
                        enable=[True,True,False],
                        type="range",
                        offset_range=[0.03,0.03,0.0],
                        # offset_list=[
                        #     [0.1,0.0,0.0],
                        #     [0.0,0.1,0.0],
                        #     [-0.1,0.0,0.0],
                        #     [0.0,-0.1,0.0],
                        # ],
                    ),
                    # orientation = None,
                    # visual_material = VisualMaterialRandomCfg(
                    #     enable=False,
                    #     shader_path=["/Looks/material/Shader"],
                    #     random_type="range",
                    #     material_type = "color",
                    #     color_range=[
                    #         [0,0,0],
                    #         [255,255,255]
                    #     ], # type: ignore
                    #     color_list = [
                    #         [0,32,54],
                    #         [231,65,0],
                    #         [21,123,10],
                    #     ], # type: ignore
                    #     roughness_range=[0.0,1.0],
                    #     roughness_list=[0.0,0.5,1.0],
                    #     metalness_range=[0.0,1.0],
                    #     metalness_list=[0.0,0.5,1.0],
                    #     specular_range=[0.0,1.0],
                    #     specular_list=[0.0,0.5,1.0],
                    #     texture_list =[]
                    # ),
                    # physics_material= RigidPhysicMaterialRandomCfg(
                    #     enable=False,
                    #     random_type="range",
                    #     static_friction_range=[0.2,0.8],
                    #     static_friction_list=[],
                    #     dynamic_friction_range=[0.2,0.8],
                    #     dynamic_friction_list=[],
                    #     restitution_range=[0.0,0.2],
                    #     restitution_list=[]
                    # )
                )
            },

        )
    
    )








PSI_DC_01_CFG = BASE_CFG.replace(
    # robot
    robots_cfg = {
        "robot" : RobotBaseCfg(
            prim_path = "/World/envs/env_[0-9]+/Robot",
            spawn = sim_utils.UsdFileCfg(
                usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_01/PsiRobot_DC_01.usd",
                activate_contact_sensors = True,

                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=255,
                )
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.5, 0.0, 0.0),
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
                    stiffness=None,
                    damping=None,

                ),
            },
            diff_ik_controllers = {
                "arm1":DiffIKControllerCfg(
                    command_type="pose", 
                    use_relative_mode=False, 
                    ik_method="dls",
                    ik_params={"lambda_val": 0.05},  # 增大阻尼系数，提高稳定性
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
                    ik_params={"lambda_val": 0.05},  # 增大阻尼系数，提高稳定性
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
    },
)

# stiffness_arm = 500
# damping_arm = 55
stiffness_arm = None
damping_arm = None
# stiffness_hand = 1
# damping_hand = 0.5
stiffness_hand = None
damping_hand = None
# stiffness_hand = 10
# damping_hand = 5

PSI_DC_02_CFG = BASE_CFG.replace(
        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_PerfectControl9.1_aligned_fingermass_D435_640x480.usd",
                    # usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_PerfectControl9.1_aligned_fingermass_D435_640x480_minimal.usd",
                    activate_contact_sensors = True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    # pos=(0.0, 0.0, 0.0),
                    # pos=(-1.7, -5.0, 0.0),
                    pos=(-1.65, -5.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    # rot=(0.0, 0.0, 0.0, 1.0),
                    # rot=(0.7071, 0.0000, 0.0000, 0.7071),
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
                        stiffness=stiffness_arm,
                        damping=damping_arm,

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
                        stiffness=stiffness_arm,
                        damping=damping_arm,
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
                        stiffness=stiffness_hand,
                        damping=damping_hand,
                        # damping=0.5,
                        # stiffness=1,

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
                        # damping=0.5,
                        # stiffness=1,
                            stiffness=stiffness_hand,
                            damping=damping_hand,
                        

                    ),
                },
                diff_ik_controllers = {
                    "arm1":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        ik_params={"lambda_val": 0.05},  # 增大阻尼系数，提高稳定性
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
                        ik_params={"lambda_val": 0.05},  # 增大阻尼系数，提高稳定性
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
                "arm1_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/falan1/arm1_camera_rgb",
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
                "chest_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_chest_base/camera_chest_color",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                    # offset=TiledCameraCfg.OffsetCfg(
                    #     pos=(0.045, -0.0115, -0.00005),
                    #     # 欧拉角 (X=-1.085°, Y=-60.448°, Z=-91.299°) 转换为四元数
                    #     rot=(0.601, -0.366, -0.347, -0.620),
                    #     convention="ros"
                    # ),
                ),
                "third_person_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/ThirdPersonCamera",
                    data_types=["rgb"],
                    width=640,
                    height=480,
                    # width=640 * 4,
                    # height=480 * 4,

                    spawn=sim_utils.PinholeCameraCfg(),
                    offset=TiledCameraCfg.OffsetCfg(
                        # pos=(2.2, 0.0, 1.2),
                        # rot=(0.0, 0.0, 0.0, 1.0),  # 180 degrees around Z axis to look towards -X
                        
                        # pos=(5.3, 1.5, 1.2),
                        # # rot=(0.0, 0.0, 0.0, 1.0),  # 180 degrees around Z axis to look towards -X
                        # # rot =(0.0000, 0.0000, 0.7071, -0.7071),
                        # rot=(-0.7071, 0, 0, 0.7071),

                        pos = (0.6,-5.0,1.2),

                        # rot = (0.5, -0.5, 0.5, 0.5),  # 欧拉角 (90°, 90°, 0°) XYZ顺序,
                        # rot = (0.5000, 0.5000, 0.5000, -0.5000),
                        # rot = (-0.5, 0.5, -0.5, -0.5),  # 欧拉角 (90°, 90°, 0°) xyz顺序
                        rot = (0, 0, 0, 1),


                        
                        # rot = (0, 0, -0.707, -0.707),
                        
                        
                        convention="world"
                    ),
                ),
            },        
            )
        },
     
)



PSI_DC_Beaker_003_CFG = PSI_DC_02_CFG.replace(
    rigid_objects_cfg={

        "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/mortar/Mortar001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_50ml/Beaker002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_500ml/Beaker005.usd",

                    ##beaker
                    scale=(1.0,1.0,1.0),

                    ##motar
                    # scale=(0.8,0.8,1.1),

                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # pos=(0.5,-0.105,0.85),
                    # pos=(-1.250358,-5.30277,0.85 ),
                    # pos=(-1.050358,-5.30277,0.85 ),
                    # pos=(-1.150358,-5.30277,1.07 ),
                    # pos=(-1.15358,-5.10277,0.9 ),


                    pos=(-1.15358,-5.10277,1.05 ),
                    rot= (0.0, 0.0, 0.0, 1.0)

                    # pos=(-1.15358,-5.20277,1.10 ),
                    # rot= (0.0, 0.0, 0.0, 1.0)
                ),
                enable_height_offset=True
            ),
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

    },


    )


PSI_DC_Chem_CFG = PSI_DC_02_CFG.replace(

    rigid_objects_cfg={


        "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_50ml/Beaker002.usd",

                    # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
                    # scale=(0.4,0.4,0.8),
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # pos=(0.5,-0.105,0.85),
                    # pos=(-5.15,0.93,0.85),
                    pos=(-1.250358,-5.30277,1.35 ),


                    rot= (0.0, 0.0, 0.0, 1.0)
                ),
                enable_height_offset=True
            ),


        # "bottle1" : RigidObjectCfg(
        #         prim_path="/World/envs/env_[0-9]+/Bottle",
        #         spawn=sim_utils.UsdFileCfg(
        #             usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",

        #             # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
        #             # scale=(0.4,0.4,0.8),
        #             scale=(1.0,1.0,1.0),
        #             rigid_props=RigidBodyPropertiesCfg(
        #                 solver_position_iteration_count=255
        #             )
        #         ),
        #         init_state=RigidObjectCfg.InitialStateCfg(
        #             # pos=(0.5,-0.105,0.85),
        #             # pos=(-5.15,0.93,0.85),
        #             pos=(-1.00358,-5.30277,1.45 ),


        #             rot= (0.0, 0.0, 0.0, 1.0)
        #         ),
        #         enable_height_offset=True
        #     ),



        # "bottle2" : RigidObjectCfg(
        #         prim_path="/World/envs/env_[0-9]+/Bottle",
        #         spawn=sim_utils.UsdFileCfg(
        #             usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",

        #             # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
        #             # scale=(0.4,0.4,0.8),
        #             scale=(1.0,1.0,1.0),
        #             rigid_props=RigidBodyPropertiesCfg(
        #                 solver_position_iteration_count=255
        #             )
        #         ),
        #         init_state=RigidObjectCfg.InitialStateCfg(
        #             # pos=(0.5,-0.105,0.85),
        #             # pos=(-5.15,0.93,0.85),
        #             pos=(-0.7358,-5.30277,1.45 ),


        #             rot= (0.0, 0.0, 0.0, 1.0)
        #         ),
        #         enable_height_offset=True
        #     ),
        # "tube" : RigidObjectCfg(
        #         prim_path="/World/envs/env_[0-9]+/Tube",
        #         spawn=sim_utils.UsdFileCfg(
        #             usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_test_tube_50ml/TestTube002.usd",

        #             # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
        #             scale=(0.9,0.9,1.0),
        #             rigid_props=RigidBodyPropertiesCfg(
        #                 solver_position_iteration_count=255
        #             )
        #         ),
        #         init_state=RigidObjectCfg.InitialStateCfg(
        #             # pos=(0.5,-0.105,0.85),
        #             # pos=(-5.15,0.93,0.85),
        #             # 0.01741 0.01844 0.07021
        #             pos=(-1.150358 + 0.01741,-5.10277 + 0.01844,1.2 + 0.14021 ),


        #             rot= (0.0, 0.0, 0.0, 1.0)
        #         ),
        #         enable_height_offset=True
        #     ),


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

    },


    )










PSI_DC_Beaker_003_Art_CFG = PSI_DC_02_CFG.replace(
    # 使用 articulated_objects_cfg 导入杯子作为 Articulation
    articulated_objects_cfg={
        "bottle" : ArticulatedObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003_art.usd",
                    scale=(1.0,1.0,1.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.5, -0.105, 0.85),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    joint_pos={},
                    joint_vel={}
                ),
                # actuators 是必需字段，即使没有关节也需要提供空字典
                actuators={},
            ),
    },

    rigid_objects_cfg={

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

        # "table" : RigidObjectCfg(
        #             prim_path="/World/envs/env_[0-9]+/Table", 
        #             spawn=sim_utils.UsdFileCfg(
        #                 usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
        #                 scale=(1.0, 1.0, 1.0),
        #                 visual_material=None,
        #                 rigid_props=RigidBodyPropertiesCfg(
        #                     kinematic_enabled = True,
        #                     solver_position_iteration_count=255
        #                 )
        #             ),
        #             init_state = RigidObjectCfg.InitialStateCfg(
        #                 pos=(0.65, 0.0, 0.0), 
        #                 rot= (0.707, 0.0, 0.0, 0.707)
        #             )
        #         ),
    },
    )



PSI_DC_01_VUER_CFG = PSI_DC_01_CFG.replace(
    # camera sensor
    cameras_cfg={
        "eye_left": CameraCfg(
            height=720,
            width=1280,
            data_types=['rgb'],
            prim_path = "/World/CameraLeft",
            spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            offset = CameraCfg().OffsetCfg(
                pos = (-0.3,0.033,1.6),
                rot = (1,0,0,0),
                convention='world')
        ),
        "eye_right": CameraCfg(
            height=720,
            width=1280,
            data_types=['rgb'],
            prim_path = "/World/CameraRight",
            spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            offset = CameraCfg().OffsetCfg(
                pos = (-0.3,-0.033,1.6),
                rot = (1,0,0,0),
                convention='world')
        ),
    },
)

PSI_DC_02_VUER_CFG = PSI_DC_02_CFG.replace(
    # camera sensor
    cameras_cfg={
        "eye_left": CameraCfg(
            height=720,
            width=1280,
            data_types=['rgb'],
            prim_path = "/World/CameraLeft",
            spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            offset = CameraCfg().OffsetCfg(
                pos = (-0.3,0.033,1.6),
                rot = (1,0,0,0),
                convention='world')
        ),
        "eye_right": CameraCfg(
            height=720,
            width=1280,
            data_types=['rgb'],
            prim_path = "/World/CameraRight",
            spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            offset = CameraCfg().OffsetCfg(
                pos = (-0.3,-0.033,1.6),
                rot = (1,0,0,0),
                convention='world')
        ),






    },
)

