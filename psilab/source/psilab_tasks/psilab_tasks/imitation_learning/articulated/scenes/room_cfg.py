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
        # global_light_cfg = DomeLightCfg(
        #     prim_path="/World/Light", 
        #     spawn=sim_utils.DomeLightCfg(
        #         intensity=3000.0, 
        #         color=(0.75, 0.75, 0.75)
        #     )
        # ),
   
        # static object
        static_objects_cfg = {
            "room" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Room", 
                spawn=sim_utils.UsdFileCfg(
                    # usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_real.usd"


                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_simple.usd"
                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_simple_1.28.usd"
                    
                    usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_simple_1.25.usd"
                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_no_desk.usd"
                    # usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_simple_1.35.usd"
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
                    # usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
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
                    usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    
                    # usd_path="/home/psibot/psi-lab-v2/assets/Laboratory_LW_Assets_20251128/Beaker005/Beaker005.usd",
                    # usd_path="/home/psibot/psi-lab-v2/assets/Laboratory_LW_Assets_20251128/TestTube002/TestTube002.usd",
                    # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    
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
                    pos=(0.5,-0.105,1.0),
                    # rot= (0.707, 0.707, 0.0, 0.0)
                    rot= (0.0, 0.0, 0.0, 1.0)
                ),
                enable_height_offset=False
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
                        offset_range=[0.0,0.0,0.0],
                        # offset_range=[0.02,0.02,0.0],
                        # offset_range=[0.08,0.08,0.0],
                        # offset_range=[0.0,0.0,0.0],
                        # offset_list=[
                        #     [0.1,0.0,0.0],
                        #     [0.0,0.1,0.0],
                        #     [-0.1,0.0,0.0],
                        #     [0.0,-0.1,0.0],
                        # ],
                    ),
                    # orientation=OrientationRandomCfg(
                    # enable=[False, False, True],
                    # type="range",
                    # eular_base=[
                    #     [0.0,0.0,0.0],
                    #     [0.0, 0.5 * numpy.pi, 0.0],
                    #     [0.0, 1.0 * numpy.pi, 0.0],
                    #     [0.5 * numpy.pi, 0.0, 0.0],

                    # ],
                    # # pos_z_offset=[
                    # #     -0.0013,
                    # #     0.01,
                    # #     0.0187,
                    # #     -0.0050
                    # # ],
                    # eular_range=[
                    #     [0.0,0.0,numpy.pi],
                    #     [0.0,0.0,numpy.pi],
                    #     [0.0,0.0,numpy.pi],
                    #     [0.0,0.0,numpy.pi],
                    # ],
                    # eular_list=[],
                    # ),
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
                    # usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_PerfectControl9.1_aligned_fingermass_D435_640x480.usd",
                    # usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_PerfectControl9.1_aligned_fingermass_D435_640x480_minimal.usd",
                    # usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_onlyarm_liyufeng.usd",
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02_onlyarm_liyufeng_new.usd",
                    
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
                    # data_types=["rgb", "instance_segmentation_fast"],
                    data_types=["rgb", "depth","normals","instance_segmentation_fast"],
                    # data_types=["rgb", "semantic_segmentation"],  # 使用语义分割
                    # colorize_instance_segmentation=True,
                    # semantic_filter="class:target",  # 只分割 class=target 的物体
                    # colorize_instance_segmentation=False,  # 🔑 推荐：获取原始ID
                    # data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),

                "third_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_third_base/camera_third_color",
                    
                    # data_types=["rgb", "instance_segmentation_fast"],
                    data_types=["rgb", "depth","normals","instance_segmentation_fast"],
                    # data_types=["rgb", "semantic_segmentation"],  # 使用语义分割
                    # semantic_filter="class:target",  # 只分割 class=target 的物体
                    # colorize_instance_segmentation=True,  # 已开启：对实例分割图像进行着色
                    # colorize_semantic_segmentation=True, # 如果使用语义分割，可开启此项
                    # data_types=["rgb"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
                # "arm1_camera": TiledCameraCfg(
                #     prim_path="/World/envs/env_[0-9]+/Robot/falan1/arm1_camera_rgb",
                #     data_types=["rgb"],
                #     width=640,
                #     height=480,
                #     spawn=None,
                # ),
                # "arm2_camera": TiledCameraCfg(
                #     prim_path="/World/envs/env_[0-9]+/Robot/falan2/arm2_camera_rgb",
                #     data_types=["rgb"],
                #     width=640,
                #     height=480,
                #     spawn=None,
                # ),
                "chest_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_chest_base/camera_chest_color",
                    
                    # data_types=["rgb", "instance_segmentation_fast"],
                    data_types=["rgb", "depth","normals","instance_segmentation_fast"],
                    # data_types=["rgb", "semantic_segmentation"],  # 使用语义分割
                    # semantic_filter="class:target",  # 只分割 class=target 的物体
                    # colorize_instance_segmentation=True,  # 已开启：对实例分割图像进行着色
                    # colorize_semantic_segmentation=True, # 如果使用语义分割，可开启此项
                    # data_types=["rgb"],
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
                # "third_person_camera": TiledCameraCfg(
                #     prim_path="/World/envs/env_[0-9]+/ThirdPersonCamera",
                #     data_types=["rgb"],
                #     width=640,
                #     height=480,
                #     # width=640 * 4,
                #     # height=480 * 4,

                #     spawn=sim_utils.PinholeCameraCfg(),
                #     offset=TiledCameraCfg.OffsetCfg(
                #         # pos=(2.2, 0.0, 1.2),
                #         # rot=(0.0, 0.0, 0.0, 1.0),  # 180 degrees around Z axis to look towards -X
                        
                #         # pos=(5.3, 1.5, 1.2),
                #         # # rot=(0.0, 0.0, 0.0, 1.0),  # 180 degrees around Z axis to look towards -X
                #         # # rot =(0.0000, 0.0000, 0.7071, -0.7071),
                #         # rot=(-0.7071, 0, 0, 0.7071),

                #         pos = (0.6,-5.0,1.2),

                #         # rot = (0.5, -0.5, 0.5, 0.5),  # 欧拉角 (90°, 90°, 0°) XYZ顺序,
                #         # rot = (0.5000, 0.5000, 0.5000, -0.5000),
                #         # rot = (-0.5, 0.5, -0.5, -0.5),  # 欧拉角 (90°, 90°, 0°) xyz顺序
                #         rot = (0, 0, 0, 1),


                        
                #         # rot = (0, 0, -0.707, -0.707),
                        
                        
                #         convention="world"
                #     ),
                # ),
            },        
            )
        },
     
)

PSI_DC_Articulated_Open_Door_CFG = PSI_DC_02_CFG.replace(
    articulated_objects_cfg={
        "bottle" : ArticulatedObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003_art.usd",
                    scale=(1.0,1.0,1.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(-1.15 , -5.10    , 1.2 ),
                    rot= (0.0, 0.0, 0.0, 1.0),
                    # rot= (1.0, 0.0, 0.0, 0.0)
                    joint_pos={},
                    joint_vel={}
                ),
                actuators={},
            ),
    },


    rigid_objects_cfg={


    },


    )













