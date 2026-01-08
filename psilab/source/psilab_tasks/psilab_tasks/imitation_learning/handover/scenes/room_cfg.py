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
from isaaclab.markers import VisualizationMarkersCfg
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
            # 右手 (hand2) 接触传感器
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
            
            # 左手 (hand1) 接触传感器
            "hand1_link_base": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_base",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_1_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_1_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_1_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_1_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_1_3": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_1_3",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_2_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_2_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_2_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_2_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_3_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_3_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_3_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_3_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_4_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_4_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_4_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_4_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_5_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_5_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
            "hand1_link_5_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_5_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Bottle"],
            ),
        },
        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
                "object": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.02, 0.02, 0.02),
                ),
                # "target_position": sim_utils.UsdFileCfg(
                #     usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                #     scale=(0.02, 0.02, 0.02),
                # ),
                "handover_pos": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.02, 0.02, 0.02),
                ),
                # "middle_point": sim_utils.UsdFileCfg(
                #     usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                #     scale=(0.01, 0.01, 0.01),
                # ),


            },

        ),
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
                        # offset_range=[0.0,0.0,0.0],
                        offset_range=[0.01,0.01,0.0],
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
                    width=640,
                    height=480,
                    spawn=None,
                ),

                "third_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_third_base/camera_third_color",
                    data_types=["rgb", "depth","normals","instance_segmentation_fast"],
                    width=640,
                    height=480,
                    spawn=None,
                ),
                "chest_camera": TiledCameraCfg(
                    prim_path="/World/envs/env_[0-9]+/Robot/camera_chest_base/camera_chest_color",
                    data_types=["rgb", "depth","normals","instance_segmentation_fast"],
                    width=640,
                    height=480,
                    spawn=None,
                ),


            },        
            )
        },
     
)



PSI_DC_Handover_CFG = PSI_DC_02_CFG.replace(
    rigid_objects_cfg={
        "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_250ml/VolumetricFlask002.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/glass_graduated_cylinder_500ml/GraduatedCylinder003.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_500ml/Beaker005.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_500ml/VolumetricFlask003.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/funnel_stand/FunnelStand001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_reagent_bottle_large/ReagentBottle001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/mortar/Mortar001.usd",

                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_500ml/VolumetricFlask003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/plastic_cylinder_100ml/GraduatedCylinder001.usd",
                    
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_50ml/Beaker002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_500ml/Beaker005.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_reagent_bottle_large/ReagentBottle002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/hygrothermometer/TemperatureAndHumidityMeter001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_test_tube_20ml/TestTube001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_test_tube_50ml/TestTube002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_reagent_bottle_small/ReagentBottle004.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_reagent_bottle_small/ReagentBottle003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/spherical_condenser/SphericalCondenser001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/erlenmeyer_flask/ErlenmeyerFlask001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Micropipette/Micropipette001_new.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_cylinder_100ml/GraduatedCylinder004.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_1000ml/VolumetricFlask004.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_250ml/VolumetricFlask002.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_500ml/VolumetricFlask003.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/plastic_cylinder_100ml/GraduatedCylinder001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Centrifuge_Tube/CentrifugeTube001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_volumetric_flask_250ml/VolumetricFlask001.usd",
                    # usd_path  = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Test_Tube_Rack/TestTubeRack001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/plastic_cylinder_100ml/GraduatedCylinder001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Glass_Graduated_Cylinder_500ml/GraduatedCylinder003.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Plastic_Graduated_Cylinder_500ml/GraduatedCylinder002.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Alcohol_Lamp/AlcoholLamp001.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Erlenmeyer_Flask_With_Stopper/ErlenmeyerFlask002.usd",
                    # usd_path = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets_new/Funnel/Funnel001.usd",
                    ##normal
                    scale=(1.0,1.0,1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    ),
                    semantic_tags=[("class","target")]
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    #old
                    # pos=(-1.15358,-5.10277, 1.15 ),
                    
                    ##泛化都0.05吧


                    ## 烧杯系手势的位置
                    ## 烧杯系手势x上限+-0.1，y上限+-0.2
                    # pos=(-1.10 , -5.10   , 1.25 ),

                    ##烧杯位置
                    # pos=(-1.15 , -5.25    , 1.2 ),

                    ##坩埚位置
                    pos=(-1.15 , -5.10    , 1.2 ),
                    rot= (0.0, 0.0, 0.0, 1.0),
                    
                    ## 坩埚系手势x上限+-0.05，y上限+-0.10
                    ## 坩埚系手势的位置
                    # pos=(-1.15 - 0.00  ,-5.2    , 1.2 ),
                    # pos=(-1.15 - 0.00  ,-5.10    , 1.2 ),
                    # rot= (1.0, 0.0, 0.0, 0.0),

                    #试管架
                    # pos=(-1.25, -5.25, 1.2),
                    # rot = (0.707, 0.0, 0.0, 0.707)


                    # ##normal
                    # rot= (1.0, 0.0, 0.0, 0.0),


                    #离心管：
                    # rot = ( 0, 0, 0.7071068, 0.7071068  ),

                    ##烧杯
                    # rot= (0.0, 0.0, 0.0, 1.0),

                    ##温湿度计
                    # rot= ( -0.2623749, 0, 0, 0.964966)

                    #漏斗架
                    # rot= (  -0.7071, 0, 0, -0.7071 )



                ),
                enable_height_offset=True
            ),


        
        # "table" : RigidObjectCfg(
        #             prim_path="/World/envs/env_[0-9]+/Table", 
        #             spawn=sim_utils.UsdFileCfg(
        #                 usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd",
        #                 scale=(1.0, 1.0, 1.22),
        #                 visual_material=None,
        #                 rigid_props=RigidBodyPropertiesCfg(
        #                     kinematic_enabled = True,
        #                     solver_position_iteration_count=255
        #                 )
        #             ),
        #             init_state = RigidObjectCfg.InitialStateCfg(
        #                 # pos=(0.65, 0.0, 0.0), 
        #                 pos=(-1.65 + 0.54, -5.0, 0.0),
        #                 rot= (0.707, 0.0, 0.0, 0.707)
        #             )
        #         ),

    },


    )





PSI_DC_Grasp_Residual_RL_CFG = PSI_DC_02_CFG.replace(

    rigid_objects_cfg={
        "bottle" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/funnel_stand/FunnelStand001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_reagent_bottle_large/ReagentBottle001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/mortar/Mortar001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_50ml/Beaker002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_500ml/Beaker005.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_reagent_bottle_large/ReagentBottle002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/hygrothermometer/TemperatureAndHumidityMeter001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_test_tube_20ml/TestTube001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_test_tube_50ml/TestTube002.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_cylinder_100ml/GraduatedCylinder004.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_reagent_bottle_small/ReagentBottle004.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_reagent_bottle_small/ReagentBottle003.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/spherical_condenser/SphericalCondenser001.usd",
                    # usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/erlenmeyer_flask/ErlenmeyerFlask001.usd",

                    
                    ##normal
                    scale=(1.0,1.0,1.0),

                    ##motar
                    # scale=(0.8,0.8,1.1),

                    ##漏斗架
                    # scale= (0.3,0.3,0.6),

                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255
                    ),
                    semantic_tags=[("class","target")]
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    #normal
                    pos=(-1.15358,-5.10277,1.05 ),
                    # ##normal
                    rot= (1.0, 0.0, 0.0, 0.0),

                    ##温湿度计
                    # rot= ( -0.2623749, 0, 0, 0.964966)

                    #漏斗架
                    # rot= (  -0.7071, 0, 0, -0.7071 )

                    #50ml试管
                    # pos=(-1.1431-0.01, -5.2052, 1.09),
                    # rot= (0.0, 0.0, 0.0, 1.0)

                    ##100ml玻璃量筒
                    # pos=(-1.15358,-5.10277,1.13 )
                ),
                enable_height_offset=True
            ),

            ##50ml试管架和试管
            # "shelf" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Shelf",
            #     spawn=sim_utils.UsdFileCfg(
            #         # usd_path="/home/psibot/psi-lab-v2/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
            #         usd_path="/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/new/TestTubeRack001/TestTubeRack001.usd",
            #         # usd_path="/home/wyh/wyh/psi-lab-v2/assets/lyf_usd/objs/glass_beaker_100ml/Beaker003.usd",
            #         scale=(1.4,1.4,1.0),
            #         # scale=(1.4,1.4,1.0),
            #         rigid_props=RigidBodyPropertiesCfg(
            #             solver_position_iteration_count=2047,
            #             kinematic_enabled = True
            #         ), 
            #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         # pos=(-1.15358-0.01801, -5.2+0.01780, 1.11),
            #         pos=(-1.15358-0.01801-0.01, -5.2+0.01780, 1.01),
            #         rot=(0.707, 0.0, 0.0, 0.707)
            #     ),
            # ),
        
        # "table" : RigidObjectCfg(
        #             prim_path="/World/envs/env_[0-9]+/Table", 
        #             spawn=sim_utils.UsdFileCfg(
        #                 usd_path=PSILAB_USD_ASSET_DIR + "/Chemistrylab/table_chem.usd",
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















