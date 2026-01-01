# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

""" Common Modules  """ 
import numpy

""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    RigidBodyMaterialCfg,
    PinholeCameraCfg,
    PreviewSurfaceCfg,
    RigidBodyPropertiesCfg,
    MassPropertiesCfg)
from isaaclab.assets import RigidObjectCfg,ArticulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg


""" Psi Lab Modules  """
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene import SceneCfg
from psilab.assets.robot import RobotBaseCfg
from psilab.assets.light import DomeLightCfg
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

        # rigid objects
        rigid_objects_cfg ={
            #
            "table" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Table", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/table/CubeTable.usd",
                    scale=(1.5, 1.0, 0.6),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.1,0.1,0.1)
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                ),
                physics_material= RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=1.0,
                )
            ),
            #
            "target" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Lego",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/lego/lego_1x2.usd",
                    scale=(1.0,1.0,1.0),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    ),
            
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.2,0,0.7),
                    rot= (1,0,0,0)
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=0.0,
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply"
                ),
                enable_height_offset=True
            ),
        },
        
        # tiled camera sensor
        tiled_cameras_cfg = {
            "top_camera": TiledCameraCfg(
                prim_path="/World/envs/env_[0-9]+/top_camera",
                offset = TiledCameraCfg.OffsetCfg(
                    pos = (0.2,0.0,1.6),
                    rot = (0.707,0.0,0.707,0.0),
                    convention = "world"
                ),
                data_types=["rgb"],
                width=640,
                height=480,
                spawn=PinholeCameraCfg(),
            ),
            "front_camera": TiledCameraCfg(
                prim_path="/World/envs/env_[0-9]+/front_camera",
                offset = TiledCameraCfg.OffsetCfg(
                    pos = (1.0,0.0,0.7),
                    rot = (0.0,0.0,0.0,1.0),
                    convention = "world"
                ),
                data_types=["rgb"],
                width=640,
                height=480,
                spawn=PinholeCameraCfg(),
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

        # random config
        random = RandomCfg(
            rigid_objects_cfg = {
                "target": RigidRandomCfg(
                    #
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
                        # for sync reset
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
                    )
                )
            },

        )

    )


PSI_AWH_01_CFG = BASE_CFG.replace(
    robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_AWH_01/PsiRobot_AWH_01_Left_Flattened.usd",
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(-0.3, 0.0, 0.6),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "joint_rev_link1": 0.0,
                        "joint_rev_link2": 0.45,
                        "joint_rev_link3": 0.0,
                        "joint_rev_link4": 1.78,
                        "joint_rev_link5": 0.0,
                        "joint_rev_link6": -0.5,
                        "joint_rev_link7": -2.54,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_2_1":3.10,
                        "hand1_joint_link_3_1":3.06,
                        "hand1_joint_link_4_1":3.07,
                        "hand1_joint_link_5_1":3.04,
                        "hand1_joint_link_1_2":0.63,
                        "hand1_joint_link_2_2":1.56,
                        "hand1_joint_link_3_2":1.56,
                        "hand1_joint_link_4_2":1.56,
                        "hand1_joint_link_5_2":1.56,
                        "hand1_joint_link_1_3":0.03,
                    }
                ),
                                    
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "joint_rev_link1",
                            "joint_rev_link2",
                            "joint_rev_link3",
                            "joint_rev_link4",
                            "joint_rev_link5",
                            "joint_rev_link6",
                            "joint_rev_link7",
                        ],
                        stiffness=None,
                        damping=None,
                    ),
                    "hand": ImplicitActuatorCfg(
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
                            "hand1_joint_link_1_3",
                        ],
                        stiffness=None,
                        damping=None,
                    ),
                },
                eef_links={
                    "arm":"link7_left",
                },
                tiled_cameras={
                    "wrist_camera": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera/camera",
                        offset = TiledCameraCfg.OffsetCfg(
                            convention = "world"
                        ),
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    )
                }
    
            )     
        },
)