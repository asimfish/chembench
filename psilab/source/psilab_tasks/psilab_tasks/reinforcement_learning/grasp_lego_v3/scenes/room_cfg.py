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
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg


""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_TEXTURE_ASSET_DIR
from psilab.random import (
    RandomCfg,
    RigidRandomCfg,
    VisualMaterialRandomCfg,
    RigidPhysicMaterialRandomCfg,
    PositionRandomCfg,
    OrientationRandomCfg,
    MassRandomCfg,
    LightRandomCfg
)

from psilab.assets.light.light_cfg import (
    DomeLightCfg,
    DiskLightCfg
)

from .empty_cfg import PSI_DC_02_CFG as PSI_DC_02_EMPTY_CFG


PSI_DC_02_CFG = PSI_DC_02_EMPTY_CFG.replace(

    env_spacing=10.0, 

    # local light
    local_lights_cfg={
        "disk_light_01":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_01",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_02":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_02",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_03":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_03",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_04":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_04",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_05":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_05",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_06":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_06",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_07":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_07",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_08":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_08",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_09":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_09",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_10":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_10",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_11":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_11",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        "disk_light_12":DiskLightCfg(
            prim_path="/World/envs/env_[0-9]+/Room/Lights/DiskLight_12",
            light_type="DiskLight",
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2,1,2)
            )
        ),
        
    },
    #
    global_light_cfg = None,
    #
    static_objects_cfg={
        "room" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Room", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_hall/Psi_hall_test/Psi_hall_test.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(-3.0, -4.0, 0.2), 
                    rot= (0.707,0,0,0.707)
                )
        )

    },
    #
    rigid_objects_cfg = {
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
                rot= (1,0,0,0)
            ),
            enable_height_offset=True
        ),

    },

    random = RandomCfg(
        rigid_objects_cfg = {
            "table" : RigidRandomCfg(
                visual_material = VisualMaterialRandomCfg(
                    enable=True,
                    shader_path=["/Looks/Wood/Shader"],
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
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/Textures/WillowTable_Wood_A.png",
                        PSILAB_TEXTURE_ASSET_DIR + "/20250311-092148.jpg",
                        PSILAB_TEXTURE_ASSET_DIR + "/20250311-092142.jpg",
                        PSILAB_TEXTURE_ASSET_DIR + "/20250311-092135.jpg",
                    ]
                ),
            ),
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
        local_lights_cfg = {
            "disk_light_01": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_02": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_03": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_04": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_05": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_06": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_07": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_08": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_09": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_10": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_11": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
            "disk_light_12": LightRandomCfg(
                random_type="range",
                random_intensity= True,
                random_color=True,
                intensity_range=[20000,60000],
                color_range = [
                    [0,0,0],
                    [255,255,255]
                ], # type: ignore
                intensity_list=[
                    0,
                    1000,
                    8000
                ],
                color_list=[
                    [0,0,0],
                    [210,124,14],
                    [0,30,20],
                    [90,0,100],
                ] # type: ignore
            ),
        }

        )
    
)