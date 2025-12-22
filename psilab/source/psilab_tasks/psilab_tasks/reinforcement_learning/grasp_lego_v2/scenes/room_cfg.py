# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

""" Common Modules  """ 

""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg


""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR

from .empty_cfg import PSI_DC_01_CFG as PSI_DC_01_EMPTY_CFG
from .empty_cfg import PSI_DC_02_CFG as PSI_DC_02_EMPTY_CFG


PSI_DC_01_CFG = PSI_DC_01_EMPTY_CFG.replace(

    env_spacing=10.0, 
    static_objects_cfg={
        "room" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Room", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (0.707, 0.707, 0.0, 0.0)
                )
        ),
    },
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
    }

)

PSI_DC_02_CFG = PSI_DC_02_EMPTY_CFG.replace(

    env_spacing=10.0, 
    static_objects_cfg={
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

    }

)