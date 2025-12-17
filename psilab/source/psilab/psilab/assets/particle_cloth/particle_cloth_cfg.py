# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-22
# Vesion: 1.0


""" IsaacLab Modules  """ 
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation,ArticulationCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg

from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg


""" PsiLab Modules  """ 
from psilab.assets.particle_cloth import ParticleCloth
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg

@configclass
class ParticleClothCfg(AssetBaseCfg):
    """Configuration parameters for an robot_base with cameras and ik controllers.""" 

    ##
    # Initialize configurations.
    ##
    class_type: type = ParticleCloth

    # init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg()
    # """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    # physics_material:RigidBodyMaterialCfg = None # type: ignore 
    # """Physics material. """




