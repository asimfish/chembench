# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


""" IsaacLab Modules  """ 
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation,ArticulationCfg
from isaaclab.sensors.camera import CameraCfg,Camera,TiledCameraCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg


""" PsiLab Modules  """ 
from psilab.assets.robot.robot_base import RobotBase
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg

@configclass
class RobotBaseCfg(ArticulationCfg):
    """Configuration parameters for an robot_base with cameras and ik controllers.""" 

    ##
    # Initialize configurations.
    ##
    class_type: type = RobotBase

    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    diff_ik_controllers: dict[str, DiffIKControllerCfg] = None   # type: ignore
    """Differential IK Controllers Config for the robot with corresponding joint group names."""

    cameras: dict[str, CameraCfg] = None     # type: ignore
    """Cameras Config for the robot with corresponding camera names."""

    tiled_cameras: dict[str, TiledCameraCfg] = None     # type: ignore
    """Tiled Cameras Config for the robot with corresponding camera names."""

    eef_links: dict[str, str] = None     # type: ignore
    """End of effactor links. """

    physics_material:RigidBodyMaterialCfg = None # type: ignore 
    """Physics material. """




