# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


from dataclasses import MISSING

from isaaclab.utils import configclass


from isaaclab.assets.articulation import Articulation
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.actuators import ActuatorBaseCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.sensors.camera import CameraCfg,Camera


from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.controllers.joint_impedance import JointImpedanceControllerCfg,JointImpedanceController


from .manipulation import Manipulation

@configclass
class ManipulationCfg(ArticulationCfg):
    """Configuration parameters for an robot(articulation) with multi arm and hand.""" 

    @configclass
    class InitialStateCfg(ArticulationCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # eef(effector) state
        eef_state: dict[str,list[float]] = {".*": list([0.0,0.0,0.0,0.0,0.0,0.0,0.0])}
        """State which include position(x,y,z) and orintation(w,x,y,z) of effector. Defaults to 0.0 for all effectors."""


    ##
    # Initialize configurations.
    ##
    class_type: type = Manipulation

    init_state: InitialStateCfg = InitialStateCfg()


    diff_ik_controllers: dict[str, DifferentialIKControllerCfg] = MISSING   # type: ignore
    """Differential IK Controllers Config for the Manipulation robot with corresponding joint group names."""

    cameras: dict[str, CameraCfg] = MISSING     # type: ignore
    """Cameras Config for the Manipulation robot with corresponding camera names."""





    # eef link name of ik, key: ik name, value: eef link name
    ik_eef_name: dict[str, str] = MISSING      # type: ignore
    # joint name of ik, key: ik name, value: list of joint names
    ik_joint_name: dict[str, list[str]] = MISSING      # type: ignore

    # effector name list
    eff_name: list[str]  = MISSING     # type: ignore
    # joint name of effector, key: ik name, value: list of joint names
    eff_joint_name: dict[str, list[str]] = MISSING     # type: ignore



    # ik or joint
    control_type: str = MISSING # type: ignore

    # joint map
    # key:real joint name
    # value vitual joint name
    real_virtual_joint_map: dict[str,dict[str,str]] = MISSING # type: ignore



