# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random import PositionRandomCfg,OrientationRandomCfg
from psilab.random.rigid_physic_material_random_cfg import RigidPhysicMaterialRandomCfg
from psilab.random.visual_material_random_cfg import VisualMaterialRandomCfg
from psilab.random.mass_random_cfg import MassRandomCfg

@configclass
class RigidRandomCfg():
    """Configuration for rigid random options."""

    mass: MassRandomCfg | None = None
    """The mass random config.
    Tips: 
    - SceneCfg::replicate_physics should be False while mass random is True, otherwise will cause error.
    """
    
    position: PositionRandomCfg | None = None

    orientation: OrientationRandomCfg | None = None
    
    visual_material : VisualMaterialRandomCfg | None  =  None # type: ignore
    """The material random config."""

    physics_material : RigidPhysicMaterialRandomCfg | None =  None # type: ignore
    """The physics material random config."""
    