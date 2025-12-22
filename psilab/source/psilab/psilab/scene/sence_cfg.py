# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg,TiledCameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.assets import (
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)

from isaaclab.sim.spawners.lights.lights_cfg import (
    LightCfg
)

""" PsiLab Modules  """ 
from psilab.assets import RobotBaseCfg,ArticulatedObjectCfg,ParticleClothCfg
from psilab.assets.light.light_cfg import (
    LightBaseCfg,
    DiskLightCfg,
    SphereLightCfg,
    DomeLightCfg
)
from psilab.random.random_cfg import RandomCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for scene."""

    global_light_cfg:  None | DomeLightCfg  = None # type: ignore
    """The global light configuration."""

    local_lights_cfg :  None | dict[str, LightBaseCfg] = None # type: ignore
    """The local lights configuration dict."""

    robots_cfg :  None | dict[str, RobotBaseCfg] = None # type: ignore
    """The robots configuration dict."""

    static_objects_cfg :  None | dict[str, AssetBaseCfg] = None # type: ignore
    """The static objects configuration dict."""

    rigid_objects_cfg :  None | dict[str, RigidObjectCfg] = None # type: ignore
    """The rigid objects configuration dict."""

    articulated_objects_cfg :  None | dict[str, ArticulatedObjectCfg] = None # type: ignore
    """The articulated objects configuration dict."""

    particle_cloths_cfg :  None | dict[str, ParticleClothCfg] = None # type: ignore
    """The particle cloths configuration dict."""

    deformable_objects_cfg :  None | dict[str, DeformableObjectCfg] = None # type: ignore
    """The deformable objects configuration dict."""

    cameras_cfg:  None | dict[str, CameraCfg] = None  # type: ignore
    """The cameras configuration dict."""

    tiled_cameras_cfg:  None | dict[str, TiledCameraCfg] = None  # type: ignore
    """The tiled cameras configuration dict."""

    contact_sensors_cfg:  None | dict[str, ContactSensorCfg] = None  # type: ignore
    """The contact sensors configuration dict."""

    marker_cfg : None | VisualizationMarkersCfg = None  # type: ignore
    """The debug markers configuration."""

    random: None | RandomCfg = None   # type: ignore
    """The random configuration."""
