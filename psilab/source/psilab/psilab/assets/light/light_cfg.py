# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-08
# Vesion: 1.0


""" IsaacLab Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" IsaacLab Modules  """ 
from isaaclab.utils import configclass
from isaaclab.assets.asset_base_cfg import AssetBaseCfg


""" PsiLab Modules  """ 
from .light import Light

@configclass
class LightBaseCfg(AssetBaseCfg):
    """Configuration parameters for LightBase.""" 

    class_type: type = Light

    light_type: str = MISSING # type: ignore
    """The prim type name for the light prim."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The color of emitted light, in energy-linear terms. Defaults to white."""

    enable_color_temperature: bool = False
    """Enables color temperature. Defaults to false."""

    color_temperature: float = 6500.0
    """Color temperature (in Kelvin) representing the white point. The valid range is [1000, 10000]. Defaults to 6500K.

    The `color temperature <https://en.wikipedia.org/wiki/Color_temperature>`_ corresponds to the warmth
    or coolness of light. Warmer light has a lower color temperature, while cooler light has a higher
    color temperature.

    Note:
        It only takes effect when :attr:`enable_color_temperature` is true.
    """

    normalize: bool = False
    """Normalizes power by the surface area of the light. Defaults to false.

    This makes it easier to independently adjust the power and shape of the light, by causing the power
    to not vary with the area or angular size of the light.
    """

    exposure: float = 0.0
    """Scales the power of the light exponentially as a power of 2. Defaults to 0.0.

    The result is multiplied against the intensity.
    """

    intensity: float = 1.0
    """Scales the power of the light linearly. Defaults to 1.0."""

@configclass
class DiskLightCfg(LightBaseCfg):
    light_type = "DiskLight"

    radius: float = 0.5
    """Radius of the disk (in m). Defaults to 0.5m."""

@configclass
class DistantLightCfg(LightBaseCfg):
    
    light_type = "DistantLight"

    angle: float = 0.53
    """Angular size of the light (in degrees). Defaults to 0.53 degrees.

    As an example, the Sun is approximately 0.53 degrees as seen from Earth.
    Higher values broaden the light and therefore soften shadow edges.
    """

@configclass
class DomeLightCfg(LightBaseCfg):
    light_type = "DomeLight"

    texture_file: str | None = None
    """A color texture to use on the dome, such as an HDR (high dynamic range) texture intended
    for IBL (image based lighting). Defaults to None.
    If None, the dome will emit a uniform color.
    """

    texture_format: Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"] = "automatic"
    """The parametrization format of the color map file. Defaults to "automatic".

    Valid values are:

    * ``"automatic"``: Tries to determine the layout from the file itself. For example, Renderman texture files embed an explicit parameterization.
    * ``"latlong"``: Latitude as X, longitude as Y.
    * ``"mirroredBall"``: An image of the environment reflected in a sphere, using an implicitly orthogonal projection.
    * ``"angular"``: Similar to mirroredBall but the radial dimension is mapped linearly to the angle, providing better sampling at the edges.
    * ``"cubeMapVerticalCross"``: A cube map with faces laid out as a vertical cross.
    """

    visible_in_primary_ray: bool = True
    """Whether the dome light is visible in the primary ray. Defaults to True.

    If true, the texture in the sky is visible, otherwise the sky is black.
    """

@configclass
class CylinderLightCfg(LightBaseCfg):

    light_type = "CylinderLight"

    length: float = 1.0
    """Length of the cylinder (in m). Defaults to 1.0m."""

    radius: float = 0.5
    """Radius of the cylinder (in m). Defaults to 0.5m."""

    treat_as_line: bool = False
    """Treats the cylinder as a line source, i.e. a zero-radius cylinder. Defaults to false."""


@configclass
class SphereLightCfg(LightBaseCfg):
    prim_type = "SphereLight"

    radius: float = 0.5
    """Radius of the sphere. Defaults to 0.5m."""

    treat_as_point: bool = False
    """Treats the sphere as a point source, i.e. a zero-radius sphere. Defaults to false."""

@configclass
class RectLightCfg(LightBaseCfg):
    """Configuration parameters for creating a Rectangle light in the scene.
    """

    prim_type = "RectLight"

    height: float = 0.5

    width: float = 0.5