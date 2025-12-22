# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-05-10
# Vesion: 1.0

"""Sub-module with USD-related utilities."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
from isaacsim.core.cloner import Cloner
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade,Gf

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaaclab.utils.string import to_camel_case
from isaaclab.sim import utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import apply_nested

"""
Attribute - Setters.
"""

def safe_set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool):
    """Set the value of a attribute on its USD prim.

    This function is a wrapper around the isaac lab funtion  `safe_set_attribute_on_usd_prim`_.

    The function in isaac lab not support String, so if value is string, set attribute in this function, if value is not string, use funtion in isaac lab.

    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")
    # resolve sdf type based on value
    if isinstance(value, str):
        sdf_type = Sdf.ValueTypeNames.Bool
        # change property
        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=Sdf.Path(f"{prim.GetPath()}.{attr_name}"),
            value=value,
            prev=None,
            type_to_create_if_not_exist=sdf_type,
            usd_context_name=prim.GetStage(),
        )

    else:
        utils.safe_set_attribute_on_usd_prim(prim, attr_name, value, camel_case)

def set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool):
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")
    # resolve sdf type based on value
    # resolve sdf type based on value
    if isinstance(value, bool):
        sdf_type = Sdf.ValueTypeNames.Bool
    elif isinstance(value, int):
        sdf_type = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        sdf_type = Sdf.ValueTypeNames.Float
    elif isinstance(value, (tuple, list)) and len(value) == 3 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float3
    elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float2
    elif isinstance(value, str):
        if attr_name in ["inputs:diffuse_texture"]:
            sdf_type = Sdf.ValueTypeNames.Asset
        else:
            sdf_type = Sdf.ValueTypeNames.String
    else:
        raise NotImplementedError(
            f"Cannot set attribute '{attr_name}' with value '{value}'. Please modify the code to support this type."
        )
    # change property
    attribute = prim.GetAttribute(attr_name)
    if not attribute.IsValid():
        prim.CreateAttribute(attr_name, sdf_type)
        attribute = prim.GetAttribute(attr_name)
    attribute.Set(value) # type: ignore
    pass

def get_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, camel_case: bool):

    attribute = prim.GetAttribute(attr_name)

    if attribute.IsValid():
        value = attribute.Get()
    else:
        return None
    #
    # if isinstance(value, bool):
    #     sdf_type = Sdf.ValueTypeNames.Bool
    # elif isinstance(value, int):
    #     sdf_type = Sdf.ValueTypeNames.Int
    # elif isinstance(value, float):
    #     sdf_type = Sdf.ValueTypeNames.Float
    if isinstance(value, Gf.Vec2d) or isinstance(value, Gf.Vec2f):
        value = [value[0], value[1]]
    elif isinstance(value, Gf.Vec3d) or isinstance(value, Gf.Vec3f):
        value = [value[0], value[1], value[2]]
    # elif isinstance(value, str):
    #     sdf_type = Sdf.ValueTypeNames.Float3
    # elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
    #     sdf_type = Sdf.ValueTypeNames.Float2
    # elif isinstance(value, str):
    #     if attr_name in ["inputs:diffuse_texture"]:
    #         sdf_type = Sdf.ValueTypeNames.Asset
    #     else:
    #         sdf_type = Sdf.ValueTypeNames.String
    else:
        raise NotImplementedError(
            f"Cannot get attribute '{attr_name}' with unkown type. Please modify the code to support this type."
        )
    # attribute.Set(value) # type: ignore
    # convert attribute name to camel case
    return value