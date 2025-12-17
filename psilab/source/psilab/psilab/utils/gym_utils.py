# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-17
# Vesion: 1.0
import copy

import importlib
import importlib.util
from typing import Any
import gymnasium

def load_env_creator(name: str):
    """Loads an environment with name of style ``"(import path):(environment name)"`` and returns the environment creation function, normally the environment class type.

    Args:
        name: The environment name

    Returns:
        The environment constructor for the given environment name.
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def make(
        env_id:str,
        **kwargs: Any,
        ):
        # The environment name can include an unloaded module in "module:env_name" style
    module, env_name = (None, env_id) if ":" not in env_id else env_id.split(":")
    if module is not None:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. Environment registration via importing a module failed. "
                f"Check whether '{module}' contains env registration and can be imported."
            ) from e
        
    # load the env spec from the registry
    env_spec = gymnasium.registry.get(env_name)

    # Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs.update(kwargs)
    #
    env_creator = load_env_creator(env_spec.entry_point)     # type: ignore

    try:
        env = env_creator(**env_spec_kwargs)
    except TypeError as e:
        raise type(e)(
                f"{e} was raised from the environment creator for {env_spec.id} with kwargs ({env_spec_kwargs})"
            )
    
    return env