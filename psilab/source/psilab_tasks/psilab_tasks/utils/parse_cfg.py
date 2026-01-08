# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-24
# Vesion: 1.0

""" Common Modules  """ 
import os
import json
import importlib
import gymnasium as gym

""" Isaac Lab Modules  """ 
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

""" Psi Modules  """ 
from psilab.utils.config_utils import scene_cfg
from psilab.scene.sence_cfg import SceneCfg
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.envs.tp_env_cfg import TPEnvCfg
from psilab.envs.il_env_cfg import ILEnvCfg

def parse_scene_cfg(
        task_name: str| None = None,
        enable_json: bool = False,
        scene_config: str | None = None,
        json_file: str | None = None,
        num_envs: int | None = None)->SceneCfg:

    # get scene config from json while "enable_json" is True
    if enable_json:
        # get scene config accordding to "scene_cfg_entry_point" while "scene_file" is None
        if json_file is None:
            # scene_cfg_entry_point = gym.spec(task_name).kwargs.get("scene_cfg_entry_point")
            # # resolve path to the scene config location
            # mod_name, file_name = scene_cfg_entry_point.split(":") # type: ignore
            scene_cfg_entry_point = gym.spec(task_name).kwargs.get("scene_cfg_entry_point") # type: ignore
            mod_name = scene_cfg_entry_point # type: ignore
            file_name = scene_name.split(":")[1] + ".json" # type: ignore
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__) # type: ignore
            json_file = os.path.join(mod_path, file_name)
        # get
        scene_json = open(json_file, 'r') # type: ignore
        scene_dict = json.loads(scene_json.read())
        scene = scene_cfg(scene_dict)
    else:
        scene_cfg_entry_point = gym.spec(task_name).kwargs.get("scene_cfg_entry_point") # type: ignore
        mod_name, attr_name = scene_cfg_entry_point.split(":") # type: ignore
        if scene_config is not None:
            mod_name = mod_name.rsplit(".",1)[0] + "." + scene_config.split(":")[0]
            attr_name = scene_config.split(":")[1]
        #
        mod = importlib.import_module(mod_name)
        scene = getattr(mod, attr_name)
    #
    scene.num_envs = num_envs # type: ignore
    #
    return scene


def parse_rl_env_cfg(
        env_cfg,
        seed: int | None = None,
        enable_wandb: bool = False,
        enable_eval: bool = False,
        enable_output: bool = False,
        output_folder: str | None = None,
        sample_step: int = 1,
        async_reset: bool = False,
        enable_random:bool = False,
        enable_marker:bool = False,
        task_id: str | None = None,
        scene_id: str | None = None,
        )->RLEnvCfg:
    # 
    env_cfg.seed = seed # type: ignore
    env_cfg.enable_wandb = enable_wandb # type: ignore
    env_cfg.enable_eval = enable_eval # type: ignore
    env_cfg.enable_output = enable_output # type: ignore
    env_cfg.sample_step = sample_step # type: ignore
    env_cfg.async_reset = async_reset # type: ignore
    env_cfg.enable_random = enable_random # type: ignore
    env_cfg.enable_marker = enable_marker # type: ignore

    if task_id is not None:
        env_cfg.task_id = task_id  # type: ignore
    if scene_id is not None:
        env_cfg.scene_id = scene_id  # type: ignore
    if output_folder is not None:
        env_cfg.output_folder = output_folder + "/rl"  # type: ignore

    #
    return env_cfg


def parse_il_env_cfg(
        env_cfg,
        seed: int | None = None,
        enable_wandb: bool = False,
        enable_output: bool = False,
        output_folder: str | None = None,
        sample_step: int = 1,
        async_reset: bool = False,
        enable_random:bool = False,
        enable_marker:bool = False,   
        enable_eval:bool = False,     
        checkpoint: str | None = None,
        max_step: int = 1, 
        max_episode: int = 1, 
        task_id: str | None = None,
        scene_id: str | None = None,
        # ACT Policy specific arguments
        obs_mode: str | None = None,
        camera_names: list[str] | None = None,
        num_queries: int | None = None,
        temporal_agg: str | None = None,
        mask_mode: str | None = None,
        )->ILEnvCfg:
    # 
    env_cfg.seed = seed # type: ignore
    env_cfg.enable_wandb = enable_wandb # type: ignore
    env_cfg.enable_output = enable_output # type: ignore
    env_cfg.sample_step = sample_step # type: ignore
    env_cfg.async_reset = async_reset # type: ignore
    env_cfg.enable_random = enable_random # type: ignore
    env_cfg.enable_marker = enable_marker # type: ignore
    env_cfg.checkpoint = checkpoint # type: ignore
    env_cfg.max_step = max_step # type: ignore
    env_cfg.max_episode = max_episode # type: ignore
    env_cfg.enable_eval = enable_eval # type: ignore

    if task_id is not None:
        env_cfg.task_id = task_id  # type: ignore
    if scene_id is not None:
        env_cfg.scene_id = scene_id  # type: ignore
    if output_folder is not None:
        env_cfg.output_folder = output_folder  # type: ignore
    
    # ACT specific configurations
    if obs_mode is not None:
        env_cfg.obs_mode = obs_mode  # type: ignore
    if camera_names is not None:
        env_cfg.camera_names = camera_names  # type: ignore
    if num_queries is not None:
        env_cfg.num_queries = num_queries  # type: ignore
    if temporal_agg is not None:
        # Convert string to boolean
        env_cfg.temporal_agg = (temporal_agg.lower() == 'true')  # type: ignore
    if mask_mode is not None:
        env_cfg.mask_mode = mask_mode  # type: ignore

    #
    return env_cfg



def parse_mp_env_cfg(
        env_cfg,
        seed: int | None = None,
        enable_output: bool = False,
        output_folder: str | None = None,
        sample_step: int = 1,
        async_reset: bool = False,
        enable_random:bool = False,
        enable_marker:bool = False, 
        enable_eval:bool = False,     
        max_step: int = 1, 
        max_episode: int = 1, 
        task_id: str | None = None,
        scene_id: str | None = None,
        )->ILEnvCfg:
    # 
    env_cfg.seed = seed # type: ignore
    env_cfg.enable_output = enable_output # type: ignore
    env_cfg.sample_step = sample_step # type: ignore
    env_cfg.async_reset = async_reset # type: ignore
    env_cfg.enable_random = enable_random # type: ignore
    env_cfg.enable_marker = enable_marker # type: ignore
    env_cfg.max_step = max_step # type: ignore
    env_cfg.max_episode = max_episode # type: ignore
    env_cfg.enable_eval = enable_eval # type: ignore

    if task_id is not None:
        env_cfg.task_id = task_id  # type: ignore
    if scene_id is not None:
        env_cfg.scene_id = scene_id  # type: ignore
    if output_folder is not None:
        env_cfg.output_folder = output_folder  # type: ignore

    #
    return env_cfg