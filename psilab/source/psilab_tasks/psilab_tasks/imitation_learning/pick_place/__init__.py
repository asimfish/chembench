import gymnasium as gym
from . import scenes

# 导出配置工具函数（从上级目录的 config_loader 模块）
from ..config_loader import (
    load_grasp_config,
    load_handover_config,
    load_pick_place_config,
    load_pour_config,
    load_operation_config,
    get_object_names,
    get_supported_operations,
)


##
# Register Gym environments for Pick and Place tasks.
##

# Motion Planning (MP) - Pick and Place
gym.register(
    id="Psi-MP-PickPlace-v1",
    entry_point=f"{__name__}.pick_place_mp:PickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_mp:PickPlaceEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_PickPlace_CFG",
    },
)

# Imitation Learning (IL) - Pick and Place
gym.register(
    id="Psi-IL-PickPlace-v1",
    entry_point=f"{__name__}.pickplace_il:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_il:GraspBottleEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_PickPlace_CFG",
    },
)

# Reinforcement Learning (RL) - Pick and Place
gym.register(
    id="Psi-RL-PickPlace-v1",
    entry_point=f"{__name__}.pick_place_rl:DexPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_rl:DexPickPlaceEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_PickPlace_CFG",
    },
)
