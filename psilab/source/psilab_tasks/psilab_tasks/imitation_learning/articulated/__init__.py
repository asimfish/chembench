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
# Register Gym environments.
##


gym.register(
    id="Psi-MP-Articulated-v1",
    entry_point=f"{__name__}.articulated_mp:ArticulatedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.articulated_mp:ArticulatedEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Articulated_CFG",
    },
)


gym.register(
    id="Psi-IL-Articulated-Bottle-v1",
    entry_point=f"{__name__}.articulated_il:ArticulatedBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.articulated_il:ArticulatedBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Articulated_CFG",
    },
)


gym.register(
    id="Psi-Residual-RL-Articulated-v1",
    entry_point=f"{__name__}.articulated_residual_rl:ArticulatedResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.articulated_residual_rl:ArticulatedResidualRLEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Articulated_CFG",
    },
)





