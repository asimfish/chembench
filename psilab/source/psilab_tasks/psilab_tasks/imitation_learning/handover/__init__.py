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
    id="Psi-MP-Handover-v1",
    entry_point=f"{__name__}.handover_mp:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_mp:HandoverEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_Handover_CFG",
    },
)

# Imitation Learning (IL) - Pick and Place
gym.register(
    id="Psi-IL-Handover-v1",
    entry_point=f"{__name__}.handover_il:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_il:HandoverEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_Handover_CFG",
    },
)

# Reinforcement Learning (RL) - Pick and Place
gym.register(
    id="Psi-RL-Handover-v1",
    entry_point=f"{__name__}.handover_rl:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_rl:HandoverEnvCfg",
        "scene_cfg_entry_point": f"{scenes.__name__}.room_cfg:PSI_DC_Handover_CFG",
    },
)
