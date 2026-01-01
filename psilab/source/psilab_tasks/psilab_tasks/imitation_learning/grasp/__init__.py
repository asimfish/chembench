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
    id="Psi-TP-Grasp-Bottle-v1",
    entry_point=f"{__name__}.grasp_bottle_tp_env_from_lego:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_bottle_tp_env_from_lego:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)

gym.register(
    id="Psi-MP-Grasp-Bottle-v1",
    entry_point=f"{__name__}.grasp_bottle_mp_env:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_bottle_mp_env:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)


gym.register(
    id="Psi-IL-Grasp-Bottle-v1",
    entry_point=f"{__name__}.grasp_bottle_il_env:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_bottle_il_env:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)

gym.register(
    id="Psi-IL-Grasp-Bottle-Chempi-v1",
    entry_point=f"{__name__}.grasp_bottle_il_chempi_env:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_bottle_il_chempi_env:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)



gym.register(
    id="Psi-MP-Grasp-v1",
    entry_point=f"{__name__}.grasp_mp:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_mp:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Grasp_CFG",
    },
)


gym.register(
    id="Psi-Residual-RL-Grasp-v1",
    entry_point=f"{__name__}.grasp_residual_rl:GraspResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_residual_rl:GraspResidualRLEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Grasp_CFG",
    },
)

gym.register(
    id="Psi-MP-Grasp-Art-v1",
    entry_point=f"{__name__}.grasp_mp_art:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_mp_art:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Grasp_Art_CFG",
    },
)


gym.register(
    id="Psi-IL-Grasp-v1",
    entry_point=f"{__name__}.grasp_il:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_il:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Grasp_CFG",
    },
)

# gym.register(
#     id="Psi-MP-Grasp-Beaker-003-Art-v1",
#     entry_point=f"{__name__}.grasp_beaker_003_art_mp:GraspBottleEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.grasp_beaker_003_art_mp:GraspBottleEnvCfg",
#         "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Beaker_003_Art_CFG",
#     },
# )


gym.register(
    id="Psi-TP-Grasp-Beaker-003-v1",
    entry_point=f"{__name__}.grasp_beaker_003_tp:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_beaker_003_tp:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Beaker_003_CFG",
    },
)


gym.register(
    id="Psi-TP-Grasp-Chem-v1",
    entry_point=f"{__name__}.grasp_chem_tp:GraspBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_chem_tp:GraspBottleEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Chem_CFG",
    },
)


# 抓取参数查找器 - 用于通过遥操作找到合适的抓取位置和旋转
gym.register(
    id="Psi-TP-Grasp-Param-Finder-v1",
    entry_point=f"{__name__}.grasp_tp:GraspParamFinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_tp:GraspParamFinderEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_Grasp_CFG",
    },
)