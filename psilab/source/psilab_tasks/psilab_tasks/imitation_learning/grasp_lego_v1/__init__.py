import gymnasium as gym
from . import scenes


##
# Register Gym environments.
##

gym.register(
    id="Psi-TP-Grasp-Lego-v1",
    entry_point=f"{__name__}.grasp_lego_tp_env:GraspLegoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_lego_tp_env:GraspLegoEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)

gym.register(
    id="Psi-IL-Grasp-Lego-v1",
    entry_point=f"{__name__}.grasp_lego_il_env:GraspLegoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_lego_il_env:GraspLegoEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)

# Diffusion Policy 测试环境
gym.register(
    id="Psi-DP-Grasp-Lego-v1",
    entry_point=f"{__name__}.grasp_lego_dp_env:GraspLegoDPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_lego_dp_env:GraspLegoDPEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:PSI_DC_02_CFG",
    },
)