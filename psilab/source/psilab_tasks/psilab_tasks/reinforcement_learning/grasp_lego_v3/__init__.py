import gymnasium as gym

from . import agents
from . import scenes

##
# Register Gym environments.
##

gym.register(
    id="Psi-RL-Grasp-Lego-v3",
    entry_point=f"{__name__}.grasp_lego_env:GraspLegoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_lego_env:GraspLegoEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.empty_cfg:PSI_DC_02_CFG",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)