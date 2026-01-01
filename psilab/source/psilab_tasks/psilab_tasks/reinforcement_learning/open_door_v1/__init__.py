import gymnasium as gym

from . import agents
from . import scenes

##
# Register Gym environments.
##

gym.register(
    id="Psi-RL-Open-Door-v1",
    entry_point=f"{__name__}.open_door_env:OpenDoorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_door_env:OpenDoorEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.empty_cfg:PSI_DC_01_CFG",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)