# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--width", type=int, default=1920, help="Width of the viewport and generated images.")
parser.add_argument("--height", type=int, default=1080, help="Height of the viewport and generated images.")


# add argparse arguments from Psi
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Whether update data to wandb or not.")
parser.add_argument("--enable_json", action="store_true", default=False, help="Whether create scene from json or not.")
parser.add_argument("--scene", type=str, default=None, help="Scene.")
parser.add_argument("--json_file", type=str, default=None, help="Scene json file.")
parser.add_argument("--enable_output", action="store_true", default=False, help="Whether output data to hdf5 files or not.")
parser.add_argument("--enable_eval", action="store_true", default=False, help="Enable print evalutation result or not.")
parser.add_argument("--output_folder", type=str, default=None, help="Hdf5 files folder.")
parser.add_argument("--sample_step", type=int, default=1, help="Simulation steps per sample step")
parser.add_argument("--async_reset", action="store_true", default=False, help="Whether reset envs asynchronous or asynchronous.")
parser.add_argument("--enable_random", action="store_true", default=False, help="Whether enbale random when envs reset.")
parser.add_argument("--enable_marker", action="store_true", default=False, help="Whether show marker or not.")

parser.add_argument("--max_epoch", type=int, default=100000, help="max_epoch in rl games ppo config")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size in rl games ppo config")
parser.add_argument("--log_path", type=str, default=None, help="Path to the log file.")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
    
# parse the arguments
args_cli = parser.parse_args()

# store args befor create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import os
import sys
import math
import importlib
import json
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from datetime import datetime

""" Isaac Lab Modules  """ 
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab.utils.io import dump_pickle, dump_yaml

""" Psi Modules  """ 
# import psilab.tasks # noqa: F401
import psilab_tasks
from psilab_tasks.utils import parse_scene_cfg,parse_rl_env_cfg

def train(args_cli:argparse.Namespace):

    # parse argumanets for isaac lab rl env config
    env_cfg= parse_env_cfg(
        args_cli.task, 
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        # use_fabric=not args_cli.disable_fabric
        )

    # parse argumanets for psi lab rl env config
    env_cfg = parse_rl_env_cfg(
        env_cfg,
        args_cli.seed,
        args_cli.enable_wandb,
        args_cli.enable_eval,
        args_cli.enable_output,
        args_cli.output_folder,
        args_cli.sample_step,
        args_cli.async_reset,
        args_cli.enable_random,
        args_cli.enable_marker
    )

    # parse argumanets for psi lab scene config
    env_cfg.scene = parse_scene_cfg(
        args_cli.task, 
        args_cli.enable_json,
        args_cli.scene,
        args_cli.json_file,
        args_cli.num_envs,
    )

    # clear camera configs in scene while "enable_cameras" flag is True
    if enable_cameras is False:
        env_cfg.scene.cameras_cfg ={}
        env_cfg.scene.tiled_cameras_cfg = {}
        for robot_cfg in env_cfg.scene.robots_cfg.values():
            robot_cfg.cameras = {} # type: ignore
            robot_cfg.tiled_cameras = {} # type: ignore

    # clear random configs while "enable_random" flag is false
    if not env_cfg.enable_random:
        env_cfg.scene.random = None

    # clear marker configs while "enable_marker" flag is false
    if not env_cfg.enable_marker:
        env_cfg.scene.marker_cfg = None

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg)

    # parse agent configuration
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.max_epoch is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_epoch # type: ignore
    if args_cli.batch_size is not None:
        agent_cfg["params"]["config"]["minibatch_size"] = args_cli.batch_size # type: ignore
        agent_cfg["params"]["config"]["central_value_config"]["minibatch_size"] = args_cli.batch_size # type: ignore

    agent_cfg["params"]["seed"] = args_cli.seed # type: ignore

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]) # type: ignore
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))# type: ignore
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path # type: ignore
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir # type: ignore

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"] # type: ignore
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf) # type: ignore
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf) # type: ignore

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions) # type: ignore


    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs # type: ignore

    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()

    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": None})
    else:
        runner.run({"train": True, "play": False, "sigma": None})

    # close the simulator
    env.close()
    
    # write result to log file
    if args_cli.log_path is not None:
        with open(args_cli.log_path + "/temp.log", "w") as f:
            f.write(f"Result folder:{os.path.join(log_root_path, log_dir)}")
        f.close()


if __name__ == "__main__":
    # run the main function
    train(args_cli)
    # close sim app 
    # simulation_app.close()