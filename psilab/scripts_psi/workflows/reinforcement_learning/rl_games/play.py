# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_false",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

# add argparse arguments from Psi
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
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
parser.add_argument("--play_times", type=int, default=1, help="Times to play.")

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
simulation_app = app_launcher.app

""" Common Modules  """ 
import os
import gymnasium as gym
import math
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

""" Isaac Lab Modules  """ 
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

""" Psi RL Modules  """ 
import psilab_tasks
from psilab_tasks.utils import parse_scene_cfg,parse_rl_env_cfg

def play(args_cli:argparse.Namespace):

    """Play with RL-Games agent."""
    # parse argumanets for isaac lab rl env config
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        # use_fabric=not args_cli.disable_fabric
    )

    if args_cli.scene is None:
        args_cli.scene = gym.spec(args_cli.task).kwargs.get("scene_cfg_entry_point").split(".")[-1] # type: ignore

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
        args_cli.enable_marker,
        args_cli.task,
        args_cli.scene
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

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]) # type: ignore
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*") # type: ignore
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"    # type: ignore
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"] # type: ignore
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)    # type: ignore
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf) # type: ignore

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # type: ignore

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)  # type: ignore

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True   # type: ignore
    agent_cfg["params"]["load_path"] = resume_path  # type: ignore
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}") # type: ignore

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs    # type: ignore
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:    # type: ignore
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running() and env.env.env.episode_num < args_cli.play_times * args_cli.num_envs: # type: ignore
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:   # type: ignore
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

    print("#" * 17, " Statistics", "#" * 17)
    print(f"运行次数: {env.env.env.episode_num} 次数") # type: ignore
    print(f"成功率: {env.env.env.episode_success_num}/{env.env.env.episode_num} 次") # type: ignore
    print("#" * 17, " Statistics", "#" * 17)

    # return
    success_rate = env.env.env.episode_success_num / env.env.env.episode_num # type: ignore

    print(args_cli.log_path)
    # write result to log file
    if args_cli.log_path is not None:
        with open(args_cli.log_path + "/temp.log", "w") as f:
            f.write(f"Success rate:{success_rate}")
        f.close()

if __name__ == "__main__":
    # run the main function
    play(args_cli)
    # close sim app 
    # simulation_app.close()
