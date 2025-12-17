# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-08-13
# Vesion: 1.0

"""
This file contains the settings for the tests.
"""
import os

PSILAB_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Path to the root directory of the Isaac Lab repository."""

RL_TRAIN_CONFIG = {
    # Grasp Lego v1
    "Psi-RL-Grasp-Lego-v1-Sync":[
        "--task","Psi-RL-Grasp-Lego-v1",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--max_epoch","15000",
        "--batch_size","2048"
    ],
    "Psi-RL-Grasp-Lego-v1-Async":[
        "--task","Psi-RL-Grasp-Lego-v1",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--async_reset",
        "--max_epoch","15000",
        "--batch_size","2048"
    ],
    # Grasp Lego v2 - PSI-DC-01
    "Psi-RL-Grasp-Lego-v2-Sync-PSI-DC-01":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--scene","empty_cfg:PSI_DC_01_CFG",
        "--headless",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--max_epoch","5000",
        "--batch_size","2048"
    ],
    "Psi-RL-Grasp-Lego-v2-Async-PSI-DC-01":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--scene","empty_cfg:PSI_DC_01_CFG",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--async_reset",
        "--max_epoch","5000",
        "--batch_size","2048"
    ],
    # Grasp Lego v2 - PSI-DC-02
    "Psi-RL-Grasp-Lego-v2-Sync-PSI-DC-02":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--headless",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--max_epoch","5000",
        "--batch_size","2048"
    ],
    "Psi-RL-Grasp-Lego-v2-Async-PSI-DC-02":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--async_reset",
        "--max_epoch","5000",
        "--batch_size","2048"
    ],
    # Grasp Lego v3 - PsiSyncHand
    "Psi-RL-Grasp-Lego-v3-Async-PSI-DC-02":[
        "--task","Psi-RL-Grasp-Lego-v3",
        "--num_envs","1024",
        "--seed","17",
        "--headless",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--async_reset",
        "--max_epoch","5000",
        "--batch_size","1024"
    ],
    # Open-Door-v1 - Psi-DC-02
    "Psi-RL-Open-Door-v1-Async-PSI-DC-02":[
        "--task","Psi-RL-Open-Door-v1",
        "--num_envs","1024",
        "--seed","17",
        "--headless",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--enable_random",
        "--enable_wandb",
        "--enable_marker",
        "--async_reset",
        "--max_epoch","5000",
        "--batch_size","1024"
    ],

}
"""A list of RL environments to test task train """

RL_PLAY_CONFIG = {
    # Grasp Lego v1
    "Psi-RL-Grasp-Lego-v1-Sync":[
        "--task","Psi-RL-Grasp-Lego-v1",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v1/Psi-RL-Grasp-Lego-v1-Sync/nn/grasp_lego_v1.pth",
        "--enable_eval",
        "--enable_random",
        "--play_times","10"
    ],
    "Psi-RL-Grasp-Lego-v1-Async":[
        "--task","Psi-RL-Grasp-Lego-v1",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v1/Psi-RL-Grasp-Lego-v1-Async/nn/grasp_lego_v1.pth",
        "--enable_eval",
        "--enable_random",
        "--async_reset",
        "--play_times","10"
    ],
    # Grasp Lego v2 - PSI-DC-01
    "Psi-RL-Grasp-Lego-v2-Sync":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v2/Psi-RL-Grasp-Lego-v2-Sync/nn/grasp_lego_v2.pth",
        "--scene","empty_cfg:PSI_DC_01_CFG",
        "--enable_eval",
        "--enable_random",
        "--play_times","10"
    ],
    "Psi-RL-Grasp-Lego-v2-Async":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v2/Psi-RL-Grasp-Lego-v2-Async/nn/grasp_lego_v2.pth",
        "--scene","empty_cfg:PSI_DC_01_CFG",
        "--enable_eval",
        "--enable_random",
        "--async_reset",
        "--play_times","10"
    ],
    # Grasp Lego v2 -PSI_DC_02
    "Psi-RL-Grasp-Lego-v2-Sync":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v2/Psi-RL-Grasp-Lego-v2-Sync/nn/grasp_lego_v2.pth",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--enable_eval",
        "--enable_random",
        "--play_times","10"
    ],
    "Psi-RL-Grasp-Lego-v2-Async":[
        "--task","Psi-RL-Grasp-Lego-v2",
        "--num_envs","2048",
        "--seed","17",
        "--headless",
        "--checkpoint",PSILAB_PATH+"/logs/rl_games/grasp_lego_v2/Psi-RL-Grasp-Lego-v2-Async/nn/grasp_lego_v2.pth",
        "--scene","empty_cfg:PSI_DC_02_CFG",
        "--enable_eval",
        "--enable_random",
        "--async_reset",
        "--play_times","10"
    ]
    # Grasp Lego v3 - PsiSyncHand

    # Open-Door-v1 - Psi-DC-02

}
"""A list of RL environments to test task play """
