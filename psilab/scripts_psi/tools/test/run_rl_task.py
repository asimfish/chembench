# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-08-13
# Vesion: 1.0

import re
import os
import argparse
import subprocess
import logging
import yaml
import time
from datetime import datetime
from test_settings import PSILAB_PATH, RL_TRAIN_CONFIG,RL_PLAY_CONFIG


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib-name",
        type=str,
        default="rl_games",
        choices=["rl_games"],
        help="The name of the library to use for training.",
    )
    parser.add_argument("--quiet", action="store_true", help="Don't print to console, only log to file.")

    parser.add_argument(
        "--log_path", type=str, default=os.path.join(PSILAB_PATH, "logs","test_results",datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"), help="Path to the log file to store the results in."
    )
    return parser.parse_args()

def train_test(args: argparse.Namespace):

    # Create the log directory if it doesn't exist
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # get folder name from log_path
    log_folder = os.path.dirname(args.log_path)

    # Add file handler to log to file
    logging_handlers = [logging.FileHandler(args.log_path)]

    # We also want to print to console
    if not args.quiet:
        logging_handlers.append(logging.StreamHandler()) # type: ignore
    # Set up logger
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logging_handlers)

    # train on each environment
    for test_id in RL_TRAIN_CONFIG:
        # get train config
        train_config : list[str] = RL_TRAIN_CONFIG[test_id]

        # log train info
        logging.info("\n" + "=" * 60 + "\n")
        logging.info(f"Test id: {test_id}")
        logging.info(f"Framework: {args.lib_name}")
        logging.info(f"Args: {' '.join(train_config)}")

        # clear temp log before train
        if os.path.exists(log_folder + "/temp.log"):
            os.remove(log_folder + "/temp.log")

        # train
        while True:
            # run the train script
            try:
                subprocess.run(
                    [
                        "python",
                        f"{PSILAB_PATH}/scripts_psi/workflows/reinforcement_learning/{args.lib_name}/train.py"
                    ]
                    + train_config + [f"--log_path={log_folder}"],
                    check=False,  # do not raise an error if the script fails
                    capture_output=True, 
                    text=True, 
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Training failed with return code {e.returncode}")
                print(e.stderr)
                raise
            except Exception as e:
                # pass
                logging.error(f"Unexpected exception {e}.")
            
            #
            if not os.path.exists(log_folder + "/temp.log"):
                logging.error(f"Failed to get temp log. Train again...")
                continue

            # result
            with open(log_folder + "/temp.log", "r") as f:
                train_result = f.read()
                result = re.findall(r'Result folder:(.*)', train_result) # type: ignore
            # check if result is empty
            if len(result)==0:
                logging.error(f"Failed to get train logs folder. Train again...")
            else:
                train_log_folder = result[0]
                logging.info(f"Training results:{train_log_folder}")
                break
        
        #
        time.sleep(20)

        # get checkpoint
        agent_yaml = open(os.path.join(train_log_folder, "params", "agent.yaml"), 'r')
        agent = yaml.safe_load(agent_yaml)
        
        checkpoint_path = os.path.join(train_log_folder,"nn",agent["params"]["config"]["name"] + ".pth")
        # evaluate the checkpoint

        play_config = RL_PLAY_CONFIG[test_id]
        play_config[play_config.index("--checkpoint") +1] = checkpoint_path

        logging.info(f"Checkpoint: {checkpoint_path}")
        
        # clear temp log before play
        if os.path.exists(log_folder + "/temp.log"):
            os.remove(log_folder + "/temp.log")

        # play
        while True:
            # run the play script
            try:
                play_result = subprocess.run(
                    [
                        "python",
                        f"{PSILAB_PATH}/scripts_psi/workflows/reinforcement_learning/{args.lib_name}/play.py"
                    ]
                    + play_config + [f"--log_path={log_folder}"],
                    check=False,  # do not raise an error if the script fails
                    capture_output=True, 
                    text=True, 
                )
            except subprocess.CalledProcessError as e:
                logging.info(f"Eval failed with return code {e.returncode}")
                # print(e.stderr)
                raise
            
            #
            if not os.path.exists(log_folder + "/temp.log"):
                logging.error(f"Failed to get temp log. Play again...")
                continue

            # result
            with open(log_folder + "/temp.log", "r") as f:
                play_result = f.read()
                result = re.findall(r'Success rate:(.*)', play_result) # type: ignore
            # check if result is empty
            if len(result)==0:
                logging.error(f"Failed to get eval result. Play again...")
                continue
            else:
                success_rate = result[0]
                logging.info(f"Success rate: {success_rate}")
                break
        
        logging.info("\n" + "=" * 60 + "\n")

        time.sleep(20)

if __name__ == "__main__":
    # parse command line arguments
    args = parse_args()

    # train test
    train_test(args)
