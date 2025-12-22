# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-19
# Vesion: 1.0

import re
import os
import argparse
import subprocess
import logging
import yaml
import time
from datetime import datetime
from robot_settings import PSILAB_PATH,ROBOT_CONFIG


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_name",
        type=str,
        default=None,
        choices=[""],
        help="The name of the robot to analyze.",
    )
    parser.add_argument("--quiet", action="store_true", help="Don't print to console, only log to file.")
    parser.add_argument("--headless", action="store_true", help="Don't show the viewer.")
    parser.add_argument(
        "--log_path", type=str, default=os.path.join(PSILAB_PATH, "logs","robot_analysis",datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/results.log"), help="Path to the log file to store the results in."
    )
    return parser.parse_args()

def analyze(args: argparse.Namespace):

    # Create the log directory if it doesn't exist
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # get folder name from log_path
    log_dir = os.path.dirname(args.log_path)

    # Add file handler to log to file
    logging_handlers = [logging.FileHandler(args.log_path)]

    # We also want to print to console
    if not args.quiet:
        logging_handlers.append(logging.StreamHandler()) # type: ignore

    # Set up logger
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logging_handlers)

    if args.robot_name:
        robot_configs = {args.robot_name: ROBOT_CONFIG[args.robot_name]}
    else:
        robot_configs = ROBOT_CONFIG


    # analyze on each robot
    for name in robot_configs.keys():

        # log train info
        logging.info("\n" + "=" * 60 + "\n")
        #
        cmd = ["python",f"{PSILAB_PATH}/scripts_psi/tools/robot/robot_analysis.py"] \
                + [f"--robot_name={name}"] \
                + [f"--log_path={args.log_path}"]
        #
        if args.headless:
            cmd.append("--headless")
        # 
        subprocess.run(
            cmd,
            check=False,  # do not raise an error if the script fails
            capture_output=True, 
            text=True, 
        )

        # get result
        with open(f'{log_dir}/{name}/Result.log', "r") as f:
            result = f.read()
            #
            logging.info(result)

        time.sleep(10)
    # analysis finished
    logging.info("\n" + "=" * 60 + "\n")
       
if __name__ == "__main__":
    # parse command line arguments
    args = parse_args()

    # train test
    analyze(args)
