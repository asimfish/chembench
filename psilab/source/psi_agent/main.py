# -*- coding: utf-8 -*-
import os
import sys
import json
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_directory)
import argparse
from agent.omniagent import Agent
from layout.task_generate import TaskGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimGraspingAgent Command Line Interface")
 
    parser.add_argument(
        "--task_template",
        type=str,
        default="task/task.json",
        help="",
    )
    args = parser.parse_args()
    task_template_file = args.task_template
    with open(task_template_file, 'r') as file:
        task_info = json.load(file)
        task_generator = TaskGenerator(task_info)
        task_folder = 'saved_task/%s'%(task_info["task"])
        task_generator.generate_tasks(save_path=task_folder, task_num=task_info["recording_setting"]["num_of_episode"],task_name=task_info["task"])
        robot_position = task_generator.robot_init_pose["position"]
        robot_rotation = task_generator.robot_init_pose["quaternion"]
        print("generate job done")
