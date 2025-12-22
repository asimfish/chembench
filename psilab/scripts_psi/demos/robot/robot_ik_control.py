# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to control a robot's ik in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python robot_ik_control.py

===============================================================================
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates random demo")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import numpy
import torch

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    SimulationContext,SimulationCfg,PhysxCfg,RenderCfg,
    RigidBodyPropertiesCfg,RigidBodyMaterialCfg)
from isaaclab.envs.common import ViewerCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import quat_from_euler_xyz,euler_xyz_from_quat

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets.robot import RobotBaseCfg
from psilab.assets.light import DomeLightCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(3.75,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=1,
    enable_scene_query_support=True,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        gpu_max_rigid_patch_count = 4096 * 4096,
    ),
    render=RenderCfg(),
)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 5, 
        env_spacing=2.5, 
        replicate_physics=True,

        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=100.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02.usd",
                    activate_contact_sensors = False,

                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "arm1_joint_link1": -0.36939895,
                        "arm1_joint_link2": -1.42726047,
                        "arm1_joint_link3": -0.32529447,
                        "arm1_joint_link4": -0.78829542,
                        "arm1_joint_link5": 1.78686804,
                        "arm1_joint_link6": 0.85681702,
                        "arm1_joint_link7": -2.33696087,
                        "arm2_joint_link1": 0.36939895,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.85681702,
                        "arm2_joint_link7": 2.33696087,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_1_2":0.63,
                        "hand1_joint_link_1_3":0.03,
                        "hand1_joint_link_2_1":3.10,
                        "hand1_joint_link_2_2":1.56,
                        "hand1_joint_link_3_1":3.06,
                        "hand1_joint_link_3_2":1.56,
                        "hand1_joint_link_4_1":3.06,
                        "hand1_joint_link_4_2":1.56,
                        "hand1_joint_link_5_1":3.04,
                        "hand1_joint_link_5_2":1.56,
                        "hand2_joint_link_1_1":0.0,
                        "hand2_joint_link_1_2":0.64,
                        "hand2_joint_link_1_3":0.03,
                        "hand2_joint_link_2_1":3.11,
                        "hand2_joint_link_2_2":1.56,
                        "hand2_joint_link_3_1":3.06,
                        "hand2_joint_link_3_2":1.56,
                        "hand2_joint_link_4_1":3.08,
                        "hand2_joint_link_4_2":1.56,
                        "hand2_joint_link_5_1":3.05,
                        "hand2_joint_link_5_2":1.56,
                    }
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0
                ),        
                actuators={
                    "arm1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,

                    ),
                    "arm2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,
                    ),
                    "hand1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand1_joint_link_1_1",
                            "hand1_joint_link_2_1",
                            "hand1_joint_link_3_1",
                            "hand1_joint_link_4_1",
                            "hand1_joint_link_5_1",
                            "hand1_joint_link_1_2",
                            "hand1_joint_link_2_2",
                            "hand1_joint_link_3_2",
                            "hand1_joint_link_4_2",
                            "hand1_joint_link_5_2",
                            "hand1_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                    "hand2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand2_joint_link_1_1",
                            "hand2_joint_link_2_1",
                            "hand2_joint_link_3_1",
                            "hand2_joint_link_4_1",
                            "hand2_joint_link_5_1",
                            "hand2_joint_link_1_2",
                            "hand2_joint_link_2_2",
                            "hand2_joint_link_3_2",
                            "hand2_joint_link_4_2",
                            "hand2_joint_link_5_2",
                            "hand2_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                },
                diff_ik_controllers = {
                    "arm1":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7"
                        ],
                        eef_link_name="arm1_link7"
                    ),
                    "arm2":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7"
                        ],
                        eef_link_name="arm2_link7"
                    ),
                },
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },
                tiled_cameras={
                    "camera_head_color": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera_head_base/camera_head_color",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                    "camera_chest_color": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera_chest_base/camera_chest_color",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                },
            )
        },

        # static object
        static_objects_cfg = {
            "ground" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Ground", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/grid/default_environment.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                )
            ),
        },
        
    )


# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(SCENE_CFG)
#
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
#
step = 0
step_max = 400

arm1_eef_index = scene.robots['robot'].find_bodies("arm1_link7")[0][0]
arm2_eef_index = scene.robots['robot'].find_bodies("arm2_link7")[0][0]

# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        #
        scene.reset()

        robot_pos = scene.robots["robot"].data.root_pos_w[:,:3]

        # get eef pose 
        arm1_eef_pose = scene.robots['robot'].data.body_link_state_w[:,arm1_eef_index,:][:,:7].clone()
        arm2_eef_pose = scene.robots['robot'].data.body_link_state_w[:,arm2_eef_index,:][:,:7].clone()
        #     torch.tensor([0.2,0.2,1.3],device=scene.device).unsqueeze(0).repeat(scene.cfg.num_envs,1)

        # random eef position offset
        arm1_eef_pos_offset = (2 * torch.rand((scene.cfg.num_envs,3),device=scene.device) - 1) * \
            torch.tensor([0.2,0.2,0.2],device=scene.device).unsqueeze(0).repeat(scene.cfg.num_envs,1)
        arm2_eef_pos_offset = (2 * torch.rand((scene.cfg.num_envs,3),device=scene.device) - 1) * \
            torch.tensor([0.2,0.2,0.2],device=scene.device).unsqueeze(0).repeat(scene.cfg.num_envs,1)
        
        # random eef angle offset
        arm1_eef_eular_offset = (2 * torch.rand((scene.cfg.num_envs,3),device=scene.device) - 1) * \
            torch.tensor([0.1,0.1,0.1],device=scene.device).unsqueeze(0).repeat(scene.cfg.num_envs,1)
        arm2_eef_eular_offset = (2 * torch.rand((scene.cfg.num_envs,3),device=scene.device) - 1) * \
            torch.tensor([0.1,0.1,0.1],device=scene.device).unsqueeze(0).repeat(scene.cfg.num_envs,1)
        

        arm1_eef_eular_x,arm1_eef_eular_y,arm1_eef_eular_z= euler_xyz_from_quat(arm1_eef_pose[:,3:])
        arm2_eef_eular_x,arm2_eef_eular_y,arm2_eef_eular_z = euler_xyz_from_quat(arm2_eef_pose[:,3:])
        # change position
        arm1_eef_pose[:,:3] = arm1_eef_pose[:,:3] + arm1_eef_pos_offset - robot_pos
        arm2_eef_pose[:,:3] = arm2_eef_pose[:,:3] + arm2_eef_pos_offset - robot_pos
        # change quat
        arm1_eef_pose[:,3:] = quat_from_euler_xyz(
            arm1_eef_eular_x + arm1_eef_eular_offset[:,0],
            arm1_eef_eular_y + arm1_eef_eular_offset[:,1],
            arm1_eef_eular_z + arm1_eef_eular_offset[:,2])
        arm2_eef_pose[:,3:] = quat_from_euler_xyz(
            arm2_eef_eular_x + arm2_eef_eular_offset[:,0],
            arm2_eef_eular_y + arm2_eef_eular_offset[:,1],
            arm2_eef_eular_z + arm2_eef_eular_offset[:,2])

        #
        scene.robots["robot"].set_ik_command({
                "arm1": arm1_eef_pose,
                "arm2": arm2_eef_pose,
            })

    # 
    scene.robots["robot"].ik_step()
    scene.robots["robot"].step()
    scene.robots["robot"].write_data_to_sim()

    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1