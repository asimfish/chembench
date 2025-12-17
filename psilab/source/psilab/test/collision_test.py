# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-21
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")



""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher



# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 
import torch
#
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import carb
import omni
from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    ArticulationRootPropertiesCfg,
    FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    DeformableBodyPropertiesCfg
)

from isaaclab.sim.spawners import (
    SpawnerCfg,
    DeformableObjectSpawnerCfg, 
    RigidObjectSpawnerCfg, 
    FileCfg,
    UsdFileCfg,
    UrdfFileCfg,
    GroundPlaneCfg,
    LightCfg,
    DiskLightCfg,
    DistantLightCfg,
    DomeLightCfg,
    CylinderLightCfg,
    SphereLightCfg,
    VisualMaterialCfg, 
    PreviewSurfaceCfg, 
    MdlFileCfg,
    GlassMdlCfg,
    PhysicsMaterialCfg, 
    RigidBodyMaterialCfg, 
    DeformableBodyMaterialCfg,
    PinholeCameraCfg, 
    FisheyeCameraCfg,
    MultiAssetSpawnerCfg, 
    MultiUsdFileCfg, 
)
from isaaclab.assets import (
    AssetBaseCfg,
    ArticulationCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
    DeformableObjectCfg
)
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.sim import PhysxCfg
from isaaclab.sim import RenderCfg

#
from psilab import PSILAB_USD_ASSET_DIR
from psilab.configs.robots.psi_dc_01 import PSI_DC_01_CFG
from psilab.configs.robots.psi_awh_01 import PSI_AWH_01_CFG

from psilab.assets.robot_base import RobotBase


# Initialize the simulation context
sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
sim_cfg.dt = 1/120
sim_cfg.physx = PhysxCfg(
    solver_type=1,
    enable_ccd=True,
    max_position_iteration_count=16,
    max_velocity_iteration_count=0,
    min_position_iteration_count=1,
    bounce_threshold_velocity = 0.02,

    gpu_found_lost_pairs_capacity = 137401003


)

# sim_cfg.render = RenderCfg(

# )
sim = sim_utils.SimulationContext(sim_cfg)

cfg = sim_utils.GroundPlaneCfg(
    usd_path = PSILAB_USD_ASSET_DIR + "/others/Grid/default_environment.usd",
    color=None)
cfg.func("/World/defaultGroundPlane", cfg)

# Lights
cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
cfg.func("/World/Light", cfg)

# table

cfg = RigidObjectCfg(
    prim_path="/World/Table", 
    spawn=sim_utils.UsdFileCfg(
        usd_path=PSILAB_USD_ASSET_DIR + "/others/table_cube.usd",
        scale=(1.5, 1.0, 0.6),
        visual_material=None,
        # rigid_props=RigidBodyPropertiesCfg(
        #     kinematic_enabled = True,
        # )
    ),
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        rot= (1.0, 0.0, 0.0, 0.0)
    )
)

cfg.spawn.func("/World/table", cfg.spawn,cfg.init_state.pos,cfg.init_state.rot)


# target
cfg = RigidObjectCfg(
    prim_path="/World/Target",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=PSILAB_USD_ASSET_DIR + "/others/cube.usd",
        # scale=(0.01,0.01,0.01),

        usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2.usd",
        scale=(3.0,3.0,3.0),

        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.80, 0.64, 0.20)
        ),
        mass_props=MassPropertiesCfg(
            mass = 1
        ),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=255,
            # max_linear_velocity=1.0,
            # max_angular_velocity=180,
        ),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.29,0,0.7),
        # rot= (1,0,0,0)
        rot= (0.707,0,0.707,0)

    )
)
# cfg.spawn.func("/World/target", cfg.spawn,cfg.init_state.pos,cfg.init_state.rot)
target = RigidObject(cfg)

# Create robot instance
robot_cfg = PSI_AWH_01_CFG.replace(prim_path="/World/Robot") # type: ignore
# robot_cfg.spawn.func("/World/Robot",robot_cfg.spawn,(-0.3, 0.0, 0.6),(1.0,0.0,0.0,0.0))
robot = RobotBase(cfg=robot_cfg)


# reset simulation before running
sim.reset()

# print()
joint_pos_target = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(
    joint_pos_target,
    torch.zeros(18,device="cuda:0"))
robot.set_joint_position_target(joint_pos_target)

target_defualt_state = target.data.root_com_state_w
bReset = False

def register_keyboard_handler():
    """
    Sets up the keyboard callback functionality with omniverse
    """
    appwindow = omni.appwindow.get_default_app_window() # type: ignore
    input_interface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

def keyboard_event_handler(event, *args, **kwargs):
    
    if (
        event.type == carb.input.KeyboardEventType.KEY_PRESS
        or event.type == carb.input.KeyboardEventType.KEY_REPEAT
    ):
        # Z键：重置
        global bReset
        if event.input == carb.input.KeyboardInput.Z:
            bReset = True

        
    # If we release a key, clear the active action and keypress
    # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        
        

    # Callback always needs to return True
    return True
    
register_keyboard_handler()

import time


# loop
while(True):
    if bReset:

        joint_pos_target = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(
            joint_pos_target,
            torch.zeros(18,device="cuda:0"))
        # robot.set_joint_position_target(joint_pos_target)
        # 
        target.write_root_state_to_sim(target_defualt_state)
        #
        bReset = False


    joint_pos_target[0,5]+=0.1 * sim.cfg.dt
    joint_pos_target[0,8:12]-=0.1 * sim.cfg.dt

    robot.set_joint_position_target(joint_pos_target)
    robot.update(sim.cfg.dt)
    robot.write_data_to_sim()
    start = time.time()

    sim.step()
    current = time.time()

    print(current-start)
