# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-05
# Vesion: 1.0

"""
===============================================================================
Description:
    This script demonstrates how to use the ROS2 Bridge in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python ros2.py

===============================================================================
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates Ros2 demo")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)

import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension

#
import os
# os.environ['RMW_IMPLEMENTATION']="rmw_cyclonedds_cpp"
# os.environ['LD_LIBRARY_PATH']+=":/home/admin01/anaconda3/envs/psilab/lib/python3.10/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"

# enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.envs.common import ViewerCfg
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg

""" Psi Lab Modules  """ 
from psilab.scene.sence import Scene
from psilab.scene.sence_cfg import SceneCfg
from psilab import PSILAB_USD_ASSET_DIR
from psilab.random.random_cfg import RandomCfg

from psilab.assets.light.light_cfg import DomeLightCfg
from psilab.assets.robot.robot_base_cfg import RobotBaseCfg
from isaaclab.assets.articulation import ArticulationCfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(2.2,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 120, 
    render_interval=1,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        enable_ccd=True,
        gpu_found_lost_pairs_capacity = 137401003
    ),

    render=RenderCfg(),

)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=4.0, 
        replicate_physics=True,

        
        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0,0,2)
            )

        ),

        # local light
        local_lights_cfg={},

        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_02/PsiRobot_DC_02.usd",

                    activate_contact_sensors = True,

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
                        "arm1_joint_link1": -0.24,
                        "arm1_joint_link2": -0.64,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 0.94,
                        "arm2_joint_link1": 0.54,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.85681702,
                        "arm2_joint_link7": 1.871788888888889,
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
                        # "camera_chest_base_joint":0.0,
                        # "camera_head_base_joint":0.0,
                    }
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
                    # "camera": ImplicitActuatorCfg(
                    #     joint_names_expr=[
                    #         "camera_chest_base_joint",
                    #         "camera_head_base_joint"],
                    #     stiffness=None,
                    #     damping=None,

                    # ),
                },
                diff_ik_controllers = {},
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },
                cameras = {},
                tiled_cameras={},
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
        
        # rigid objects
        rigid_objects_cfg ={},
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={},
        
        # tiled camera sensor
        tiled_cameras_cfg={},
        
        # contact sensor
        contact_sensors_cfg={},

        # debug marker
        marker_cfg = None,
        
        # 
        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                
            },
            #


        ),
        
    )

# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(SCENE_CFG)

# joint names of robot
joint_names = [
    "arm1_joint_link1",
    "arm1_joint_link2",
    "arm1_joint_link3",
    "arm1_joint_link4",
    "arm1_joint_link5",
    "arm1_joint_link6",
    "arm1_joint_link7",
    "arm2_joint_link1",
    "arm2_joint_link2",
    "arm2_joint_link3",
    "arm2_joint_link4",
    "arm2_joint_link5",
    "arm2_joint_link6",
    "arm2_joint_link7",
    "hand1_joint_link_1_1",
    "hand1_joint_link_2_1",
    "hand1_joint_link_3_1",
    "hand1_joint_link_4_1",
    "hand1_joint_link_5_1",
    "hand1_joint_link_2_1",
    "hand2_joint_link_1_1",
    "hand2_joint_link_2_1",
    "hand2_joint_link_3_1",
    "hand2_joint_link_4_1",
    "hand2_joint_link_5_1",
    "hand2_joint_link_2_1",
]

# Creating a action graph with ROS component nodes
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ROS2Subscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("OgnIsaacRunOneSimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("ROS2PublishImage", "isaacsim.ros2.bridge.ROS2PublishImage"),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                #
                ("OnImpulseEvent.outputs:execOut", "ROS2Subscriber.inputs:execIn"),
                #
                ("Ros2Context.outputs:context", "ROS2Subscriber.inputs:context"),
                ("Ros2Context.outputs:context", "ROS2PublishImage.inputs:context"),
                ("Ros2Context.outputs:context", "ROS2CameraHelper.inputs:context"),
                #
                ("ROS2Subscriber.outputs:execOut", "ArticulationController.inputs:execIn"),

                ("OnImpulseEvent.outputs:execOut", "IsaacCreateRenderProduct.inputs:execIn"),
                ("IsaacCreateRenderProduct.outputs:execOut", "ROS2PublishImage.inputs:execIn"),
                ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
                ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),

            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:jointNames", joint_names),
                ("ROS2Subscriber.inputs:messageName", "Float64MultiArray"),
                ("ROS2Subscriber.inputs:messagePackage", "std_msgs"),
                ("ROS2Subscriber.inputs:messageSubfolder", "msg"),
                ("ROS2Subscriber.inputs:topicName", "ControlCmd"),
                ("ArticulationController.inputs:targetPrim", "/World/envs/env_0/Robot/joints/base_root_joint"),
                ("IsaacCreateRenderProduct.inputs:cameraPrim", "/World/envs/env_0/Robot/falan2/arm2_camera_rgb"),
                ("IsaacCreateRenderProduct.inputs:height", 480),
                ("IsaacCreateRenderProduct.inputs:width", 640),
                ("ROS2PublishImage.inputs:height", 480),
                ("ROS2PublishImage.inputs:width", 640),

            ],
        },
    )
except Exception as e:
    print(e)

# ros2 output should connect after og create, otherwise will cause error
og.Controller.connect("/ActionGraph/ROS2Subscriber.outputs:data", "/ActionGraph/ArticulationController.inputs:positionCommand")

#
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
# 
step = 0
step_max = 10000
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    
    # impluse
    og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)

    # 
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


