# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-09
# Vesion: 1.0

"""
===============================================================================
Description:
    This script is used to demonstrate how to detect mesh overlap with omni graph node in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python mesh_overlap.py

===============================================================================
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates mesh overlap detect demo")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Omni Modules  """ 
import omni.graph.core as og

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext,SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets import DomeLightCfg,RobotBaseCfg
from psilab.random.random_cfg import RandomCfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(2.2,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=1,
    # mesh overlap only support cpu
    device="cpu",
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 64,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        enable_ccd=True,
        gpu_found_lost_pairs_capacity = 137401003
    ),
    enable_scene_query_support=True,
    use_fabric=False,
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


        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/Robot_Psi_DC_01/PsiRobot_DC_01.usd",

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
                        "arm2_joint_link1": 1.351944444,
                        "arm2_joint_link2": -1.620588848,
                        "arm2_joint_link3": 0,
                        "arm2_joint_link4": 0,
                        "arm2_joint_link5": 0,
                        "arm2_joint_link6": 0,
                        "arm2_joint_link7": 0,
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
            "table" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.0 ,1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=None,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8817000758263068, 0, 0.5)),
            ),
        },
        
        # rigid objects
        rigid_objects_cfg ={
            "target" :RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/table/CubeTable.usd",
                    activate_contact_sensors = True,
                    scale=(0.1, 0.1, 0.1),
                    visual_material=None,
                    mass_props=MassPropertiesCfg(
                        mass = 1
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        # solver_position_iteration_count=255,
                    )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7861376292693595, -0.22842012009466084, 1.006588856466109),
                    rot= (1,0,0,0)
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

# variables for omni graph
prims=[
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Robot/hand2_link_1_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_1_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_1_3/collisions",
    "/World/envs/env_0/Robot/hand2_link_2_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_2_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_3_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_3_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_4_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_4_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_5_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_5_2/collisions",
    
]

overlapsPair0 = [
    "/World/envs/env_0/Robot/hand2_link_1_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_1_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_1_3/collisions",
    "/World/envs/env_0/Robot/hand2_link_2_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_2_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_3_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_3_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_4_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_4_2/collisions",
    "/World/envs/env_0/Robot/hand2_link_5_1/collisions",
    "/World/envs/env_0/Robot/hand2_link_5_2/collisions",
]

overlapsPair1 = [
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
    "/World/envs/env_0/Target/base_link/mesh",
]

# Creating a action graph
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadPrimsV2", "omni.graph.nodes.ReadPrimsV2"),
                ("ImmediateComputeGeometryOverlaps", "omni.physx.graph.ImmediateComputeGeometryOverlaps"),

            ],
            og.Controller.Keys.CONNECT: [
                #
                ("OnImpulseEvent.outputs:execOut", "ImmediateComputeGeometryOverlaps.inputs:execIn"),
                #
                ("ReadPrimsV2.outputs_primsBundle", "ImmediateComputeGeometryOverlaps.inputs:primsBundle"),
                
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ReadPrimsV2.inputs:prims", prims),
                ("ImmediateComputeGeometryOverlaps.inputs:overlapsPair0", overlapsPair0),
                ("ImmediateComputeGeometryOverlaps.inputs:overlapsPair1", overlapsPair1),

            ],
        },
    )
except Exception as e:
    print(e)

# 
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
# 
step = 0
step_max = 100
#
overlap_num = 0
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    # control arm 
    if step > 20:
        joint_index = scene.robots["robot"].find_joints("arm2_joint_link6")[0]
        joint_pos = scene.robots["robot"].data.joint_pos[0,joint_index]

        joint_pos -= 1

        scene.robots["robot"].set_joint_position_target(joint_pos,joint_index)
        scene.robots["robot"].write_data_to_sim()
    # impluse
    og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)
    # get overlap result
    overlap_result = og.Controller.get(og.Controller.attribute("/ActionGraph/ImmediateComputeGeometryOverlaps.outputs:overlaps"))
    if True in list(overlap_result):
        overlap_num +=list(overlap_result).count(True)
        print(f"Overlap Counts: {overlap_num} \r", end="", flush=True)
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1



