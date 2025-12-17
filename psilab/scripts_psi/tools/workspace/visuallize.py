
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
import csv
import torch
import pandas
import random

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.envs.common import ViewerCfg
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg,MassPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.materials import VisualMaterialCfg,PreviewSurfaceCfg

""" Psi Lab Modules  """ 
from psilab.scene.sence import Scene
from psilab.scene.sence_cfg import SceneCfg
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_URDF_ASSET_DIR
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg
from psilab.random.visual_material_random_cfg import VisualMaterialRandomCfg
from psilab.random.rigid_physic_material_random_cfg import RigidPhysicMaterialRandomCfg
from psilab.random.position_random_cfg import PositionRandomCfg
from psilab.random.orientation_random_cfg import OrientationRandomCfg
from psilab.random.robot_random_cfg import RobotRandomCfg
from psilab.assets.robot import RobotBaseCfg
from psilab.random.joint_random_cfg import JointRandomCfg

from psilab.assets.light import DomeLightCfg,SphereLightCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg

""" Curobo Modules  """ 
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig # type: ignore
from curobo.geom.sdf.world import CollisionCheckerType

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
        
        num_envs = 1, 
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

        # local light
        local_lights_cfg={},

        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
            prim_path = "/World/envs/env_[0-9]+/Robot",
            spawn = sim_utils.UsdFileCfg(
                usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_DC_02/Version_4.0/PsiRobot_DC_02.usd",
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                )
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0,0.0,0.0,0.0),
                joint_pos={
                    "arm1_joint_link1": -0.24,
                    "arm1_joint_link2": -1.73,
                    "arm1_joint_link3": -1.52,
                    "arm1_joint_link4": -0.81,
                    "arm1_joint_link5": 0.30,
                    "arm1_joint_link6": -1.03,
                    "arm1_joint_link7": 0.94,
                    "arm2_joint_link1": 0.24,
                    "arm2_joint_link2": -1.73,
                    "arm2_joint_link3": 1.52,
                    "arm2_joint_link4": -0.81,
                    "arm2_joint_link5": -0.30,
                    "arm2_joint_link6": -1.03,
                    "arm2_joint_link7": -0.94,
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
                # "arm2":DiffIKControllerCfg(
                #     command_type="pose", 
                #     use_relative_mode=False, 
                #     ik_method="dls",
                #     joint_name=[
                #         "arm2_joint_link1",
                #         "arm2_joint_link2",
                #         "arm2_joint_link3",
                #         "arm2_joint_link4",
                #         "arm2_joint_link5",
                #         "arm2_joint_link6",
                #         "arm2_joint_link7"
                #     ],
                #     eef_link_name="arm2_link7"
                # ),
            },
            eef_links={
                "arm1":"arm1_link7",
                # "arm2":"arm2_link7"
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
)

# get workspace 
csv_data = pandas.read_csv('scripts_psi/tools/workspace/psi_dc_02/psi_dc_02_ws.csv')
pose = torch.tensor(csv_data.iloc[:, 0:7].values,dtype=torch.float,device="cuda:0")
success = torch.tensor(csv_data.iloc[:, 7].values,dtype=torch.bool,device="cuda:0")
index_success = torch.nonzero(success==True).squeeze()


marker_pos = pose[:,:3]
marker_rot = pose[:,3:7]

marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/Markaers",
    markers={
        "fail":sim_utils.UsdFileCfg(
            usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/table/CubeTable.usd",
            scale=(0.001, 0.001, 0.01),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(1.0,0.0,0.0)
            )
        ),
        "success":sim_utils.UsdFileCfg(
            usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/low_poly/table/CubeTable.usd",
            scale=(0.001, 0.001, 0.01),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.0,1.0,0.0)
            )
        ),
    }
)

visualizer = VisualizationMarkers(marker_cfg)

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
step_max = 100
#
arm_joint_index = scene.robots["robot"].find_joints(scene.robots["robot"].actuators["arm1"].joint_names,preserve_order=True)[0]

######## Curobo config ########
tensor_args = TensorDeviceType()

# load curobo config
robot_cfg_yaml = load_yaml("scripts_psi/tools/workspace/psi_dc_02/psi_dc_02.yml")["robot_cfg"]
robot_cfg_yaml["kinematics"]["urdf_path"] = PSILAB_URDF_ASSET_DIR + robot_cfg_yaml["kinematics"]["urdf_path"]
#
robot_cfg = RobotConfig.from_basic(
    robot_cfg_yaml["kinematics"]["urdf_path"],
    robot_cfg_yaml["kinematics"]["base_link"], 
    robot_cfg_yaml["kinematics"]["ee_link"],
    tensor_args)

# create ik
ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        collision_checker_type=CollisionCheckerType.MESH,
        # collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        # use_fixed_samples=True,
    )
ik_solver = IKSolver(ik_config)

# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
        # get random pose
        index = random.randint(0,index_success.shape[0]-1)
        # pose_temp = pose[indexs_success[index],:7]

        # curobo ik
        position = pose[index_success[index],:3].repeat(1,1)
        quaternion = pose[index_success[index],3:].repeat(1,1)
        goal = Pose(position, quaternion)
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        scene.robots['robot'].write_joint_position_to_sim(result.js_solution.position.squeeze(1),joint_ids=arm_joint_index) # type: ignore
        scene.robots['robot'].data.joint_pos_target[:,arm_joint_index] = result.js_solution.position.squeeze(1) # type: ignore
        scene.robots['robot'].write_data_to_sim()
        # isaaclab ik
        # scene.robots["robot"].set_ik_command({"arm1": pose_temp})
        # scene.robots["robot"].step()

        pass
    
    if step < 20:
        visualizer.visualize(marker_pos, marker_rot,marker_indices=(success==False).int())
 
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


