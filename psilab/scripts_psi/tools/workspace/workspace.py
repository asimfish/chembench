
import numpy
import torch
import csv
#

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig # type: ignore
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


from psilab import PSILAB_URDF_ASSET_DIR

@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)

def get_position_grid(n_x, n_y, n_z, max_x, max_y, max_z):
    x = numpy.linspace(-max_x, max_x, n_x)
    y = numpy.linspace(-max_y, max_y, n_y)
    z = numpy.linspace(0, max_z, n_z)
    x, y, z = numpy.meshgrid(x, y, z, indexing="ij")

    position_arr = numpy.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()
    return position_arr

def get_orientation_grid(n_roll,n_pitch,n_yaw, max_roll, max_pitch, max_yaw):

    x = numpy.linspace(-max_roll, max_roll, n_roll)
    y = numpy.linspace(-max_pitch, max_pitch, n_pitch)
    z = numpy.linspace(-max_yaw, max_yaw, n_yaw)
    x, y, z = numpy.meshgrid(x, y, z, indexing="ij")

    orientation_arr = numpy.zeros((n_roll * n_pitch * n_yaw, 3))
    orientation_arr[:, 0] = x.flatten()
    orientation_arr[:, 1] = y.flatten()
    orientation_arr[:, 2] = z.flatten()


    return orientation_arr


tensor_args = TensorDeviceType()

# load curobo config
robot_cfg_yaml = load_yaml("scripts_psi/tools/workspace/psi_dc_02/psi_dc_02.yml")["robot_cfg"]
robot_cfg_yaml["kinematics"]["urdf_path"] = PSILAB_URDF_ASSET_DIR + robot_cfg_yaml["kinematics"]["urdf_path"]

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

# 
position_goal = tensor_args.to_device(get_position_grid(20, 20, 20, 1, 1, 2))
orientation_eular_goal = tensor_args.to_device(get_orientation_grid(10, 10, 10, 3.14,3.14,3.14))
orientation_goal = quat_from_euler_xyz(orientation_eular_goal[:,0],orientation_eular_goal[:,1],orientation_eular_goal[:,2])

batch_size = 500

print("Curobo is Ready")

for i in range(position_goal.shape[0]):
    csv_file = open(file="scripts_psi/tools/workspace/psi_dc_02/psi_dc_02_ws.csv",mode="a")
    csv_writer = csv.writer(csv_file)

    print(
        f"pos {position_goal[i,0].cpu()} {position_goal[i,1].cpu()} {position_goal[i,2].cpu()}   "
    )
    b_pos_reached = False

    pos_result = []
    for j in range(int(orientation_goal.shape[0] / batch_size)):
        position = position_goal[i,:].unsqueeze(0).repeat(batch_size,1)
        quaternion = orientation_goal[batch_size * j:batch_size * (j+1),:]

        goal = Pose(position, quaternion)
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        # print(
        #     f"pos {position_goal[i,0].cpu()} {position_goal[i,1].cpu()} {position_goal[i,2].cpu()} , pos_error: {torch.mean(result.position_error)}, ro_error: {torch.mean(result.rotation_error)} "
        # )

        b_pos_reached = b_pos_reached or torch.any(result.success)
 
        succ_index = torch.nonzero(result.success[:,0]==True).squeeze()
        for index in range(result.success.shape[0]):
            
            pos_cpu = position_goal[i,:].cpu().numpy()
            ori_cpu = orientation_goal[batch_size * j + index,:].cpu().numpy()

            pos_result.append([
                # position
                pos_cpu[0],
                pos_cpu[1],
                pos_cpu[2], 
                # ori
                ori_cpu[0],
                ori_cpu[1],
                ori_cpu[2],
                ori_cpu[3],
                #
                True if index in succ_index else False
            ])
            
    
    if b_pos_reached:
        for data in pos_result:
            csv_writer.writerow([
                # position
                data[0],
                data[1],
                data[2], 
                # ori
                data[3],
                data[4],
                data[5],
                data[6],
                #
                data[7]
            ])

    csv_file.close()
