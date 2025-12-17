from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.optimize import dual_annealing
import open3d as o3d
import numpy as np
import time
import yaml
import os


from .utils.transform_utils import farthest_point_sampling, transform_points, random_point, unnormalize_vars, normalize_vars, pose2mat, euler2quat
from .cost_function import get_cost_func
np.random.seed(0)




# align
def generate_random_pose(mesh, plane_size):
    bounding_box = mesh.bounds
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]

    object_width = max_corner[0] - min_corner[0]
    object_height = max_corner[1] - min_corner[1]
    position_range_x = (plane_size[0] / 2) - (object_width / 2)
    position_range_y = (plane_size[1] / 2) - (object_height / 2)
    position_x = np.random.uniform(-position_range_x, position_range_x)
    position_y = np.random.uniform(-position_range_y, position_range_y)
    yaw = np.random.uniform(0, 360)
    
    rotation = R.from_euler('z', yaw, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [position_x, position_y, -min_corner[2] + 0.002]

    return pose




def passive_setup_sdf(bounding_box_range, sdf_voxels):
    # create callable sdf function with interpolation
    min_corner = bounding_box_range[0]
    max_corner = bounding_box_range[1]

    x = np.linspace(min_corner[0], max_corner[0], sdf_voxels.shape[0])
    y = np.linspace(min_corner[1], max_corner[1], sdf_voxels.shape[1])
    z = np.linspace(min_corner[2], max_corner[2], sdf_voxels.shape[2])
    sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
    return sdf_func

def passive_setup_sdf_all(bounds_max,bounds_min, sdf_voxels):
    x = np.linspace(bounds_min[0], bounds_max[0], sdf_voxels.shape[0])
    y = np.linspace(bounds_min[1], bounds_max[1], sdf_voxels.shape[1])
    z = np.linspace(bounds_min[2], bounds_max[2], sdf_voxels.shape[2])
    sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
    return sdf_func


from .utils.object import LayoutObject



class LayoutSolver3D:
    def __init__(self, workspace_xyz, workspace_size, objects=None, obj_infos=None):
        x_half, y_half, z_half = workspace_size / 2
        bounds = {"min_bound":[-x_half, -y_half, 0.1], "max_bound":[x_half, y_half, z_half]} 
        self.bounds_min = bounds["min_bound"]
        self.bounds_max = bounds["max_bound"]
        self.workspace_size = np.array(bounds["max_bound"])- np.array(bounds["min_bound"])


        self.objects = objects


        


    def init_pose(self, opt_obj_ids, constraint):
        active_obj_id, passive_obj_id, constraint = constraint['active'], constraint['passive'], constraint['constraint']
        obj_active = self.objects[active_obj_id]
        obj_passive = self.objects[passive_obj_id]
        
        print("init_poseinit_poseinit_poseinit_poseinit_poseinit_poseinit_poseinit_pose")
        print(active_obj_id)
        print(passive_obj_id)


        # if 'out' in constraints or 'on' in constraints or 'in' in constraints:
        # pose_passive = generate_random_pose(obj_passive.mesh, self.workspace_size)
        #     self.update_obj(passive_obj_id, pose_passive)

        pose_active = generate_random_pose(obj_active.mesh, self.workspace_size)
        self.update_obj(active_obj_id, pose_active)


        self.safe_distance = np.mean(obj_active.size[:2]) + np.mean(obj_passive.size[:2]) / 2 + 5 



        optimize_z = constraint=='on'
        if optimize_z:
            pose_bounds_min_ac = np.append(self.bounds_min[:2], obj_passive.obj_pose[2, 3] + obj_passive.size[2] / 2.0)
            pose_bounds_max_ac = np.append(self.bounds_max[:2], self.bounds_max[2])
        else:
            pose_bounds_min_ac = np.append(self.bounds_min[:2], pose_active[2, 3])
            pose_bounds_max_ac = np.append(self.bounds_max[:2],  pose_active[2, 3])

        # Define bounds for optimization
        rot_bounds_min = np.array([0])
        rot_bounds_max = np.array([np.radians(360)])
        self.og_bounds =  [(b_min, b_max) for b_min, b_max in zip(
            np.concatenate([pose_bounds_min_ac, rot_bounds_min]),
            np.concatenate([pose_bounds_max_ac, rot_bounds_max])
        )]
        self.norm_bounds = np.array([(-1, 1)] * len(self.og_bounds))


        # Initialize optimization
        translation_active = pose_active[:3, 3]
        rotation_active = R.from_matrix(pose_active[:3, :3])
        euler_angles_active = rotation_active.as_euler('xyz')
        st_pose_euler_active = np.append(translation_active, euler_angles_active[2])
        init_x = normalize_vars(st_pose_euler_active, self.og_bounds)
        return init_x




    def update_obj(self, obj_id, pose):
        self.objects[obj_id].obj_pose = pose
        
    
    def cost_function(self, opt_vars, opt_ids, constraints):
        # Update obj pose
        for i in range(len(opt_ids)):
            obj_id = opt_ids[i]
            xyz_yaw = unnormalize_vars(opt_vars[i*4:i*4+4], self.og_bounds)
            
            pose = pose2mat([xyz_yaw[:3], euler2quat(np.append([0,0], xyz_yaw[3]))])
            self.update_obj(obj_id, pose)

        # Calculate cost for each constraint
        cost = 0
        for info  in constraints:
            active_id, passive_id, constraint = info['active'], info['passive'], info['constraint']
            cost += get_cost_func(constraint)(
                self.objects[active_id],
                self.objects[passive_id],
                safe_distance = self.safe_distance
            )
        return cost



    def __call__(self, opt_obj_ids, exist_obj_ids, constraint, visual=False):
        x0 = self.init_pose(opt_obj_ids, constraint)

        constraints = [constraint]
        constraints.append({'active': constraint['active'], 'passive': constraint['passive'], 'constraint': 'out'})



        saved_solutions={}
        # Optimization 
        opt_result = dual_annealing(
            func = self.cost_function,
            args = [opt_obj_ids, constraints],
            bounds = self.norm_bounds,
            maxfun = 10000,
            x0 = x0,
            no_local_search = True,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'maxiter': 200},
            },
            restart_temp_ratio=2e-5  # 
        )



        if opt_result.success:
            print("Optimization successful.")
            return opt_obj_ids
        else:
            print("Optimization failed:", opt_result.message)
            saved_solutions = None
        
        return saved_solutions






