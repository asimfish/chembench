from .utils.transform_utils import farthest_point_sampling, transform_points, random_point, unnormalize_vars, normalize_vars, pose2mat, euler2quat
from .utils.sdf import get_distance_with_sdf, compute_sdf_from_obj, compute_sdf_from_obj_surface

import numpy as np


def cost_without_collision(obj_active, obj_passive, threshold=1, **kwargs):
    collision_points = obj_active.collision_points
    active2passive = np.linalg.inv(obj_passive.obj_pose) @ obj_active.obj_pose
    sdf_distance = get_distance_with_sdf(collision_points, active2passive, obj_passive.sdf) 
    
    cost = np.sum(sdf_distance<0)
    print('sdf:', cost, sdf_distance.min(), sdf_distance.max())
    return cost * 10


def cost_A_out_B(obj_active, obj_passive, safe_distance, **kwargs):
    activate_point = obj_active.obj_pose[:3, 3]
    passive_point = obj_passive.obj_pose[:3, 3]
    distance_vector = activate_point[:2] - passive_point[:2]
    distance = np.linalg.norm(distance_vector) 

    cost = 0
    if distance<(safe_distance):
        cost += 10
    return cost


def cost_A_on_B(obj_active, obj_passive, **kwargs):
    active_pts = transform_points(obj_active.anchor_points['buttom'], obj_active.obj_pose)[0]
    passive_pts = transform_points(obj_passive.anchor_points['top'], obj_passive.obj_pose)[0]

    # active_pts = transform_points(obj_active.anchor_points['buttom'], obj_active.obj_pose)[0]
    # passive_pts = transform_points(obj_passive.anchor_points['top'], obj_passive.obj_pose)[0]

    cost_add = 0
    # if activate_point[2] <= passive_point[2]:
    #     cost_add += 10  # Large penalty if the constraint is violated
    distance_vector = active_pts - passive_pts
    # import ipdb; ipdb.set_trace()
    distance_cost = np.linalg.norm(distance_vector)
    return cost_add + distance_cost


def cost_A_in_B(opt_pose_active_homo,opt_pose_passive_homo, selected_points_activate_sam,selected_points_passive_sam,selected_points_passive_bottom):

    # import pdb;pdb.set_trace()

    transformed_points_activate = batch_transform_points(selected_points_activate_sam, opt_pose_active_homo)
    transformed_points_activate = transformed_points_activate.reshape(-1, 3) 

    transformed_points_passive = selected_points_passive_sam
    transformed_points_passive = transformed_points_passive.reshape(-1, 3) 

    transformed_points_passive_bot = selected_points_passive_bottom
    transformed_points_passive_bot = transformed_points_passive_bot.reshape(-1, 3) 
    # Get the lowest point of the active object
    activate_point = random_point(transformed_points_activate,3)
    # Get the highest point of the passive object
    passive_point_top = random_point(transformed_points_passive,3)

    passive_point_bot = random_point(transformed_points_passive_bot,10)  # 约束靠近中心，避免求解出来碰撞
    cost_add = 0
    if activate_point[2] >= passive_point_top[2]:
        cost_add += 10  # Large penalty if the constraint is violated
    distance_vector = activate_point - passive_point_bot
    distance_cost = np.linalg.norm(distance_vector)
    return cost_add + distance_cost








CONSTRAINT_COST = {
    'wo_collision': cost_without_collision,
    "out": cost_A_out_B,
    "on": cost_A_on_B,
    "in": cost_A_in_B,
    "faceto": NotImplemented,
    'backto': NotImplemented
}

def get_cost_func(constraint):
    if constraint not in constraint:
        raise NotImplementedError("Constraint {} not implemented".format(constraint))
    return CONSTRAINT_COST[constraint]
