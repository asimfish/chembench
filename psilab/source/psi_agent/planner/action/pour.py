from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference, interpolate_rotation_matrices

import copy
import numpy as np





class PourStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=None, current_pose=None, obj2part=None, **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        
        self.pre_pour_height = 0.25
        self.pour_heigth = 0.08
        self.generate_substage(target_pose, current_pose, obj2part)
        

    def generate_substage(self, target_pose, current_pose, obj2part):
        target_part_pose = target_pose @ np.linalg.inv(obj2part)
        current_part_pose = current_pose @ np.linalg.inv(obj2part)

        gripper_action = None
        transform_pre_pour = np.eye(4)
        transform_pre_pour[2,3] = self.pre_pour_height    # 8cm above the target pose


        # moveTo pre-pour position         
        pre_pour_part_pose = np.copy(target_part_pose)
        pre_pour_part_pose[:3, :3] = current_part_pose[:3, :3]
        # # pre_pour at 2/3 position from current to target
        # pos_diff = target_part_pose[:3, 3] - current_part_pose[:3, 3]
        # pre_pour_part_pose[:3, 3] = current_part_pose[:3, 3] + pos_diff * 1/2
        # pre_pour_part_pose[:3, :3] = current_part_pose[:3, :3]
        pre_pour_pose = pre_pour_part_pose @ obj2part
        motion_type = 'AvoidObs'
        self.sub_stages.append([pre_pour_pose, gripper_action, transform_pre_pour, motion_type])


        motion_type = 'Trajectory'
        rotations = interpolate_rotation_matrices(current_part_pose[:3,:3], target_part_pose[:3,:3], 200)
        target_part_pose_list = np.tile(target_part_pose, (len(rotations), 1, 1))
        target_part_pose_list[:, :3, :3] = rotations
        target_pose_list = target_part_pose_list @ obj2part[np.newaxis, ...]

        transform_pour = np.tile(np.eye(4), (len(rotations), 1, 1))
        # transform_pour[:, 2, 3] = self.pre_pour_height
        transform_pour[:, 2, 3] = np.linspace(self.pre_pour_height, self.pour_heigth, len(rotations))
        # import ipdb; ipdb.set_trace()
        self.sub_stages.append([target_pose_list, gripper_action, transform_pour, motion_type])
            
            
    def get_action(self, objects):
        if self.__len__()==0:
            return None
        gripper_pose_canonical, gripper_action, transform_world, motion_type = self.sub_stages[self.step_id]

        if gripper_pose_canonical is None:
            target_gripper_pose = None
        else: 
            goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
            target_gripper_pose = solve_target_gripper_pose(goal_datapack, objects)

            # current_gripper_pose = objects['gripper'].obj_pose
            # if self.step_id==0:
            #     # pre_pour at 5cm away from current to target
            #     diff_xy_direction = target_gripper_pose[:2, 3] - current_gripper_pose[:2, 3]
            #     pos_diff = np.linalg.norm(diff_xy_direction) * 0.10
            #     target_gripper_pose[:2, 3] = current_gripper_pose[:2, 3] + pos_diff

            # elif self.step_id==1:
            #     target_xyz = target_gripper_pose[-1, :3, 3]
            #     current_xyz = current_gripper_pose[:3, 3]

            #     xyz_interp = np.linspace(current_xyz, target_xyz, len(target_xyz))
            #     target_gripper_pose[:, :3, 3] = xyz_interp

            #     import ipdb; ipdb.set_trace()
        
        last_statement = {'objects': copy.deepcopy(objects), 'target_gripper_pose': target_gripper_pose}
        self.last_statement = last_statement
        return target_gripper_pose, motion_type, gripper_action , "right"
    

    def check_completion(self, objects):
        if self.__len__()==0:
            return True
        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
        succ = simple_check_completion(goal_datapack, objects)

        
            

        if succ:
            self.step_id += 1
        return succ