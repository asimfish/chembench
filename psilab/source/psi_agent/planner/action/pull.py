from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference

import numpy as np

class PullStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=np.eye(4), **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        self.pull_distance = passive_element['distance']
        assert self.pull_distance > 0
        self.joint_position_threshold = passive_element['joint_position_threshold']
        assert self.joint_position_threshold >= 0 and self.joint_position_threshold <= 1 
        self.joint_direction = passive_element['joint_direction']
        assert self.joint_direction in [-1, 1]
        self.joint_velocity_threshold = passive_element['joint_velocity_threshold']
        vector_direction = passive_element['direction']
        self.generate_substage(target_pose, vector_direction)

    def generate_substage(self, target_pose, vector_direction):
        print("PullStagePullStagePullStagePullStagePullStagePullStagePullStage")
        vector_direction = vector_direction / np.linalg.norm(vector_direction)
        # moveTo pre-place position
        move_pose = target_pose.copy()
        move_pose[:3,3] += vector_direction * self.pull_distance
        self.sub_stages.append([move_pose, None, np.eye(4), 'Straight'])
        self.sub_stages.append([None, "open", np.eye(4), 'Simple'])

        free_delta_pose = np.eye(4)
        free_delta_pose[2,3] = -0.03
        self.sub_stages.append([free_delta_pose, None, np.eye(4), 'local_gripper'])

        


    def check_completion(self, objects):
        if self.__len__()==0:
            return True
        
        if self.step_id >= 0:
            joint_position = objects[self.passive_obj_id].joint_position
            joint_velocity = objects[self.passive_obj_id].joint_velocity
            lower_bound = objects[self.passive_obj_id].part_joint_limit['lower_bound']
            upper_bound = objects[self.passive_obj_id].part_joint_limit['upper_bound']
            if self.joint_direction == 1:
                succ = joint_position > lower_bound + (upper_bound - lower_bound) * self.joint_position_threshold
            else:
                succ = joint_position < lower_bound + (upper_bound - lower_bound) * self.joint_position_threshold
            succ = succ and joint_velocity < self.joint_velocity_threshold

        if succ:
            self.step_id += 1
        # else:
            # import ipdb;ipdb.set_trace()

        return succ
        