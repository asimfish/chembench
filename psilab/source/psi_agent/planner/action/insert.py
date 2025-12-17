from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference

import numpy as np



class InsertStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=None, vector_direction=None, extra_params={}, **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)


        self.move_distance = 0.01
        self.generate_substage(target_pose, vector_direction, extra_params)
        

    def generate_substage(self, target_pose, vector_direction, extra_params={}):
        insert_pre_distance = extra_params.get('insert_pre_distance', -0.11)
        insert_motion_type = extra_params.get('insert_motion_type', 'Simple')

        vector_direction = vector_direction / np.linalg.norm(vector_direction)

        # moveTo pre-place position
        pre_pose = target_pose.copy()
        pre_pose[:3,3] += vector_direction * insert_pre_distance
        self.sub_stages.append([pre_pose, None, np.eye(4), 'AvoidObs'])
        
        # insert
        move_pose = target_pose.copy()
        move_pose[:3,3] += vector_direction * self.move_distance
        self.sub_stages.append([move_pose, None, np.eye(4), insert_motion_type])

        # open gripper
        self.sub_stages.append([None, 'open', np.eye(4), None])

    def check_completion(self, objects):
        if self.__len__()==0:
            return True
        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
        succ = simple_check_completion(goal_datapack, objects)
        if succ:
            self.step_id += 1
        return succ
        

        
class HitStage(InsertStage):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=np.eye(4), vector_direction=None, **kwargs):
        super(InsertStage, self).__init__(active_obj_id, passive_obj_id, active_element, passive_element)

        self.pre_distance = -0.05
        self.move_distance = 0.005
        self.generate_substage(target_pose, vector_direction)


