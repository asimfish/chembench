from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference

import numpy as np





class PlaceStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=None, **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        self.pre_transform_up = np.array([0, 0, 0.05])
        self.place_transform_up = np.array([0, 0, 0.01])
        self.generate_substage(target_pose)
        
        
 
        
    def generate_substage(self, target_pose):
        # moveTo pre-place position
        target_pose_canonical = target_pose
        motion_type = 'AvoidObs'
        transform_up = np.eye(4)
        transform_up[:3,3] = self.pre_transform_up
        self.sub_stages.append([target_pose_canonical, None, transform_up, motion_type])
        
        # place
        palce_transform_up = np.eye(4)
        palce_transform_up[:3,3] = self.place_transform_up
        self.sub_stages.append([target_pose_canonical, 'open', palce_transform_up, 'Simple'])
        

    def check_completion(self, objects):
        if self.__len__()==0:
            return True
        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
        succ = True
        if self.step_id==0:
            succ = simple_check_completion(goal_datapack, objects)
        if succ:
            self.step_id += 1
        return succ
        


