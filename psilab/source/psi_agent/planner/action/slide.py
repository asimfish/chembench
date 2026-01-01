from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference

import copy
import numpy as np





class SlideStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=np.eye(4), **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)

        self.pre_distance = 0.0
        self.move_distance = 0.08
        vector_direction = passive_element['direction']
        self.generate_substage(target_pose, vector_direction)



    def generate_substage(self, target_pose, vector_direction):
        vector_direction = vector_direction / np.linalg.norm(vector_direction)
        # slide
        move_pose = target_pose.copy()
        move_pose[:3,3] += vector_direction * self.move_distance
        self.sub_stages.append([move_pose, None, np.eye(4), 'Simple'])

        # open gripper
        self.sub_stages.append([None, 'open', np.eye(4), None])

    

    def check_completion(self, objects):
        if self.__len__()==0:
            return True
        sub_stage = self.sub_stages[self.step_id]
        move_transform, gripper_action = sub_stage[0], sub_stage[1]
        if gripper_action=='open':
            self.step_id += 1
            return True
        last_pose = self.last_statement['objects'][self.passive_obj_id].obj_pose
        target_obj_pose = last_pose @ move_transform
        current_obj_pose = objects[self.passive_obj_id].obj_pose
        pos_diff, _ = pose_difference(current_obj_pose, target_obj_pose)
        succ = pos_diff < 0.04
        print(pos_diff, succ)
        if succ:
            self.step_id += 1
        return succ


