from .base import StageTemplate, solve_target_gripper_pose, simple_check_completion, pose_difference
import numpy as np




class PushStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=np.eye(4), vector_direction=None, extra_params=None, **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)

        extra_params = {} if extra_params is None else extra_params

        self.pre_distance = extra_params.get('pre_distance', -0.03)
        self.move_distance = extra_params.get('move_distance', 0.14)

        vector_direction = vector_direction / np.linalg.norm(vector_direction)
        self.goal_transform = np.eye(4)
        self.goal_transform[:3,3] = vector_direction * self.move_distance

        if passive_element is not None:
            self.joint_position_threshold = passive_element.get('joint_position_threshold', 0.5)
            assert self.joint_position_threshold >= 0 and self.joint_position_threshold <= 1 
            self.joint_direction = passive_element.get('joint_direction', 1)
            assert self.joint_direction in [-1, 1]
            self.joint_velocity_threshold = passive_element['joint_velocity_threshold']

        self.generate_substage(target_pose, vector_direction)

        
    def generate_substage(self, target_pose, vector_direction):
        print("PushStagePushStagePushStagePushStagePushStagePushStagePushStagePushStage")
        vector_direction = vector_direction / np.linalg.norm(vector_direction)
        # moveTo pre-place position
        pre_pose = target_pose.copy()
        pre_pose[:3,3] += vector_direction * self.pre_distance
        self.sub_stages.append([pre_pose, 'close', np.eye(4), 'AvoidObs'])
        
        # insert
        move_pose = target_pose.copy()
        move_pose[:3,3] += vector_direction * self.move_distance
        self.sub_stages.append([move_pose, None, np.eye(4), 'Straight'])


    def check_completion(self, objects):
        print("PushStagePushStagePushStageOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOK")
        if self.__len__()==0:
            return True

        # TODO 铰接joitn判定

        if self.step_id==0:
            succ = True

        if self.step_id>=1:
            if objects[self.passive_obj_id].is_articulated:
                joint_position = objects[self.passive_obj_id].joint_position
                joint_velocity = objects[self.passive_obj_id].joint_velocity
                lower_bound = objects[self.passive_obj_id].part_joint_limit['lower_bound']
                upper_bound = objects[self.passive_obj_id].part_joint_limit['upper_bound']
                if self.joint_direction == 1:
                    succ = joint_position > lower_bound + (upper_bound - lower_bound) * self.joint_position_threshold
                else:
                    succ = joint_position < lower_bound + (upper_bound - lower_bound) * self.joint_position_threshold
                succ = succ and joint_velocity < self.joint_velocity_threshold
            else:
                sub_stage = self.sub_stages[self.step_id]
        
                last_pose = self.last_statement['objects'][self.passive_obj_id].obj_pose
                target_obj_pose = last_pose @ self.goal_transform
                current_obj_pose = objects[self.passive_obj_id].obj_pose
                pos_diff, _ = pose_difference(current_obj_pose, target_obj_pose)
                succ = pos_diff < 0.04
        
        if succ:
            self.step_id += 1
        return succ


class ClickStage(StageTemplate):
    def __init__(self, active_obj_id, passive_obj_id, active_element=None, passive_element=None, target_pose=None, vector_direction=None, **kwargs):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        self.pre_distance = -0.03
        self.move_distance = 0.00
        self.generate_substage(target_pose, vector_direction)

        
    def generate_substage(self, target_pose, vector_direction):
        vector_direction = vector_direction / np.linalg.norm(vector_direction)
        # moveTo pre-place position
        pre_pose = target_pose.copy()
        pre_pose[:3,3] += vector_direction * self.pre_distance
        self.sub_stages.append([pre_pose, 'close', np.eye(4), 'AvoidObs'])
        
        # insert
        move_pose = target_pose.copy()
        move_pose[:3,3] += vector_direction * self.move_distance
        self.sub_stages.append([move_pose, None, np.eye(4), 'Simple'])

