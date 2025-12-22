from .utils import OmniObject
from abc import abstractmethod


class Planner:
    @abstractmethod
    def deduce_target_pose(self, active_obj, passive_obj, N=1):
        pass

    @abstractmethod
    def plan_trajectory(
        self,
        active_obj,
        target_obj_pose,
        gripper_pose,
        task,
        gripper_id=None,
        ik_checker=None,
    ):
        pass
