from abc import abstractmethod


class Robot:
    @abstractmethod
    def get_prim_world_pose(self, prim_path):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def open_gripper(self, id="left", width=0.1):
        pass

    @abstractmethod
    def close_gripper(self, id="left", force=50):
        pass

    @abstractmethod
    def move(self, target, type="pose", speed=None):
        pass

    @abstractmethod
    def decode_gripper_pose(self, gripper_pose):
        pass

    @abstractmethod
    def get_ee_pose(self, id="right"):
        pass

    @abstractmethod
    def check_ik(self, pose, id="right"):
        pass
