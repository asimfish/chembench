# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-11
# Vesion: 1.0

import time
from functools import wraps

def cost_time(func):
    """
    Decorator to measure the execution time of a function in reinforcement 
    learning environment and log the result to wandb.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        finish_time = time.time()
        self._step_func_cost_time = (finish_time - start_time)
        #
        if self.cfg.enable_wandb:  # type: ignore
            # current sim fps
            self.frames = self.cfg.decimation * self.num_envs
            sim_fps = self.frames / self._step_func_cost_time
            self._wandb.step = self.common_step_counter * self.num_envs
            self._wandb.set_data("performance/sim_fps",sim_fps) # type: ignore
            self._wandb.upload("performance/sim_fps")
        return result
    return wrapper
   
