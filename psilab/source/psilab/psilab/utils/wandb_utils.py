# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import wandb


from psilab.utils.singleton_meta import SingletonMeta

class WandbLog(metaclass=SingletonMeta):


    def __init__(self):
        self.log_data : dict[ str, float]= {} # type: ignore 
        self.step : int = 0
        self._init = False


    def init_wandb(self, project:str, name:str):
        wandb.init(project=project, name=name)  
        self.project = project
        self.name = name
        self._init = True
        
    def set_data(self, key:str, value:float):
        self.log_data[key] = value

    def get_data(self, key:str)->float:
        return self.log_data[key]
    
    def get_step(self)->int:
        return self.step
    
    def upload(self,key:str):

        if key not in self.log_data.keys() or not self._init:
            return
        # if len(self.log_data.keys())==0:
        #     return
        
        wandb.log(
            { key:self.log_data[key]}, 
            step = self.step
            )

    