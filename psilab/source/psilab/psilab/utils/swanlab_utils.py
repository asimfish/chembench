# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import swanlab


from psilab.utils.singleton_meta import SingletonMeta

class SwanlabLog(metaclass=SingletonMeta):


    def __init__(self):
        self.log_data : dict[ str, float]= {} # type: ignore 
        self.step : int = 0
        self._init = False


    def init_wandb(self, project:str,experiment_name:str):
        swanlab.init(project=project,experiment_name=experiment_name)  
        self.project = project
        self.experiment_name = experiment_name
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
        
        swanlab.log(
            { key:self.log_data[key]}, 
            step = self.step
            )

    