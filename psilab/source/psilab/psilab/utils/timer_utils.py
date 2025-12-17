# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import time


from psilab.utils.singleton_meta import SingletonMeta

class Timer(metaclass=SingletonMeta):


    def __init__(self):
        self._start_time = time.time()

    def run_time(self):
        return int(time.time() - self._start_time)
        
    @property
    def start_time(self):
        return self._start_time

    