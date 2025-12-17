# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-01-22
# Vesion: 1.0

from abc import abstractmethod
import threading
import time

class TeleOperateDeviceCfgBase():
    """
    """
    def __init__(self):
        pass

class TeleOperateDeviceBase():
    """
    Base Class for TeleOperation Device
    """
    def __init__(self):

        # simulation progress flags
        self.bControl = False # reset flag
        self.bReset = False # reset flag
        self.bRecording = False # recording flag
        self.bFinished = False 
        # device update thread
        self._thread : threading.Thread = None # type: ignore
        # device update frequency
        self._update_frequency = 30.0 # HZ
        self._update_delta_time = 1 / self._update_frequency # seconds
        # device update thread running flag
        self.bRunning = False
        
    
    @abstractmethod
    def device_init(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        self.bReset = False
        self.bRecording = False
        self.bControl = False
        self.bFinished = False # recording flag

    @abstractmethod
    def update(self):
        raise NotImplementedError
    
    @abstractmethod
    def is_connected(self)->bool:
        raise NotImplementedError
    
    def start(self):
        # start device update thread
        self.is_running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self):
        # loop
        while self.is_running:
            # update
            current_time = time.time()
            self.update()
            elapsed = time.time() - current_time
            # control device update frequency
            if elapsed < self._update_delta_time:
                time.sleep(self._update_delta_time - elapsed)