# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import numpy
import h5py
import torch
import os
import re
import time
import math
from datetime import datetime
def write_data_to_hdf5(data:dict, hdf5:h5py.File, current_path:str):
    """
   Write dict data to hdf5 file with path
    """
    dtype_str = h5py.special_dtype(vlen=str)
    for key, value in data.items():
        path = current_path+key+"/"
        # 
        if isinstance(value, dict):
            #
            write_data_to_hdf5(value, hdf5, current_path+key+"/")
        #
        elif isinstance(value, list):
            #
            if len(value)==0:
                continue
            # str
            if isinstance(value[0], str):
                hdf5.create_dataset(current_path+key,dtype=dtype_str,data=value)
            # uint8
            elif type(value[0])==numpy.uint8:
                hdf5.create_dataset(current_path+key,dtype=numpy.uint8,data=value)
            # 
            elif isinstance(value[0], int):
                hdf5.create_dataset(current_path+key,dtype=numpy.int8,data=value)
            # 
            elif isinstance(value[0], torch.Tensor):
                # rgb image
                if value[0].dtype == torch.uint8:
                    hdf5.create_dataset(current_path+key,dtype=numpy.uint8,data=value)
                else:
                    hdf5.create_dataset(current_path+key,dtype=numpy.float32,data=value)
            # image 
            elif type(value[0])==numpy.ndarray:
                value_type = value[0].dtype
                hdf5.create_dataset(current_path+key,dtype=value_type,data=value)
            # elif type(value[0])==type(numpy.float64):
            #     h5_file.create_dataset(current_path+key,dtype=numpy.float64,data=value)
            else:
                hdf5.create_dataset(current_path+key,dtype=numpy.float32,data=value)
        #
        elif isinstance(value, str):
            hdf5.create_dataset(current_path+key,dtype=dtype_str,data=value)


def get_data_from_hdf5(data:dict, hdf5:h5py.File |  h5py.Group):
    for key, item in hdf5.items():
        if isinstance(item, h5py.Group):
            # If it's a group, create a nested dictionary and recurse
            data[key] = {}
            get_data_from_hdf5(data[key],item)
        elif isinstance(item, h5py.Dataset):
            # If it's a dataset, load the data into the dictionary
            data[key] = item[()] # Use [()] to load the entire dataset
        else:
            pass

def convert_tensor_to_cpu(data:dict):
    """
    Convert all tensor value in dict to cpu
    """
    for key, value in data.items():
        if isinstance(value, dict):
            convert_tensor_to_cpu(value)
        elif isinstance(value, list):
            if len(value)==0:
                continue
            if isinstance(value[0], torch.Tensor):
                list_temp = []
                for list_gpu_value in value:
                    list_temp.append(list_gpu_value.cpu().numpy())
                data[key] = list_temp   
        elif isinstance(value, torch.Tensor):
            data[key] = value.cpu().tolist()


def convert_hdf5_to_single_env(source_path:str, destination:str):

    count = 0

    # 
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok= True)

    # Get All H5 in File
    h5_file_names=[]
    for file in os.listdir(source_path):
        if file.split(".")[-1]=="hdf5":
            h5_file_names.append(file)
    ignore_count = 0
    # 
    for h5_file_name in h5_file_names:

        h5_file = h5py.File(os.path.join(source_path,h5_file_name), 'r')
        h5_file_name_base = h5_file_name.split(".")[0]
        # get all env name
        env_names = []
        for key_name in list(h5_file.keys()):
            if re.match(r"env_[0-9]+", key_name) is not None:
                env_names.append(key_name)
        # 
        for env_name in env_names:
            
            filename = f"{h5_file_name_base}_{env_name}.hdf5"
            h5_single_file = h5py.File(os.path.join(destination,filename),'w')
            #
            for key_name in list(h5_file[env_name].keys()): # type: ignore
                h5_file.copy(env_name+"/" + key_name,h5_single_file)
            #
            h5_file.copy("scene",h5_single_file)
            h5_file.copy("task",h5_single_file)

            h5_single_file.close()

        
        h5_file.close()         
        print(ignore_count)
        # h5_temp.clear()
        # print(h5_file_name)
        

def hdf5_filter(env:h5py.Dataset)-> bool:
    pass
    # 判断物体是否有水平位移
    length = env["rigid_objects"]["target"].shape[0]
    pos_init = env["rigid_objects"]["target"][0,0:2]
    quat_init = env["rigid_objects"]["target"][0,3:7]

    bignore = False
    for i in range(length):
        pos = env["rigid_objects"]["target"][i,0:2]
        dx = pos[0] - pos_init[0]
        dy = pos[1] - pos_init[1]
        if (dx**2 + dy**2) >= 0.005:
            bignore = True
            break
    #
    for i in range(length):
        quat = env["rigid_objects"]["target"][i,3:7]
        dw = quat[0] - quat_init[0]
        dx = quat[1] - quat_init[1]
        dy = quat[2] - quat_init[2]
        dz = quat[3] - quat_init[3]

        if math.sqrt(dw**2 + dx**2 + dy**2 + dz**2) >= 0.7:
            bignore = True
            break
    return bignore