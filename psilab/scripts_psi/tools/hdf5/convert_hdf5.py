# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-08-04
# Vesion: 1.0
import argparse
import numpy
import h5py
import torch
import os
import re
from datetime import datetime

import os
import getpass 
import shutil


from psilab.utils.hdf5_utils import convert_hdf5_to_single_env

# add argparse arguments
parser = argparse.ArgumentParser(description="This script fix dependecies bug.")
parser.add_argument("--source", type=str, default=None, help="The folder of HDF5 files to convert.")
parser.add_argument("--destination", type=str, default=None, help="The folder to save converted HDF5 files.")

# parse the arguments
args_cli = parser.parse_args()


if args_cli.source is None:
    raise Exception(f"The source folder is None")

if args_cli.destination is None:
    raise Exception(f"The destination folder is None")

convert_hdf5_to_single_env(args_cli.source,args_cli.destination)

