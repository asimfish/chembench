#!/usr/bin/env bash
# ==============================================================================
# install.sh
# 
# This script is used to fix dependency bugs in the given conda environment.
# please ensure that you have the necessary permissions and a network connection.
#
# Usage:
#   1. Grant executable permissions (if not already set):
#      chmod +x fix_deps_bug.sh
#   2. Run the script:
#      ./fix_deps_bug.sh
#
# Overview of features:
#   - Configure the environment required for the project to run
#   - Install dependent software packages and tools
#   - Fix dependency library bugs
#
# Notes:
#   - It is recommended to run this script in a virtual environment or a clean environment.
#   - If you encounter permission issues, please use sudo or run as the root user.
#   - The specific installation content please adjust the script subsequent implementation part according to the project.
#
# ==============================================================================


# Get GPU information.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi tool not found, NVIDIA driver or NVIDIA GPU not detected."
    exit 1
fi
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [ -z "$gpu_info" ]; then
    echo "No available NVIDIA GPU detected, nvidia-smi did not output valid information." >&2
    exit 1
else
    gpu_version=$(echo $gpu_info | cut -d " " -f 4)
fi

if [[ "$gpu_info" != *"4090"* && "$gpu_info" != *"5090"* ]]; then
    echo "This script only supports NVIDIA 4090 or 5090 GPUs, currently detected GPU: $gpu_info" >&2
    exit 1
fi
echo "GPU Info:"
echo "Version: $gpu_info"

echo "---------------------------------------"

# Get Conda Path
if command -v conda >/dev/null 2>&1; then
    conda_path=$(conda info --base)
    # conda_path=$(command -v conda)
    if [[ $conda_path == *"anaconda"* ]]; then
        echo "Anaconda detected: $conda_path"
    elif [[ $conda_path == *"miniconda"* ]]; then
        echo "Miniconda detected: $conda_path"
    else
        echo "conda detected, but cannot determine if it is Anaconda or Miniconda: $conda_path"
    fi
else
    echo "Anaconda or Miniconda not detected, please install one of them first."
    exit 1
fi

# Create Conda Environment
read -p "Enter the name of the conda environment to create (default: psilab): " conda_env_name
conda_env_name=${conda_env_name:-psilab}


# Check if the environment already exists
env_list=$(conda env list | awk '{print $1}' | grep -w "^${conda_env_name}$" || true)
if [ -z "$env_list" ]; then
    echo "The environment named ${conda_env_name} is not exists."
    exit 1


# Fix bugs with RTX 5090
if [ "$gpu_version" == "5090" ]; then
    echo "Fix dependency library bugs: Rl-games with RTX 5090"
    cp $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/rl_games/algos_torch/torch_ext.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/rl_games/algos_torch/torch_ext-bak.py
    cp scripts_psi/tools/fix_deps_bug/rl-games/torch_ext.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/rl_games/algos_torch/torch_ext.py
fi

# Fix dependency library bugs
echo "Fix dependency library bugs: Numba"
cp $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/numba/__init__.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/numba/__init__-bak.py
cp scripts_psi/tools/fix_deps_bug/numba-0.57.0/__init__.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/numba/__init__.py

echo "Fix dependency library bugs: Trimesh"
cp $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/trimesh/transformations.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/trimesh/transformations-bak.py
cp scripts_psi/tools/fix_deps_bug/trimesh-4.6.5/transformations.py \
    $conda_path/envs/$conda_env_name/lib/python3.10/site-packages/trimesh/transformations.py

echo "---------------------------------------"