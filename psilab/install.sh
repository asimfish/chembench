#!/usr/bin/env bash
# ==============================================================================
# install.sh
# 
# This script is used to automate the installation of PsiLab and the configuration of the 
# environment and dependencies required for the project. Before executing this script, 
# please ensure that you have the necessary permissions and a network connection.
#
# Usage:
#   1. Grant executable permissions (if not already set):
#      chmod +x install.sh
#   2. Run the script:
#      ./install.sh
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


# Get operating system information.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    os_name=$NAME
    os_version=$(echo $VERSION | cut -d "." -f -2)
    host_name=$(uname -n)
fi

# This script only supports Ubuntu 22.04 and 24.04.
if [ "$os_name" != "Ubuntu" ] || [ "$os_version" != "22.04" ] && [ "$os_version" != "24.04" ]; then
    echo "This script only supports Ubuntu 22.04 and 24.04" >&2
    exit 1
fi


echo "---------------------------------------"
echo "Operating System Info:"
echo "Name: $(uname -s)"
echo "Kernel Version: $(uname -r)"
echo "Host Name: $host_name"
echo "Architecture: $(uname -m)"
echo "Distribution: $os_name"
echo "Version: $os_version"
echo "---------------------------------------"

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
if [ ! -z "$env_list" ]; then
    echo "The environment named ${conda_env_name} already exists, no need to create it again."
else
    echo "Creating conda environment named ${conda_env_name} with Python version 3.10..."
    conda create -y -n ${conda_env_name} python=3.10 > /dev/null
    if [ $? -eq 0 ]; then
        echo "conda environment ${conda_env_name} created successfully!"
    else
        echo "Failed to create conda environment ${conda_env_name}, please check the output information." >&2
        exit 1
    fi
fi

# Check if conda shell hook is initialized, if not, run 'conda init'
if ! command -v conda >/dev/null 2>&1 || ! grep -q 'conda.sh' ~/.bashrc; then
    echo "Running 'conda init'..."
    conda init
    # After running conda init, the user may need to restart the shell.
    # We'll source the hook for current session if possible.
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Activate the environment
echo "Activating conda environment: ${conda_env_name}"
conda activate ${conda_env_name}

echo "---------------------------------------"

# # Update pip
# pip install --upgrade pip > /dev/null 2>&1
# echo "Upgrade pip"
# echo "---------------------------------------"

# # install ffmpeg, otherwise isaaclab.sh will get error: failed to build 'av' 
# # when getting required to build wheel
# sudo apt-get install -y ffmpeg > /dev/null
# sudo apt-get install libavformat-dev libavcodec-dev \
#     libavdevice-dev libavfilter-dev libavutil-dev \
#     libswscale-dev libswresample-dev > /dev/null 2>&1
# echo "Install ffmpeg and dependencies"


# echo "---------------------------------------"

# # Install pytorch
# if [ "$gpu_version" == "4090" ]; then
#     echo "Install PyTorch with CUDA 12.1 for RTX 4090"
#     # CUDA 12.1
#     pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 > /dev/null
# elif [ "$gpu_version" == "5090" ]; then
#     echo "Install PyTorch with CUDA 12.8 for RTX 5090"
#     # CUDA 12.8
#     pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 > /dev/null
# fi
# echo "---------------------------------------"

# # Install Isaac Sim
# echo "Install Isaac Sim 4.5.0"
# pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com > /dev/null
# echo "Testing IsaacSim, please close it to continue script."
# echo "Tip:The first run of the Isaac Sim app takes some time(about 5-10 minutes) to warm up the shader cache."
# export OMNI_KIT_ACCEPT_EULA=YES
# isaacsim isaacsim.exp.full.kit > /dev/null 2>&1

# echo "---------------------------------------"

# # Install dependencies
# echo "Run isaaclab.sh to install dependencies"
# sudo apt-get install cmake build-essential > /dev/null
# bash isaaclab.sh --install
# echo "---------------------------------------"


# Fix bugs with RTX 5090
if [ "$gpu_version" == "5090" ]; then
    echo "Fix dependency library bugs: Pytorch with RTX 5090"
    torch_version=$($conda_path/envs/$conda_env_name/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ ! -z $(echo $torch_version | grep "2.5.1") ]; then
        echo "Current Torch version: $torch_version is not com for RTX 5090"
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 > /dev/null
        echo "Install PyTorch with CUDA 12.8 for RTX 5090, again ..."
    fi
    echo "Fix dependency library bugs: Numpy with RTX 5090"
    pip install numpy==1.23.5 > /dev/null 2>&1
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

# # Check if sshpass is installed
# if ! command -v sshpass &> /dev/null
# then
#     echo "Installing sshpass..."
#     sudo apt install sshpass
# fi

# echo "Download Psi Lab Assets"
# sshpass -p PsiRobot2024 scp -r psirobot@172.16.0.11:/volume1/home/psirobot/psi-lab-v2/assets .
# echo "---------------------------------------"
# echo "Download Checkpoints"
# sshpass -p PsiRobot2024 scp -r psirobot@172.16.0.11:/volume1/home/psirobot/psi-lab-v2/logs .
# echo "---------------------------------------"
# echo "The Installation is complete, you can now run the project."
# echo "---------------------------------------"

