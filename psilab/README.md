# PSI LAB  
    A framework for reinforcement learning, imitation learning, and motion planning.

---


# 🌟 Key Features
- **Scene config**: general config used for generating the environment for reinforcement learning, imitation learning, and motion planning.
- **Random Config**: provide various randomization options such as rigid object translations, light parameters, visual material, etc

# 💻 Recommended Configuration
- **OS**: **Ubuntu 22.04** is recommended. Other Linux distributions (such as Ubuntu 20.04 and 24.04) may work but are not officially supported.
- **GPU**: **Nvidia RTX 4090** is recommended. Other GPU (such as RTX 5090)  may work but are not officially supported.
- **GPU Driver**: Install the official NVIDIA driver version ≥ 525 (choose the driver version according to your CUDA version).
- **GLIBC Requirement**: GLIBC version ≥ 2.34 is required. Check your system version with `ldd --version`.

# Isaac Sim Version Dependency

Psi Lab is built on top of Isaac Sim and Isaac Lab.
| Psi Lab Version  | Isaac Lab Version | Isaac Sim Version  |
|:-----------------|:------------------|:-------------------|
|       v2.0       |  Isaac Lab v2.02  |   Isaac Sim 4.5    |

# 📦 Installation
## ⚡ Quick Installation
1. Download Source Code
   ```
   git clone -b develop ssh://git@14.103.194.101:10022/dev-simulation/psi-lab-v2.git 
   ```
2. Run Install Shell Script
   
   **Tip**: during the installation, you may be prompted to enter conda env name and your sudo password.
   ```
   cd psi-lab-v2
   chmod +x install.sh
   ./install.sh
   ```

## 🧰 Normal Installation
1. Create Conda Environment

   ```
   conda create -n psilab python=3.10
   conda activate psilab
   ```

2. Install PyTorch
   ```
   # CUDA 11
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   # CUDA 12.8 For RTX 5090
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   ```
3. Update pip
   ```
   pip install --upgrade pip
   ```
4. Install Isaac Sim 4.5

   ```
   pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
   ```
   Verifying the Isaac Sim installation.

   Make sure that your virtual environment is activated

   Check that the simulator runs as expected:
   ```
   # note: you can pass the argument "--help" to see all arguments possible.
   isaacsim
   ```
   It’s also possible to run with a specific experience file, run:
   ```
   # experience files can be absolute path, or relative path searched in isaacsim/apps or omni/apps
   isaacsim isaacsim.exp.full.kit
   ```
5. Download Source Code
   ```
   git clone ssh://git@14.103.194.101:10022/dev-simulation/psi-lab-v2.git
   ```
6. Install Dependencies
   ```
   cd psi-lab-v2
   ./isaaclab.sh -i
   ```

7. Fix Denpendecies Bug
   ```
   chmox +x scripts_psi/tools/fix_deps_bug/fix_deps_bug.sh
   ./scripts_psi/tools/fix_deps_bug/fix_deps_bug.sh
   ```


# 🛠️ Troubleshooting
## 1. libstdc++.so.6: version `GLIBCXX_3.4.30' not found
   The symbolic link of libstdc++.so.6 in conda env is wrong. Run the following commands to fix the problem.
   ```
   cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /your_conda_path/envs/env_name/lib/
   mv /your_conda_path/envs/env_name/lib/libstdc++.so.6 /your_conda_path/envs/env_name//lib/libstdc++.so.6.bak
   ln -s libstdc++.so.6.0.30 libstdc++.so.6
   ```
## 2. Could not load the dynamic library from librosidl_runtime_c.so.
   The System has no ffmpeg and dependencies. Run the following commands to fix the problem.
   ```
   sudo apt install -y ffmpeg
   sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev libswresample-dev
   ```
## 3. Code IDE is not Vscode
   Overwrite_python_analysis_extra_paths function in setup_vscode.py will ouput error:
   ```
   settings = settings.group(0)
   AttributeError: 'NoneType' object has no attribute 'group'
   ```
   If your IDE is cursor, change variables in settings.json from 
   ```
   "python.analysis.extraPaths":[]
   ```
   to 
   ```
   "cursorpyright.analysis.extraPaths":[]
   ```
     