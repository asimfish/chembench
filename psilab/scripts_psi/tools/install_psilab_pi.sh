#!/usr/bin/env bash
# ==============================================================================
# install_psilab_pi.sh
# 
# This script creates a unified Python environment that supports both:
#   - psilab (IsaacLab-based simulation)
#   - psibot_pi (openpi / pi0.5 policy inference)
#
# Usage:
#   chmod +x install_psilab_pi.sh
#   ./install_psilab_pi.sh
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "========================================"
echo "  PsiLab + Pi0.5 Unified Environment   "
echo "========================================"

# Get operating system information
if [ -f /etc/os-release ]; then
    . /etc/os-release
    os_name=$NAME
    os_version=$(echo $VERSION | cut -d "." -f -2)
fi

# Check OS
if [ "$os_name" != "Ubuntu" ] || { [ "$os_version" != "22.04" ] && [ "$os_version" != "24.04" ]; }; then
    echo_error "This script only supports Ubuntu 22.04 and 24.04"
    exit 1
fi
echo_info "OS: $os_name $os_version"

# Check GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo_error "nvidia-smi not found. NVIDIA driver required."
    exit 1
fi
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo_info "GPU: $gpu_info"

# Check conda
if ! command -v conda >/dev/null 2>&1; then
    echo_error "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi
conda_path=$(conda info --base)
echo_info "Conda: $conda_path"

echo "========================================"

# Environment name
ENV_NAME="psilab_pi"
read -p "Enter the name of the conda environment (default: $ENV_NAME): " user_env_name
ENV_NAME=${user_env_name:-$ENV_NAME}

# Check if environment exists
env_exists=$(conda env list | awk '{print $1}' | grep -w "^${ENV_NAME}$" || true)
if [ ! -z "$env_exists" ]; then
    read -p "Environment '$ENV_NAME' already exists. Remove and recreate? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo_info "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo_info "Using existing environment. Skipping creation."
    fi
fi

# Create environment if it doesn't exist
env_exists=$(conda env list | awk '{print $1}' | grep -w "^${ENV_NAME}$" || true)
if [ -z "$env_exists" ]; then
    echo_info "Creating conda environment '$ENV_NAME' with Python 3.11..."
    conda create -y -n $ENV_NAME python=3.11
fi

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
echo_info "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

echo "========================================"
echo_info "Step 1: Installing system dependencies..."
echo "========================================"

sudo apt-get update
sudo apt-get install -y ffmpeg cmake build-essential
sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev libswresample-dev

echo "========================================"
echo_info "Step 2: Installing PyTorch..."
echo "========================================"

# Install PyTorch 2.7.1 with CUDA 12.8 (compatible with both projects)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

echo "========================================"
echo_info "Step 3: Installing Isaac Sim..."
echo "========================================"

pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

echo "========================================"
echo_info "Step 4: Installing IsaacLab and PsiLab..."
echo "========================================"

cd "$PROJECT_ROOT"
export OMNI_KIT_ACCEPT_EULA=YES

# Install IsaacLab and extensions
bash isaaclab.sh --install

echo "========================================"
echo_info "Step 5: Installing openpi (pi0.5) dependencies..."
echo "========================================"

# Install JAX with CUDA 12
pip install "jax[cuda12]==0.5.3"

# Install other openpi dependencies (excluding torch which is already installed)
pip install \
    "augmax>=0.3.4" \
    "dm-tree>=0.1.8" \
    "einops>=0.8.0" \
    "equinox>=0.11.8" \
    "flatbuffers>=24.3.25" \
    "flax==0.10.2" \
    "fsspec[gcs]>=2024.6.0" \
    "gym-aloha>=0.1.1" \
    "imageio>=2.36.1" \
    "jaxtyping==0.2.36" \
    "ml_collections==1.0.0" \
    "numpydantic>=1.6.6" \
    "opencv-python>=4.10.0.84" \
    "orbax-checkpoint==0.11.13" \
    "pillow>=11.0.0" \
    "sentencepiece>=0.2.0" \
    "tqdm-loggable>=0.2" \
    "typing-extensions>=4.12.2" \
    "tyro>=0.9.5" \
    "wandb>=0.19.1" \
    "filelock>=3.16.1" \
    "beartype==0.19.0" \
    "treescope>=0.1.7" \
    "transformers==4.53.2" \
    "rich>=14.0.0" \
    "polars>=1.30.0"

# Install lerobot from git
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# Install openpi-client from local package
pip install -e "$PROJECT_ROOT/psibot_pi/packages/openpi-client"

# Install openpi from local source
pip install -e "$PROJECT_ROOT/psibot_pi"

echo "========================================"
echo_info "Step 6: Applying patches and fixes..."
echo "========================================"

# Fix numpy version (need to balance between IsaacLab and openpi requirements)
pip install "numpy>=1.23.5,<2.0.0"

# Apply transformers patches for PyTorch pi0.5 support
if [ -d "$PROJECT_ROOT/psibot_pi/src/openpi/models_pytorch/transformers_replace" ]; then
    echo_info "Applying transformers patches for pi0.5 PyTorch support..."
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    cp -r "$PROJECT_ROOT/psibot_pi/src/openpi/models_pytorch/transformers_replace/"* "$SITE_PACKAGES/transformers/"
fi

# Fix numba bug
if [ -f "$PROJECT_ROOT/scripts_psi/tools/fix_deps_bug/numba-0.57.0/__init__.py" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    if [ -f "$SITE_PACKAGES/numba/__init__.py" ]; then
        echo_info "Applying numba fix..."
        cp "$SITE_PACKAGES/numba/__init__.py" "$SITE_PACKAGES/numba/__init__-bak.py"
        cp "$PROJECT_ROOT/scripts_psi/tools/fix_deps_bug/numba-0.57.0/__init__.py" "$SITE_PACKAGES/numba/__init__.py"
    fi
fi

# Fix trimesh bug
if [ -f "$PROJECT_ROOT/scripts_psi/tools/fix_deps_bug/trimesh-4.6.5/transformations.py" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    if [ -f "$SITE_PACKAGES/trimesh/transformations.py" ]; then
        echo_info "Applying trimesh fix..."
        cp "$SITE_PACKAGES/trimesh/transformations.py" "$SITE_PACKAGES/trimesh/transformations-bak.py"
        cp "$PROJECT_ROOT/scripts_psi/tools/fix_deps_bug/trimesh-4.6.5/transformations.py" "$SITE_PACKAGES/trimesh/transformations.py"
    fi
fi

echo "========================================"
echo_info "Step 7: Verifying installation..."
echo "========================================"

# Test imports
python -c "
import sys
print(f'Python version: {sys.version}')

# Test PyTorch
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

# Test JAX
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')

# Test Isaac Sim
try:
    import isaacsim
    print(f'Isaac Sim: OK')
except Exception as e:
    print(f'Isaac Sim: {e}')

# Test IsaacLab
try:
    import isaaclab
    print(f'IsaacLab: OK')
except Exception as e:
    print(f'IsaacLab: {e}')

# Test psilab
try:
    import psilab
    print(f'PsiLab: OK')
except Exception as e:
    print(f'PsiLab: {e}')

# Test openpi
try:
    from openpi.policies import policy_config
    from openpi.training import config
    print(f'openpi: OK')
except Exception as e:
    print(f'openpi: {e}')

print('\\n✅ All core packages verified!')
"

echo "========================================"
echo -e "${GREEN}Installation complete!${NC}"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run IsaacLab with pi0.5 inference:"
echo "  python scripts_psi/workflows/imitation_learning/play.py --task Psi-IL-Grasp-Bottle-Chempi-v1 ..."
echo ""

