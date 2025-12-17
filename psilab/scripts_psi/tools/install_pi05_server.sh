#!/usr/bin/env bash
# ==============================================================================
# install_pi05_server.sh
# 
# This script creates a Python 3.11 environment for running the pi0.5 policy
# server. The server communicates with IsaacLab via websocket.
#
# Usage:
#   chmod +x install_pi05_server.sh
#   ./install_pi05_server.sh
#
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
PI_DIR="$PROJECT_ROOT/psibot_pi"

echo "========================================"
echo "  Pi0.5 Policy Server Installation     "
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo_info "uv version: $(uv --version)"

# Navigate to psibot_pi directory
cd "$PI_DIR"

echo "========================================"
echo_info "Step 1: Creating Python 3.11 virtual environment..."
echo "========================================"

# Create virtual environment with Python 3.11
uv venv --python 3.11

echo "========================================"
echo_info "Step 2: Installing dependencies..."
echo "========================================"

# Install all dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Install the package in editable mode
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "========================================"
echo_info "Step 3: Applying transformers patches..."
echo "========================================"

# Apply transformers patches for PyTorch support
if [ -d "$PI_DIR/src/openpi/models_pytorch/transformers_replace" ]; then
    SITE_PACKAGES="$PI_DIR/.venv/lib/python3.11/site-packages"
    if [ -d "$SITE_PACKAGES/transformers" ]; then
        echo_info "Applying transformers patches..."
        cp -r "$PI_DIR/src/openpi/models_pytorch/transformers_replace/"* "$SITE_PACKAGES/transformers/"
    fi
fi

echo "========================================"
echo_info "Step 4: Verifying installation..."
echo "========================================"

source "$PI_DIR/.venv/bin/activate"

python -c "
import sys
print(f'Python version: {sys.version}')

# Test JAX
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')

# Test PyTorch
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test openpi
from openpi.policies import policy_config
from openpi.training import config
print(f'openpi: OK')

print('\\n✅ Pi0.5 server environment ready!')
"

echo "========================================"
echo -e "${GREEN}Installation complete!${NC}"
echo "========================================"
echo ""
echo "To start the pi0.5 policy server:"
echo ""
echo "  cd $PI_DIR"
echo "  source .venv/bin/activate"
echo "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo "      --policy.config=pi05_gbimg \\"
echo "      --policy.dir=/home/psibot/psi-lab-v2/psibot_pi/ckpt/3000 \\"
echo "      --default_prompt=\"Pick up the green carton of drink from the table.\" \\"
echo "      --port=8000"
echo ""
echo "Or use the provided launch script:"
echo "  ./scripts_psi/tools/start_pi05_server.sh"
echo ""

