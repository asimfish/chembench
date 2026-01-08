#!/bin/bash

# Setup script for ACT detr module
# This script downloads the detr directory from the official ACT repository

echo "=================================================="
echo "Setting up DETR module for ACT"
echo "=================================================="

cd /home/psibot/chembench/my_act

# Check if detr already exists
if [ -d "detr" ]; then
    echo "✓ detr directory already exists"
    echo "  If you want to reinstall, delete the detr directory first"
    exit 0
fi

# Clone the official ACT repository to a temporary location
echo "1. Downloading ACT repository..."
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
git clone https://github.com/tonyzhaozh/act.git

if [ $? -ne 0 ]; then
    echo "✗ Failed to clone ACT repository"
    rm -rf $TMP_DIR
    exit 1
fi

echo "✓ ACT repository downloaded"

# Copy detr directory
echo "2. Copying detr directory..."
cp -r act/detr /home/psibot/chembench/my_act/

if [ $? -ne 0 ]; then
    echo "✗ Failed to copy detr directory"
    rm -rf $TMP_DIR
    exit 1
fi

echo "✓ detr directory copied"

# Clean up
echo "3. Cleaning up..."
rm -rf $TMP_DIR
echo "✓ Cleanup complete"

# Install detr
echo "4. Installing detr..."
cd /home/psibot/chembench/my_act/detr
pip install -e .

if [ $? -ne 0 ]; then
    echo "✗ Failed to install detr"
    echo "  You may need to install it manually: cd my_act/detr && pip install -e ."
    exit 1
fi

echo "✓ detr installed"

echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo "Now you can run ACT evaluation:"
echo "  python psilab/scripts_psi/workflows/imitation_learning/play.py \\"
echo "    --task Psi-IL-Grasp-ACT-v1 \\"
echo "    --checkpoint /path/to/your/checkpoint.ckpt \\"
echo "    ..."
echo "=================================================="


