#!/bin/bash

# Quick start guide for training grasp_100ml_beaker task
# This script demonstrates the complete workflow

echo "========================================"
echo "Quick Start: Training grasp_100ml_beaker"
echo "========================================"

# Step 1: Test setup
echo ""
echo "Step 1: Testing setup..."
echo "----------------------------------------"
python3 test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Setup test failed. Please fix the errors above."
    exit 1
fi

# Step 2: Ask user if they want to continue
echo ""
echo "=========================================="
echo "Setup test passed! Ready to train."
echo "=========================================="
echo ""
read -p "Do you want to start training now? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    echo ""
    echo "To train later, run:"
    echo "  ./train_grasp_100ml_beaker.sh"
    exit 0
fi

# Step 3: Start training
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
./train_grasp_100ml_beaker.sh

# Step 4: Show next steps
echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check training curves:"
echo "   ls -lh checkpoints/grasp_100ml_beaker/*.png"
echo ""
echo "2. Evaluate the model:"
echo "   ./eval_grasp_100ml_beaker.sh"
echo ""
echo "3. View evaluation videos:"
echo "   ls -lh checkpoints/grasp_100ml_beaker/video*.mp4"
echo "=========================================="

