# Makefile for grasp_100ml_beaker training

.PHONY: help test train eval clean

# Default target
help:
	@echo "====================================="
	@echo "Grasp 100ml Beaker Training"
	@echo "====================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make test     - Test setup (GPU, dataset, model)"
	@echo "  make train    - Train ACT policy"
	@echo "  make eval     - Evaluate trained policy"
	@echo "  make clean    - Clean checkpoints and logs"
	@echo "  make quickstart - Run full workflow interactively"
	@echo ""
	@echo "Advanced:"
	@echo "  make train-small  - Train with smaller model"
	@echo "  make train-large  - Train with larger model"
	@echo "  make train-long   - Train for more epochs"
	@echo ""

# Test setup
test:
	@echo "Testing setup..."
	@python3 test_setup.py

# Train with default parameters
train:
	@echo "Starting training with default parameters..."
	@./train_grasp_100ml_beaker.sh

# Evaluate trained model
eval:
	@echo "Evaluating trained model..."
	@./eval_grasp_100ml_beaker.sh

# Clean checkpoints
clean:
	@echo "Cleaning checkpoints..."
	@read -p "Are you sure you want to delete all checkpoints? (y/n) " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/grasp_100ml_beaker; \
		echo "✓ Checkpoints deleted"; \
	else \
		echo "Cancelled"; \
	fi

# Quick start workflow
quickstart:
	@./quickstart.sh

# Train with smaller model (for faster experimentation)
train-small:
	@echo "Training with smaller model..."
	@python3 train_grasp_beaker.py --mode train \
		--ckpt_dir ./checkpoints/grasp_beaker_small \
		--batch_size 16 \
		--num_epochs 1000 \
		--chunk_size 50 \
		--hidden_dim 256 \
		--dim_feedforward 1600

# Train with larger model (for better performance)
train-large:
	@echo "Training with larger model..."
	@python3 train_grasp_beaker.py --mode train \
		--ckpt_dir ./checkpoints/grasp_beaker_large \
		--batch_size 4 \
		--num_epochs 3000 \
		--chunk_size 150 \
		--hidden_dim 1024 \
		--dim_feedforward 4096

# Train for longer
train-long:
	@echo "Training for more epochs..."
	@python3 train_grasp_beaker.py --mode train \
		--ckpt_dir ./checkpoints/grasp_beaker_long \
		--num_epochs 5000

# Multi-seed training
train-multiseed:
	@echo "Training with multiple seeds..."
	@for seed in 0 1 2; do \
		echo "Training with seed $$seed..."; \
		python3 train_grasp_beaker.py --mode train \
			--ckpt_dir ./checkpoints/grasp_beaker_seed_$$seed \
			--seed $$seed; \
	done

# Show training curves
show-curves:
	@echo "Training curves saved in:"
	@ls -lh checkpoints/grasp_100ml_beaker/*.png 2>/dev/null || echo "No training curves found. Train first with 'make train'"

# Show evaluation results
show-results:
	@echo "Evaluation results:"
	@cat checkpoints/grasp_100ml_beaker/result_policy_best.txt 2>/dev/null || echo "No results found. Evaluate first with 'make eval'"

# Show evaluation videos
show-videos:
	@echo "Evaluation videos:"
	@ls -lh checkpoints/grasp_100ml_beaker/video*.mp4 2>/dev/null || echo "No videos found. Evaluate first with 'make eval'"

# Monitor training (show latest checkpoint info)
monitor:
	@echo "Latest checkpoints:"
	@ls -lht checkpoints/grasp_100ml_beaker/*.ckpt 2>/dev/null | head -5 || echo "No checkpoints found"
	@echo ""
	@echo "Latest training curves:"
	@ls -lht checkpoints/grasp_100ml_beaker/*.png 2>/dev/null | head -3 || echo "No curves found"

# Full workflow
all: test train eval show-results
	@echo "====================================="
	@echo "✓ Full workflow completed!"
	@echo "====================================="


