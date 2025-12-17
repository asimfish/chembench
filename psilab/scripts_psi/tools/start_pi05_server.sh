#!/usr/bin/env bash
# ==============================================================================
# start_pi05_server.sh
# 
# Quick launch script for pi0.5 policy server
#
# Usage:
#   ./start_pi05_server.sh [config_name] [checkpoint_dir] [port]
#
# Examples:
#   ./start_pi05_server.sh                                    # Use defaults
#   ./start_pi05_server.sh pi05_gbimg /path/to/ckpt 8000     # Custom config
#
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
PI_DIR="$PROJECT_ROOT/psibot_pi"

# Default values
CONFIG_NAME="${1:-pi05_gbimg}"
CHECKPOINT_DIR="${2:-$PI_DIR/ckpt/3000}"
PORT="${3:-8000}"
DEFAULT_PROMPT="Pick up the green carton of drink from the table."

echo "========================================"
echo "  Starting Pi0.5 Policy Server"
echo "========================================"
echo "Config: $CONFIG_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Port: $PORT"
echo "Prompt: $DEFAULT_PROMPT"
echo "========================================"

cd "$PI_DIR"
source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG_NAME" \
    --policy.dir="$CHECKPOINT_DIR" \
    --default_prompt="$DEFAULT_PROMPT" \
    --port="$PORT"

