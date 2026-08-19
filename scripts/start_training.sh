#!/usr/bin/env bash
# ============================================================
# Start K1 velocity training in a tmux session.
# Run this from YOUR terminal (not Claude Code).
#
# Usage:
#   chmod +x scripts/start_training.sh
#   ./scripts/start_training.sh
#
# Monitor:
#   tmux attach -t k1_train
#   tail -f logs/train_k1_velocity.log
# ============================================================
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12
TRAIN=$WS/isaac_tasks/k1_velocity/scripts/train.py
LOG=$WS/logs/train_k1_velocity.log

mkdir -p "$WS/logs"

# Kill existing session if any
tmux kill-session -t k1_train 2>/dev/null || true

echo "Starting K1 velocity training..."
echo "Task:    Isaac-Velocity-Rough-K1-v0"
echo "Envs:    1024"
echo "Iter:    5000"
echo "WandB:   thakk100/booster_k1_locomotion"
echo "Log:     $LOG"
echo ""

tmux new-session -d -s k1_train -x 220 -y 50

# Set env vars in tmux
tmux send-keys -t k1_train "export OMNI_KIT_ACCEPT_EULA=YES" Enter
tmux send-keys -t k1_train "export WANDB_ENTITY=thakk100" Enter
tmux send-keys -t k1_train "export WANDB_PROJECT=booster_k1_locomotion" Enter
tmux send-keys -t k1_train "cd $WS" Enter

# Run training with tee to log
tmux send-keys -t k1_train \
  "$PYTHON $TRAIN \
  --task Isaac-Velocity-Rough-K1-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 5000 \
  --seed 42 \
  2>&1 | tee $LOG" Enter

echo "Session 'k1_train' started."
echo ""
echo "  Attach:  tmux attach -t k1_train"
echo "  Log:     tail -f $LOG"
echo "  WandB:   https://wandb.ai/thakk100/booster_k1_locomotion"
