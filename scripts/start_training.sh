#!/usr/bin/env bash
# ============================================================
# Phase-6 K1 training launcher (guarded, headless, wandb-logged).
#
# Usage:
#   ./scripts/start_training.sh --task Isaac-Velocity-Rough-K1-v0 \
#       [--num_envs 256] [--max_iterations 1000] [--seed 42] [--name run_name]
#
# Monitor:
#   tmux attach -t k1_train              (live status lines)
#   tail -f logs/guard_<name>.log        (full training output)
#
# Machine rules enforced here: headless only, ≤512 envs, one guarded run
# at a time (train_guard.sh), wandb entity thakk100-dhyan-home.
# ============================================================
set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/phase6_env.sh"
TRAIN=$WS/isaac_tasks/k1_velocity/scripts/train.py

TASK="Isaac-Velocity-Rough-K1-v0"
NUM_ENVS=256
MAX_ITERS=1000
SEED=42
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --max_iterations) MAX_ITERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Local-box hard cap (Phase-6 machine rule: never >512 envs here)
if [ "$NUM_ENVS" -gt 512 ]; then
  echo "REFUSED: --num_envs $NUM_ENVS exceeds the 512 local cap" >&2
  exit 2
fi
NAME="${NAME:-$TASK}"

export WANDB_PROJECT="${WANDB_PROJECT:-booster_k1_soccer_hrl}"
export WANDB_ENTITY="${WANDB_ENTITY:-thakk100-dhyan-home}"
export OMNI_KIT_ACCEPT_EULA=YES

mkdir -p "$WS/logs"
tmux kill-session -t k1_train 2>/dev/null || true
tmux new-session -d -s k1_train -x 220 -y 50

tmux send-keys -t k1_train "cd $WS" Enter
tmux send-keys -t k1_train "scripts/train_guard.sh --name $NAME -- \
  $PHASE6_VENV/bin/python $TRAIN \
  --task $TASK --num_envs $NUM_ENVS --max_iterations $MAX_ITERS --seed $SEED --viz none" Enter

echo "Session 'k1_train' started."
echo "  Task:   $TASK   envs=$NUM_ENVS  iters=$MAX_ITERS  name=$NAME"
echo "  WandB:  https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "  Attach: tmux attach -t k1_train"
echo "  Output: tail -f $WS/logs/guard_$NAME.log"
