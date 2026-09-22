#!/usr/bin/env bash
# Launch the dl parallel batch:  run ON dl  (ssh dl 'bash -s' < this).
#   1. rclone folder gdrive-dhyan:Booster/logs + persistent sync tmux session
#      (mirrors $WS/logs — guard logs, checkpoints, videos — every 10 min)
#   2. P2 velocity teacher 512x3000 on GPU2  (fresh, no ckpt dependency)
#   3. supervisor tmux session: chains the P1 EXTENSION (512 envs, 3000 iters,
#      newest checkpoint) onto GPU1 after the current P1 finish run ends.
# WandB is the progress dashboard — no client-side polling by design.
set -euo pipefail
WS="$HOME/Projects/booster_ws"
KEYFILE="$WS/logs/.wandb_key"
cd "$WS"
git pull -q
export PATH="$HOME/.local/bin:$PATH"
COMMON="GPU_MAX_MB=30000 MEM_MAX_GB=64 SWAP_MAX_GB=64 DISK_PATH=$HOME"
[ -s "$KEYFILE" ] || { echo "ERROR: missing $KEYFILE" >&2; exit 1; }
mkdir -p "$WS/logs"

# --- 1. GDrive sync ---------------------------------------------------------
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q '^gdrive-dhyan:'; then
    rclone mkdir gdrive-dhyan:Booster/logs 2>/dev/null || true
    if ! tmux has-session -t k1_gdrive_sync 2>/dev/null; then
        tmux new-session -d -s k1_gdrive_sync \
            "while :; do rclone copy --transfers 4 --checkers 8 --exclude .wandb_key '$WS/logs' gdrive-dhyan:Booster/logs --log-level ERROR; sleep 600; done; exec bash"
        echo "gdrive sync session: k1_gdrive_sync -> gdrive-dhyan:Booster/logs (10 min cadence)"
    fi
else
    echo "WARN: no gdrive-dhyan rclone remote — skipping GDrive sync" >&2
fi

# --- 2. P2 teacher on GPU2 (fresh, independent of P1) -----------------------
if ! tmux has-session -t k1_p2_teacher_dl 2>/dev/null; then
    env CUDA_VISIBLE_DEVICES=2 GPU_IDX=2 $COMMON \
        WANDB_API_KEY="$(cat "$KEYFILE")" \
        scripts/tmux_train.sh --name p2_teacher_dl -- isaaclab train --rl_library rsl_rl \
          --task Isaac-Velocity-Rough-K1-Teacher-v0 --external_callback k1_velocity.register_tasks.register_tasks \
          --num_envs 512 --max_iterations 3000 --seed 42 \
          --headless --enable_cameras --video --video_length 1500 --video_interval 2457600
fi

# --- 3. P1 extension supervisor (chains after the current P1 run) -----------
if ! tmux has-session -t k1_p1_chain 2>/dev/null; then
    tmux new-session -d -s k1_p1_chain \
        "bash '$WS/scripts/chain_train.sh' k1_p1_teacher_dl 'env CUDA_VISIBLE_DEVICES=1 GPU_IDX=1 $COMMON WANDB_API_KEY=\$(cat $KEYFILE) $WS/scripts/tmux_train.sh --name p1_teacher_dl_ext -- isaaclab train --rl_library rsl_rl --task Isaac-Basic-Teacher-K1-v0 --external_callback k1_velocity.register_tasks.register_tasks --num_envs 512 --max_iterations 3000 --seed 42 --checkpoint \$(ls -t $WS/logs/rsl_rl/p1_basic_teacher/*/model_*.pt | head -1) --headless --enable_cameras --video --video_length 1500 --video_interval 2457600'"
    echo "supervisor session: k1_p1_chain (waits for k1_p1_teacher_dl, then 512x3000 ext on GPU1)"
fi

tmux ls
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
