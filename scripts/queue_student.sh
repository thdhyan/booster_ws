#!/usr/bin/env bash
# queue_student.sh — server-side student launcher (runs inside tmux):
#   1. wait until GPU <idx> total used-MB drops below 10 GB (foreign job gone
#      and ollama qwen3.8 idle-unloaded; kernel baseline ~0.5 GB),
#   2. fire a 16-env x 3-iter smoke through train_student.py (video wired),
#   3. wait for the smoke session to end and launch the FULL student run
#      only if its marker says GUARD_RC=0 (chain_train.sh + grep gate).
# Everything happens on the server — no client-side polling. Launch with:
#   tmux new-session -d -s k1_p1_student_queue 'scripts/queue_student.sh p1'
#
# GPU_MAX_MB=36000: runaway protection stays 12 GB below physical 48 GB, but
# tolerates a mid-run ollama qwen3.8 reload (+~10 GB) without a guard kill.
set -u
WHICH="${1:-}"
WS="$HOME/Projects/booster_ws"
cd "$WS" || exit 1
case "$WHICH" in
  p1)  GPU=1; SMK=p1_stud_smk; FULL=p1_student
       TASK=Isaac-Basic-Student-K1-v0
       TEACHER="$WS/logs/rsl_rl/p1_basic_teacher/2026-09-22_13-41-33_p1_basic_teacher/model_6498.pt"
       ENVS=256; ITERS=1500 ;;
  p2)  GPU=2; SMK=p2_stud_smk; FULL=p2_student
       TASK=Isaac-Velocity-Distill-K1-v0
       TEACHER="$WS/logs/rsl_rl/k1_velocity_teacher/2026-09-22_13-14-06/model_2999.pt"
       ENVS=512; ITERS=3000 ;;
  *) echo "usage: queue_student.sh p1|p2" >&2; exit 2 ;;
esac
[ -f "$TEACHER" ] || { echo "[queue:$WHICH] MISSING teacher $TEACHER" >&2; exit 4; }

echo "[queue:$WHICH] waiting for gpu$GPU total < 10000 MB (now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU+1))p") MB)"
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU+1))p")
  [ -n "$USED" ] && [ "$USED" -lt 10000 ] && break
  sleep 60
done
echo "[queue:$WHICH] gpu$GPU free (used=${USED}MB) -> smoke 16x3"

env CUDA_VISIBLE_DEVICES=$GPU GPU_IDX=$GPU GPU_MAX_MB=36000 MEM_MAX_GB=64 SWAP_MAX_GB=64 \
    DISK_PATH=$HOME WANDB_API_KEY="$(cat logs/.wandb_key)" \
    scripts/tmux_train.sh --name "$SMK" -- \
    python isaac_tasks/k1_velocity/scripts/train_student.py \
      --task "$TASK" --teacher_checkpoint "$TEACHER" \
      --num_envs 16 --max_iterations 3 --seed 42 \
      --headless --enable_cameras --video --video_length 500 --video_interval 4800 \
  || { echo "[queue:$WHICH] smoke launch failed" >&2; exit 3; }

echo "[queue:$WHICH] smoke running -> waiting for k1_$SMK to end, then marker gate -> full ${ENVS}x${ITERS}"
exec scripts/chain_train.sh "k1_$SMK" \
  "grep -q GUARD_RC=0 logs/tmux_$SMK.done && env CUDA_VISIBLE_DEVICES=$GPU GPU_IDX=$GPU GPU_MAX_MB=36000 MEM_MAX_GB=64 SWAP_MAX_GB=64 DISK_PATH=$HOME WANDB_API_KEY=\$(cat logs/.wandb_key) scripts/tmux_train.sh --name $FULL -- python isaac_tasks/k1_velocity/scripts/train_student.py --task $TASK --teacher_checkpoint $TEACHER --num_envs $ENVS --max_iterations $ITERS --seed 42 --headless --enable_cameras --video --video_length 1500 --video_interval 4800 || echo SMOKE-GATE-FAILED-$SMK-NO-LAUNCH-$FULL"
