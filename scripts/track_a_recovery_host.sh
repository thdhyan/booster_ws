#!/bin/bash
# Serialized recovery chain: P3 1900->1999, then gated P4, then gated P2.
# The P2 final recorder already waits for P2_GAIT_STUDENT_MARKER=OK.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/track_a_recovery.log"
CACHE="$HERE/k1_isaac_cache_track_a_recovery"
mkdir -p "$CACHE/p3" "$CACHE/p4" "$CACHE/p2"
: > "$LOG"

# The old P2 queue only waited for the P2 marker and would launch a duplicate
# after this recovery writes that marker.  The dedicated recorder is kept.
tmux kill-session -t k1_spark_p2_gait 2>/dev/null || true

RUN_CMD="cd '$REPO'; export WANDB_API_KEY=\$(cat logs/.wandb_key); echo P3_RECOVERY_CHAIN=START; docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx -v '$HERE/p3_recover_container.sh:/p3.sh:ro' -v '$CACHE/p3:/root/.cache' -v '$REPO:/workspace/booster_ws' nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p3.sh && grep -q 'P3_RECOVERY_MARKER=OK' scripts/p3.recover.full.train.log && echo P4_RECOVERY_CHAIN=START && rm -f scripts/p4t.soccer.log && docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e WHICH=p4t -v '$HERE/spark_soccer_container.sh:/soc.sh:ro' -v '$CACHE/p4:/root/.cache' -v '$REPO:/workspace/booster_ws' nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /soc.sh && grep -q 'SOC_FULL_MARKER=OK' scripts/p4t.soccer.log && echo P2_RECOVERY_CHAIN=START && rm -f scripts/p2_gait.log && docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx -v '$HERE/spark_p2_gait_container.sh:/p2.sh:ro' -v '$CACHE/p2:/root/.cache' -v '$REPO:/workspace/booster_ws' nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /p2.sh && grep -q 'P2_GAIT_STUDENT_MARKER=OK' scripts/p2_gait.log && echo TRACK_A_RECOVERY_MARKER=OK"
tmux new-session -d -s k1_track_a_recovery "$RUN_CMD > '$LOG' 2>&1"
sleep 2
tmux ls | grep k1_track_a_recovery
echo "Track A recovery log: $LOG"
