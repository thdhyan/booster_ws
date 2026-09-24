#!/bin/bash
# HP (hierarchical partial-control) container-side launcher: runs INSIDE
# nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via spark_partial_host.sh.
#   1. install our 3 editable pkgs (mirror of spark_full_container.sh),
#   2. SMOKE 16 envs x 3 iters of Isaac-Velocity-PartialCtrl-K1-v0 (video wired),
#   3. gate: FULL run 512x3000 only if the smoke exits 0.
# Everything happens server-side in one tmux session; wandb = live dashboard.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo "HP_CD_FAIL"; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_KEY_LOADED"; else echo "WANDB_KEY_MISSING"; fi

TASK=Isaac-Velocity-PartialCtrl-K1-v0

echo "=== HP INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== HP INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== HP INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3

echo "=== HP SMOKE $TASK 16x3"
timeout 3600 "$PY" isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 16 --max_iterations 3 --seed 42 \
  --headless --video --video_length 500 --video_interval 4800
SMOKE_RC=$?
echo "HP_SMOKE_RC=$SMOKE_RC"
if [ "$SMOKE_RC" -ne 0 ]; then
    echo "HP_GATE=SMOKE_FAILED_NO_FULL_RUN"
    exit 3
fi

echo "=== HP FULL $TASK 512x3000 (timeout 28800s)"
timeout 28800 "$PY" isaac_tasks/k1_velocity/scripts/train.py \
  --task "$TASK" --num_envs 512 --max_iterations 3000 --seed 42 \
  --headless --video --video_length 1500 --video_interval 4800
FULL_RC=$?
echo "HP_FULL_RC=$FULL_RC"
echo "HP_CONTAINER_DONE FULL_RC=$FULL_RC"
exit $FULL_RC
