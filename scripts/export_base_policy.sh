#!/bin/bash
# Export the frozen base policy (latest partial-control ckpt) to TorchScript
# for the box-push env's FrozenBaseVelocityAction. Run inside the Isaac image.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
CKPT=$(ls -t logs/rsl_rl/k1_partialctrl_base/*/model_*.pt 2>/dev/null | head -1)
echo "BASE_CKPT=$CKPT"
[ -n "$CKPT" ] || { echo "NO_BASE_CKPT"; exit 1; }
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
$PY isaac_tasks/k1_velocity/scripts/play_record.py \
  --task Isaac-Velocity-PartialCtrl-K1-Play-v0 \
  --checkpoint "$CKPT" \
  --num_envs 2 --steps 2 --headless \
  --export models/k1_partialctrl_base.pt \
  --trace_out /tmp/ignore_trace.npz
echo "BASE_EXPORT_RC=$?"
ls -la models/k1_partialctrl_base.pt 2>/dev/null
echo "BASE_EXPORT_DONE"
