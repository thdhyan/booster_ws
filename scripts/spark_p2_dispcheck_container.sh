#!/bin/bash
# Does the P2 policy actually translate, or does it just stand there?
#
# WHY THIS EXISTS
# ---------------
# A 3000-iteration run reached mean reward ~38 and episode-length ratio ~0.51, and
# that was read as "it learned to walk" because the reward matched an earlier
# verified walker's 36.61. Then a 4-panel video of the checkpoint at vx=0.6 showed
# it upright but with no clear displacement, and two frames cannot settle that.
# Reward parity with a known-good run is not a measurement of walking.
#
# So measure it. play_record.py writes root_pos (world frame) to an .npz, which is
# the quantity that actually defines walking. This runs the SAME checkpoint at two
# speeds and reports displacement for each:
#
#   0.5 m/s  inside the training range -- if it does not walk here, the problem is
#            upstream in the reward or the gains, not in the command range
#   0.6 m/s  just outside it            -- tests whether the range ceiling is real
#
# The pass criterion is a movement gate, deliberately not reward: >0.5 m of net
# displacement counts as walking, matching the rule used for the P2 videos.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo DISP_CD_FAIL; exit 1; }

TASK=Isaac-Velocity-Rough-K1-Teacher-v0
CKPT="$1"
if [ -z "$CKPT" ] || [ ! -s "$CKPT" ]; then
  echo "DISP_CKPT_MISSING=$CKPT"
  exit 1
fi
echo "DISP_CKPT=$CKPT"

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -1

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$REPO/isaac_tasks/k1_velocity/videos"
mkdir -p "$OUTDIR"

run_speed() {   # $1 = speed, $2 = tag
  local v="$1" tag="$2"
  local trace="$OUTDIR/p2_disp_${tag}_${STAMP}.npz"
  echo "=== DISP PLAY cmd vx=$v ($tag)"
  timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --num_envs 8 --steps 750 --cmd "$v" 0.0 0.0 \
    --trace_out "$trace" \
    --label "P2 displacement check | cmd vx=$v | $tag" \
    > "$REPO/scripts/p2_disp_${tag}.log" 2>&1
  local rc=$?
  echo "DISP_PLAY_${tag}_RC=$rc"
  if [ ! -s "$trace" ]; then
    echo "DISP_TRACE_MISSING_${tag}"
    return 1
  fi
  "$PY" - "$trace" "$v" "$tag" <<'PYEOF'
import sys
import numpy as np

path, speed, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = np.load(path)
if "root_pos" not in d:
    print(f"DISP_NO_ROOT_POS {tag}")
    raise SystemExit(1)
pos = d["root_pos"]                      # (steps, num_envs, 3) world frame
xy = pos[..., :2]
net = np.linalg.norm(xy[-1] - xy[0], axis=-1)      # straight-line displacement
seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
pathlen = seg.sum(axis=0)
vel = d["root_lin_vel"][..., :2] if "root_lin_vel" in d else None
# Time actually simulated: control steps at ~50 Hz, matching --steps 750 (~15 s).
dt = 0.02
mean_speed = pathlen / (len(xy) * dt)

print(f"DISP_RESULT tag={tag} cmd_vx={speed:+.2f} envs={xy.shape[1]} steps={xy.shape[0]}")
print(f"  net displacement  mean={net.mean():.3f} m  min={net.min():.3f}  max={net.max():.3f}")
print(f"  path length       mean={pathlen.mean():.3f} m  max={pathlen.max():.3f}")
print(f"  mean speed        mean={mean_speed.mean():.3f} m/s  max={mean_speed.max():.3f}")
if vel is not None:
    sp = np.linalg.norm(vel, axis=-1).mean(axis=0)
    print(f"  lin vel (logged)  mean={sp.mean():.3f} m/s  max={sp.max():.3f}")
if "dones" in d:
    dones = d["dones"]
    fell = dones[:, 0] if dones.ndim > 1 else dones
    print(f"  episode ends in run: {int(fell.sum())} of {len(fell)} steps for env 0")
gate = net.mean() > 0.5
print(f"  MOVEMENT_GATE={tag}: {'PASS' if gate else 'FAIL'} (net {net.mean():.3f} m vs 0.5 m bar)")
PYEOF
  return 0
}

run_speed 0.5 inrange
run_speed 0.6 aboverange

echo "DISP_ALL_DONE"
exit 0
