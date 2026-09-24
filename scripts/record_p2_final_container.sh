#!/bin/bash
# Record final gait-v2 P2 teacher + deployable student after their final marker.
# The host launcher waits for P2_GAIT_STUDENT_MARKER=OK before starting this.
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo P2_REC_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/p2_rec_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/p2_rec_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/p2_rec_install.log 2>&1
"$PY" -m pip install -q pillow imageio imageio-ffmpeg >>/tmp/p2_rec_install.log 2>&1

latest_ckpt() {
  find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}
TEACHER=$(latest_ckpt "$REPO/logs/rsl_rl/p2_move_teacher" model_2999.pt)
STUDENT=$(latest_ckpt "$REPO/logs/rsl_rl/p2_move_student" model_2999.pt)
if [ -z "$TEACHER" ] || [ ! -s "$TEACHER" ] || [ -z "$STUDENT" ] || [ ! -s "$STUDENT" ]; then
  echo "P2_REC_CHECKPOINT_MISSING teacher=$TEACHER student=$STUDENT"
  exit 2
fi

OUT="$REPO/isaac_tasks/k1_velocity/videos"
mkdir -p "$OUT" "$REPO/models"
run_one() {
  local name="$1" task="$2" ckpt="$3" export_path="$4" label="$5"
  echo "=== P2_REC $name checkpoint=$ckpt"
  "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
    --task "$task" --checkpoint "$ckpt" --num_envs 4 --steps 750 --headless \
    --eye 5.5,-5.5,3.0 --lookat 0,0,0.5 --cmd 0.6 0 0 \
    --video_out "$OUT/${name}.mp4" --trace_out "$OUT/${name}_trace.npz" \
    --export "$export_path" --label "$label"
  local rc=$?
  echo "P2_REC_RC_${name}=$rc"
  return "$rc"
}

run_one p2_teacher_gait_v2 Isaac-Velocity-Rough-K1-Teacher-v0 "$TEACHER" \
  "$REPO/models/p2_move_teacher_gait_v2.pt" "P2 teacher · gait-v2 · model_2999" || exit 10
run_one p2_student_gait_v2 Isaac-Velocity-Distill-K1-Play-v0 "$STUDENT" \
  "$REPO/models/p2_move_student_gait_v2.pt" "P2 student · gait-v2 · model_2999" || exit 11

for name in p2_teacher_gait_v2 p2_student_gait_v2; do
  ffmpeg -y -ss 7 -i "$OUT/$name.mp4" -frames:v 1 "$OUT/${name}_poster.png" >/tmp/p2_rec_ffmpeg_${name}.log 2>&1
  sz=$(stat -c%s "$OUT/$name.mp4" 2>/dev/null || echo 0)
  trace="$OUT/${name}_trace.npz"
  if [ "$sz" -lt 200000 ] || [ ! -s "$trace" ] || [ ! -s "$OUT/${name}_poster.png" ]; then
    echo "P2_REC_ARTIFACT_FAIL_${name} size=${sz}B"
    exit 12
  fi
done

# Do not upload a static policy as a gait result.  The trace check is a
# conservative displacement gate; the final visual review remains on the clips.
"$PY" - <<'PY'
import numpy as np
from pathlib import Path
for name in ("p2_teacher_gait_v2", "p2_student_gait_v2"):
    d = np.load(Path("isaac_tasks/k1_velocity/videos") / f"{name}_trace.npz")
    assert all(np.isfinite(d[k]).all() for k in d.files), f"non-finite trace: {name}"
    p = d["root_pos"]
    disp = np.linalg.norm(p[-1, :, :2] - p[0, :, :2], axis=1)
    print(f"P2_REC_MOVE {name} env_displacement={disp.tolist()}")
    assert float(disp.max()) > 0.5, f"no gait displacement: {name} {disp.tolist()}"
print("P2_REC_MOVE_CHECK=OK")
PY
if [ "$?" -ne 0 ]; then
  echo P2_REC_MOVE_CHECK=FAIL
  exit 13
fi

echo "P2_REC_ALL_DONE"
exit 0
