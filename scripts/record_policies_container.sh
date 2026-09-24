#!/bin/bash
# REC (container-side): play + record finished checkpoints with play_record.py
# (HUD overlay: commands / obs bars / action bars / rewards) -> mp4 + npz trace
# + TorchScript export under models/. Runs INSIDE
# nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 via record_policies_host.sh.
# SKIP_CORE=1 records only the partial-control video (Run-10 ckpt, pulled later).
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || { echo REC_CD_FAIL; exit 1; }
export WANDB_API_KEY="$(cat "$REPO/logs/.wandb_key" 2>/dev/null)"

echo "=== REC INSTALL booster_train (rsl_rl fork)"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train 2>&1 | tail -3
echo "=== REC INSTALL k1_velocity"
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity 2>&1 | tail -3
echo "=== REC INSTALL booster_assets (K1 URDF package)"
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets 2>&1 | tail -3
echo "=== REC video libs (pillow/imageio)"
"$PY" -m pip install -q pillow imageio imageio-ffmpeg 2>&1 | tail -2
"$PY" -c "import PIL, imageio, imageio_ffmpeg; print('VIDEO_LIBS_OK', PIL.__version__, imageio.__version__)" \
  || { echo REC_LIBS_FAIL; exit 4; }

OUT="$REPO/isaac_tasks/k1_velocity/videos"
mkdir -p "$OUT"
FAILS=0

rec() {
  # rec NAME TASK CKPT EYE LOOKAT EXPORT LABEL CMD("vx vy wz" or -)
  local name="$1" task="$2" ckpt="$3" eye="$4" lookat="$5" export="$6" label="$7" cmd="$8"
  if [ ! -f "$ckpt" ]; then
    echo "REC_CKPT_MISSING_$name=$ckpt"
    FAILS=$((FAILS+1))
    return 40
  fi
  local cmdargs=()
  if [ "$cmd" != "-" ]; then read -ra cmdargs <<< "--cmd $cmd"; fi
  echo "=== REC $name  task=$task"
  "$PY" isaac_tasks/k1_velocity/scripts/play_record.py \
    --task "$task" --checkpoint "$ckpt" \
    --num_envs 4 --steps 750 --headless \
    --eye "$eye" --lookat "$lookat" \
    --video_out "$OUT/$name.mp4" --trace_out "$OUT/${name}_trace.npz" \
    --export "$export" --label "$label" \
    ${cmdargs[@]+"${cmdargs[@]}"}
  local rc=$?
  echo "REC_RC_$name=$rc"
  # Artifact verification — python.sh's exit code proved unreliable (a crashed
  # run reported rc=0), so require the actual outputs: 750 frames @1024x576 is
  # well above 200 KB (a step-0 crash leaves a ~10 KB header-only mp4).
  local sz trace_ok exp_ok
  sz=$(stat -c%s "$OUT/$name.mp4" 2>/dev/null || echo 0)
  trace_ok=no; [ -f "$OUT/${name}_trace.npz" ] && trace_ok=yes
  exp_ok=no; [ -f "$export" ] && exp_ok=yes
  if [ "$rc" -ne 0 ] || [ "$sz" -lt 200000 ] || [ "$trace_ok" != yes ] || [ "$exp_ok" != yes ]; then
    echo "REC_VERIFY_FAIL_$name rc=$rc mp4=${sz}B trace=$trace_ok export=$exp_ok"
    FAILS=$((FAILS+1))
  else
    echo "REC_VERIFY_OK_$name mp4=${sz}B"
  fi
}

# camera: circle-ish framing for walking tasks (eye/lookat world coords)
CAM_STAND="3.0,-3.0,1.7|0,0,0.55"
CAM_WALK="5.5,-5.5,3.0|0,0,0.5"
IFS='|' read -r SEYE SLOOK <<< "$CAM_STAND"
IFS='|' read -r WEYE WLOOK <<< "$CAM_WALK"

if [ "${SKIP_CORE:-0}" != "1" ]; then
  # 1) P1 teacher (stand, rough terrain, active shoves)
  rec p1_teacher_stand Isaac-Basic-Teacher-K1-v0 \
      "$REPO/logs/rsl_rl/p1_basic_teacher/2026-09-22_13-41-33_p1_basic_teacher/model_6498.pt" \
      "$SEYE" "$SLOOK" "$REPO/models/p1_basic_teacher.pt" "P1 teacher · model_6498" "-"
  # 2) P1 student (blind 420-dim distill, deployed export)
  rec p1_student_stand Isaac-Basic-Student-K1-v0 \
      "$REPO/logs/rsl_rl/p1_basic_student/2026-09-23_16-20-36_p1_basic_student/model_1499.pt" \
      "$SEYE" "$SLOOK" "$REPO/models/p1_basic_student.pt" "P1 student · model_1499" "-"
  # 3) P2 teacher (rough terrain velocity, circle cmd keeps robot in frame)
  rec p2_teacher_rough Isaac-Velocity-Rough-K1-Teacher-v0 \
      "$REPO/logs/rsl_rl/k1_velocity_teacher/2026-09-22_13-14-06/model_2999.pt" \
      "$WEYE" "$WLOOK" "$REPO/models/p2_move_teacher.pt" "P2 teacher · model_2999" "0.6 0.0 0.5"
  # 4) P2 student (distill play, flat, circle cmd)
  rec p2_student_walk Isaac-Velocity-Distill-K1-Play-v0 \
      "$REPO/logs/rsl_rl/p2_move_student/2026-09-23_13-46-54/model_2999.pt" \
      "$WEYE" "$WLOOK" "$REPO/models/p2_move_student.pt" "P2 student · model_2999" "0.6 0.0 0.5"
fi

# 5) partial-control (Run-10) — recorded once its final ckpt has been pulled
PARTIAL_CKPT="$(ls -t "$REPO"/logs/rsl_rl/k1_partialctrl_base/*/model_2999.pt 2>/dev/null | head -1)"
if [ -n "$PARTIAL_CKPT" ]; then
  rec partial_walk Isaac-Velocity-PartialCtrl-K1-Play-v0 \
      "$PARTIAL_CKPT" "$WEYE" "$WLOOK" "$REPO/models/k1_partialctrl.pt" \
      "Partial-ctrl base · model_2999" "0.6 0.0 0.5"
else
  echo "REC_SKIP_partial (no logs/rsl_rl/k1_partialctrl_base/*/model_2999.pt yet)"
fi

echo "REC_FAILS=$FAILS"
ls -la "$OUT"/*.mp4 2>/dev/null
ls -la "$REPO"/models/*.pt 2>/dev/null
echo "REC_ALL_DONE"
exit "$FAILS"
