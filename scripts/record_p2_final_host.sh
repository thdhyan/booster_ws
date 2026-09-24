#!/bin/bash
# Wait server-side for the gait-v2 P2 final marker, record teacher+student,
# and mirror the verified MP4/poster artifacts to the reader-shared Drive folder.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/Projects/booster_ws"
LOG="$HERE/p2_final_record.log"
CACHE="$HERE/k1_isaac_cache_p2_final_record"
mkdir -p "$CACHE"
: > "$LOG"

WAIT_LOG="$REPO/scripts/p2_gait.log"
RUN_CMD="echo P2_REC_QUEUE=WAITING_FOR_MARKER; until grep -qE 'P2_GAIT_STUDENT_MARKER=(OK|FAIL)' '$WAIT_LOG' 2>/dev/null; do sleep 60; done; grep -q 'P2_GAIT_STUDENT_MARKER=OK' '$WAIT_LOG' || { echo P2_REC_ABORT=STUDENT_MARKER_FAIL; exit 32; }; docker run --rm --gpus all --user 0 --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx -v $HERE/record_p2_final_container.sh:/record.sh:ro -v $CACHE:/root/.cache -v $REPO:/workspace/booster_ws nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /record.sh >> '$LOG' 2>&1; rc=\$?; echo P2_REC_DOCKER_RC=\$rc >> '$LOG'; [ \$rc -eq 0 ] || exit \$rc; ~/.local/bin/rclone copyto '$REPO/isaac_tasks/k1_velocity/videos/p2_teacher_gait_v2.mp4' 'gdrive-thakk100:K1 Policy Play Videos — booster_ws/Track A Interim/p2_teacher_gait_v2.mp4' --log-level INFO >> '$LOG' 2>&1; ~/.local/bin/rclone copyto '$REPO/isaac_tasks/k1_velocity/videos/p2_student_gait_v2.mp4' 'gdrive-thakk100:K1 Policy Play Videos — booster_ws/Track A Interim/p2_student_gait_v2.mp4' --log-level INFO >> '$LOG' 2>&1; ~/.local/bin/rclone copyto '$REPO/isaac_tasks/k1_velocity/videos/p2_teacher_gait_v2_poster.png' 'gdrive-thakk100:K1 Policy Play Videos — booster_ws/Track A Interim/p2_teacher_gait_v2_poster.png' --log-level INFO >> '$LOG' 2>&1; ~/.local/bin/rclone copyto '$REPO/isaac_tasks/k1_velocity/videos/p2_student_gait_v2_poster.png' 'gdrive-thakk100:K1 Policy Play Videos — booster_ws/Track A Interim/p2_student_gait_v2_poster.png' --log-level INFO >> '$LOG' 2>&1; echo P2_REC_UPLOAD_MARKER=OK >> '$LOG'"

tmux new-session -d -s k1_spark_p2_final_record "$RUN_CMD"
sleep 2
tmux ls | grep k1_spark_p2_final_record
echo "P2 final recorder log: $LOG"
