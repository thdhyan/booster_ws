#!/bin/bash
# Start one HRL Stage-0 / skill training run on dl in tmux (run from the laptop).
# usage:
#   scripts/k1_launch_hrl.sh teacher <preset 0-3> <gpu>   # Isaac-LowerBody-V4-K1-v0, PBT member P<preset>
#   scripts/k1_launch_hrl.sh steps <gpu>                  # Isaac-Steps-Residual-K1-v0 (height scan)
#   scripts/k1_launch_hrl.sh steps_depth <gpu>            # Isaac-Steps-Depth-Residual-K1-v0 (depth camera)
#   scripts/k1_launch_hrl.sh walk <gpu>                   # Isaac-LowerBody-Walk-K1-v0 (GEAR-style, freq command)
#   scripts/k1_launch_hrl.sh stand <gpu>                  # Isaac-LowerBody-Stand-K1-v0 (GEAR-style stand)
# Allowed GPUs: dl 0/1 only (shared box). Watch with scripts/k1_train_status.sh.
# Files must already be synced to dl:~/k1_run_ws (rsync, as in HANDOFF-PUSH.md).
set -euo pipefail
WHAT=$1; shift
case $WHAT in
  teacher) P=$1; G=$2; S=lbv4_p$P; ENV="K1_TEACHER_PRESET=$P"; TASK=Isaac-LowerBody-V4-K1-v0 ;;
  steps)   G=$1; S=steps_res; ENV=""; TASK=Isaac-Steps-Residual-K1-v0 ;;
  steps_depth) G=$1; S=steps_depth; TASK=Isaac-Steps-Depth-Residual-K1-v0
           ENV="K1_ENABLE_CAMERAS=1"; N=1024 ;;
  walk)    G=$1; S=lb_walk; ENV=""; TASK=Isaac-LowerBody-Walk-K1-v0 ;;
  stand)   G=$1; S=lb_stand; ENV=""; TASK=Isaac-LowerBody-Stand-K1-v0 ;;
  *) echo "unknown run: $WHAT" >&2; exit 1 ;;
esac
N=${N:-4096}
[[ $G == 0 || $G == 1 ]] || { echo "dl GPU must be 0 or 1" >&2; exit 1; }
L="\$HOME/k1_logs/$S.log"
ssh dl "tmux has-session -t $S 2>/dev/null && { echo 'session $S already running'; exit 1; }
tmux new-session -d -s $S \"$ENV K1_GPU=$G \$HOME/run_k1_train_dl.sh --task $TASK --num_envs $N --viz none > $L 2>&1; echo RUN_RC=\\\$? >> $L\"
echo LAUNCHED session=$S gpu=$G log=$L"
