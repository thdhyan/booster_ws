#!/bin/bash
# Training status for the K1 boxes (run from the laptop).
# usage: scripts/k1_train_status.sh [pattern]   pattern filters log names (default: all logs of the last 2 days)
# Per host: GPU use, tmux sessions, and for each recent train log the last iteration and key metrics.
PAT=${1:-.}
read -r -d '' REMOTE <<'EOF'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu /'
tmux ls 2>/dev/null | sed 's/^/  tmux /' || echo "  tmux: none"
for f in $(find $LOGDIR -maxdepth 1 -name '*.log' -mtime -2 2>/dev/null | grep -E "$PAT" | sort); do
  awk -v f="$(basename $f)" '
    /^wandb:/ {next}
    /Learning iteration [0-9]+\/[0-9]+/ {match($0, /[0-9]+\/[0-9]+/); it=substr($0, RSTART, RLENGTH); delete m}
    /Mean reward:|Mean episode length:|Episode_Termination\/time_out|Episode_Termination\/base_orientation|error_vel_xy|error_vel_yaw|Curriculum\/terrain_levels|Curriculum\/gated_iter|Curriculum\/survival|goal_scored|success_rate/ {
      k=$1; for (i=2;i<NF;i++) k=k" "$i; m[k]=$NF }
    /RUN_RC=|Traceback|Error:/ {err=$0}
    END { printf "  %-32s it %s\n", f, (it==""?"-":it);
          for (k in m) printf "      %-42s %s\n", k, m[k];
          if (err!="") printf "      LAST ERR/RC: %s\n", err }' "$f"
done
EOF
for h in dl zz-bw; do
  case $h in dl) LOGDIR='~/k1_logs';; zz-bw) LOGDIR=/export/scratch/thakk100/k1/logs;; esac
  echo "== $h"
  ssh -o ConnectTimeout=10 "$h" "LOGDIR=$LOGDIR PAT='$PAT' bash -s" <<<"$REMOTE" 2>&1 | grep -v "^bind \["
done
