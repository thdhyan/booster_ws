#!/usr/bin/env bash
# spark_sync_loop.sh — runs ON dl inside tmux k1_spark_sync.
# Pulls spark04's full-student outputs (rsl_rl runs + launcher logs) into
# logs/spark04/ so the existing k1_gdrive_sync loop can push them to GDrive.
# Server-side loop; wandb stays the live dashboard. 5-min cadence, delta-only rsync.
set -u
DEST="$HOME/Projects/booster_ws/logs/spark04"
SRC="thakk100@10.131.50.67"
RSH="ssh -i $HOME/.ssh/spark04_sync -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
mkdir -p "$DEST/rsl_rl"
echo "[spark_sync] start $(date -Is) dest=$DEST"
while :; do
  # teacher dirs on spark are thin staging copies of dl's — skip them (dl syncs the real ones)
  if rsync -a --partial --timeout=60 -e "$RSH" \
      --exclude '/p1_basic_teacher/***' --exclude '/k1_velocity_teacher/***' \
      "$SRC:Projects/booster_ws/logs/rsl_rl/" "$DEST/rsl_rl/" >> "$DEST/sync.log" 2>&1; then
    echo "[spark_sync] rsl_rl ok $(date -Is)" >> "$DEST/sync.log"
  else
    echo "[spark_sync] rsl_rl FAIL rc=$? $(date -Is)" >> "$DEST/sync.log"
  fi
  for f in p1.full.log p2.full.log; do
    rsync -a --timeout=60 -e "$RSH" "$SRC:~/$f" "$DEST/" >> "$DEST/sync.log" 2>&1 \
      || echo "[spark_sync] log $f FAIL rc=$? $(date -Is)" >> "$DEST/sync.log"
  done
  sleep 300
done
