#!/bin/bash
# =============================================================================
# K1 training runner — cs-zhang-net-01 (SingularityCE 4.1.1, no sudo, proot).
# Follows the proven cosmos3-gen pattern (~/Projects/cosmos3-gen/COSMOS-bw.MD):
#   - staging/cache MUST be local ext4 (/export/scratch), never NFS/home
#   - home quota is tiny -> container gets no host home; HOME -> scratch bind
#   - durable xtra quota was full (2026-08-25) -> outputs also live on scratch;
#     rsync final checkpoints off the box when the run finishes.
#
# One-time build (tmux, 20-40 min under proot; needs k1_train.def +
# entrypoint.sh in the same dir):
#   export PATH="$HOME/bin:$PATH" SCRATCH=/export/scratch/thakk100
#   export SINGULARITY_TMPDIR=$SCRATCH/sing-tmp SINGULARITY_CACHEDIR=$SCRATCH/sing-cache
#   cd ~/k1build && singularity build $SCRATCH/k1-train.sif k1_train.def
#
# Teacher (default task baked in image):
#   ~/run_k1_train.sh
#   # = --task Isaac-Velocity-Rough-K1-Teacher-v0 --num_envs 4096 --headless
#
# Student distillation (after teacher ckpt exists):
#   K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/train_student.py \
#   ~/run_k1_train.sh --task Isaac-Velocity-Distill-K1-v0 \
#       --teacher_checkpoint /workspace/mounts/logs/rsl_rl/<run>/model_4999.pt
#
# Smoke test:
#   ~/run_k1_train.sh --task Isaac-Velocity-Rough-K1-Teacher-v0 \
#       --num_envs 64 --headless --max_iterations 2
# =============================================================================
set -euo pipefail

SCRATCH=/export/scratch/thakk100
SIF=$SCRATCH/k1-train.sif
RUN=$SCRATCH/k1                 # local NVMe ext4: clone(/tmp) + home + outputs
GPU=${K1_GPU:-1}                # GPU1 shared w/ own cosmos3 kernel (decision 2026-08-25)

mkdir -p "$RUN/tmp" "$RUN/home" "$RUN/logs" "$RUN/models"

# singularity build-time dirs must be local ext4 (NFS xattr failure mode)
export SINGULARITY_TMPDIR=$SCRATCH/sing-tmp
export SINGULARITY_CACHEDIR=$SCRATCH/sing-cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

[ -f "$SIF" ] || { echo "SIF missing: $SIF — build it first (see header)" >&2; exit 1; }
[ -f "$HOME/.config/k1/secrets.env" ] || { echo "missing ~/.config/k1/secrets.env" >&2; exit 1; }

exec singularity run --nv --containall \
  --bind "$RUN/tmp:/tmp" \
  --bind "$RUN/home:/k1home" \
  --bind "$RUN/logs:/workspace/mounts/logs" \
  --bind "$RUN/models:/workspace/mounts/models" \
  --env HOME=/k1home \
  --env CUDA_VISIBLE_DEVICES="$GPU" \
  --env-file "$HOME/.config/k1/secrets.env" \
  "$SIF" "$@"
