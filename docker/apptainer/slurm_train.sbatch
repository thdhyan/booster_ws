#!/bin/bash
#SBATCH --job-name=k1-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# K1 Isaac Lab training via Apptainer SIF.
# Submit:  sbatch docker/apptainer/slurm_train.sbatch
# Override (example):
#   sbatch --export=ALL,K1_TASK=Isaac-Velocity-Rough-K1-Teacher-v0,K1_ITERS=5000 \
#     docker/apptainer/slurm_train.sbatch

set -euo pipefail

TASK="${K1_TASK:-Isaac-Velocity-Rough-K1-Teacher-v0}"
NUM_ENVS="${K1_NUM_ENVS:-4096}"
ITERS="${K1_ITERS:-5000}"
BRANCH="${K1_GIT_BRANCH:-dev/phase-0}"
SIF="${K1_SIF:-$SLURM_SUBMIT_DIR/k1-train.sif}"

SCRATCH="${K1_SCRATCH:-/scratch/global/$USER}"   # adjust per cluster
WORK="$SCRATCH/k1_container"
mkdir -p "$WORK/tmp" "$WORK/home" "$SLURM_SUBMIT_DIR/logs" "$SLURM_SUBMIT_DIR/models"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

echo "== job $SLURM_JOB_ID node $(hostname) =="
nvidia-smi -L || true

apptainer run --nv --cleanenv \
  --bind "$SLURM_SUBMIT_DIR/logs:/workspace/mounts/logs" \
  --bind "$SLURM_SUBMIT_DIR/models:/workspace/mounts/models" \
  --bind "$WORK/tmp:/tmp" \
  --home "$WORK/home:/home/$USER" \
  --env K1_GIT_BRANCH="$BRANCH" \
  --env K1_WORKDIR=/tmp/booster_ws \
  --env-file "$SLURM_SUBMIT_DIR/docker/.env" \
  "$SIF" \
  --task "$TASK" --num_envs "$NUM_ENVS" --headless --max_iterations "$ITERS"
