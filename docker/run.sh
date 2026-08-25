#!/usr/bin/env bash
# =============================================================================
# Host-side runner for the K1 training container.
#
# Usage:
#   docker/run.sh [options] [-- extra train.py args...]
#
# Options:
#   --image NAME        image to run (default $K1_IMAGE or docker.io/$USER/k1-isaac-train:3.0.0b2)
#   --branch NAME       git branch to clone (default dev/phase-0; e.g. dev/phase-0)
#   --env-file FILE     secrets file (default docker/.env; created from .env.example if missing)
#   --source            bind-mount THIS checkout instead of cloning (K1_USE_MOUNT=1)
#   --task TASK         convenience: sets the first train.py arg
#   --num-envs N        convenience: --num_envs for train.py
#   --iters N           convenience: --max_iterations for train.py
#   --name NAME         container name (default k1-train-<ts>)
#   --network-host      use host networking (wandb behind VPN/proxy)
#   -h | --help
#
# Examples:
#   docker/run.sh --task Isaac-Velocity-Rough-K1-Teacher-v0 --num-envs 4096 --iters 5000
#   docker/run.sh --branch dev/phase-0 --source -- --task Isaac-Velocity-Distill-K1-v0 \
#       --teacher_checkpoint logs/rsl_rl/k1_velocity_teacher/run/model_4999.pt
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # workspace root

IMAGE="${K1_IMAGE:-thdhyan/k1-isaac-train:3.0.0b2}"
BRANCH="dev/phase-0"
ENV_FILE="docker/.env"
SOURCE_MOUNT=0
CONT_NAME="k1-train-$(date +%m%d-%H%M%S)"
NET_ARGS=()
TASK="" ; NUM_ENVS="" ; ITERS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE="$2"; shift 2 ;;
        --branch)       BRANCH="$2"; shift 2 ;;
        --env-file)     ENV_FILE="$2"; shift 2 ;;
        --source)       SOURCE_MOUNT=1; shift ;;
        --task)         TASK="$2"; shift 2 ;;
        --num-envs)     NUM_ENVS="$2"; shift 2 ;;
        --iters)        ITERS="$2"; shift 2 ;;
        --name)         CONT_NAME="$2"; shift 2 ;;
        --network-host) NET_ARGS=(--network host); shift ;;
        -h|--help)      grep "^#" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --)             shift; EXTRA=("$@"); break ;;
        *)              EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    cp docker/.env.example "$ENV_FILE"
    echo "created $ENV_FILE — add WANDB_API_KEY / HF_TOKEN / GITHUB_TOKEN, then rerun."
    exit 1
fi

TRAIN_ARGS=()
[[ -n "$TASK" ]]     && TRAIN_ARGS+=(--task "$TASK")
[[ -n "$NUM_ENVS" ]] && TRAIN_ARGS+=(--num_envs "$NUM_ENVS")
[[ -n "$ITERS" ]]    && TRAIN_ARGS+=(--max_iterations "$ITERS")
TRAIN_ARGS+=("${EXTRA[@]}")

mkdir -p logs models

DOCKER_ARGS=(
    --gpus all
    --rm
    --name "$CONT_NAME"
    --env-file "$ENV_FILE"
    -e K1_GIT_BRANCH="$BRANCH"
    -v "$PWD/logs:/workspace/mounts/logs"
    -v "$PWD/models:/workspace/mounts/models"
    "${NET_ARGS[@]}"
)

if [[ "$SOURCE_MOUNT" == 1 ]]; then
    DOCKER_ARGS+=( -e K1_USE_MOUNT=1 -v "$PWD:/workspace/booster_ws" )
fi

echo "image:    $IMAGE"
echo "branch:   $BRANCH   source-mount: $SOURCE_MOUNT"
echo "train.py: ${TRAIN_ARGS[*]:-(image defaults)}"
echo
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${TRAIN_ARGS[@]}"
