#!/usr/bin/env bash
# =============================================================================
# K1 training container entrypoint.
#
# Responsibilities (runtime — nothing about the workspace is baked into the
# image):
#   1. Clone the workspace repo (with submodules + LFS), OR use a bind-mounted
#      source tree when K1_USE_MOUNT=1.
#   2. pip install -e the workspace training packages (booster_train, k1_velocity).
#   3. Wire up mounted logs/models dirs.
#   4. exec the training script with the passed CLI args.
#
# Env vars:
#   K1_REPO_URL     default https://github.com/thdhyan/booster_ws
#   K1_GIT_BRANCH   default main
#   K1_WORKDIR      default /workspace/booster_ws
#   K1_USE_MOUNT    1 = skip clone, use K1_WORKDIR as-is (bind-mounted source)
#   GITHUB_TOKEN    optional; enables private repo/submodule clone
#   K1_SOURCE_ROS   1 = source /opt/ros/jazzy (NEVER for Isaac processes!)
#   WANDB_API_KEY / HF_TOKEN  read directly by wandb/huggingface libs
#   K1_MOUNT_DIR    default /workspace/mounts — host bind target root
#
# Everything after the entrypoint is passed to train.py verbatim, e.g.:
#   --task Isaac-Velocity-Rough-K1-Teacher-v0 --num_envs 4096 --headless
# =============================================================================
set -euo pipefail

K1_REPO_URL="${K1_REPO_URL:-https://github.com/thdhyan/booster_ws}"
K1_GIT_BRANCH="${K1_GIT_BRANCH:-dev/phase-0}"
K1_WORKDIR="${K1_WORKDIR:-/workspace/booster_ws}"
K1_MOUNT_DIR="${K1_MOUNT_DIR:-/workspace/mounts}"
TRAIN_SCRIPT="${K1_TRAIN_SCRIPT:-isaac_tasks/k1_velocity/scripts/train.py}"

echo "[k1-entrypoint] repo=${K1_REPO_URL} branch=${K1_GIT_BRANCH}"

# ---------------------------------------------------------------- clone -----
if [ "${K1_USE_MOUNT:-0}" != "1" ]; then
    if [ -d "${K1_WORKDIR}/.git" ]; then
        echo "[k1-entrypoint] ${K1_WORKDIR} already cloned — reusing"
    else
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            git config --global \
                url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf \
                "https://github.com/"
            echo "[k1-entrypoint] GITHUB_TOKEN set — private-clone mode"
        fi
        # submodules are registered with SSH URLs (git@github.com:) — rewrite
        # to HTTPS so container clones work without SSH keys
        git config --global url."https://github.com/".insteadOf "git@github.com:"
        git clone --recurse-submodules --shallow-submodules --depth 1 \
            -b "${K1_GIT_BRANCH}" "${K1_REPO_URL}" "${K1_WORKDIR}"
        cd "${K1_WORKDIR}"
        git lfs pull || echo "[k1-entrypoint] WARN: git lfs pull failed (models/ may be pointer files)"
        # git clone exits 0 even when submodules fail — fail loudly here
        MISSING=$(git submodule status | awk '$1 ~ /^-/ {print $2}')
        if [ -n "${MISSING}" ]; then
            echo "[k1-entrypoint] FATAL: submodules not fetched: ${MISSING}" >&2
            echo "[k1-entrypoint] hint: if the repos are private, set GITHUB_TOKEN in .env" >&2
            exit 1
        fi
    fi
else
    echo "[k1-entrypoint] K1_USE_MOUNT=1 — using bind-mounted ${K1_WORKDIR}"
fi
cd "${K1_WORKDIR}"

# ------------------------------------------------- mount wiring -------------
# Host binds live under ${K1_MOUNT_DIR}; link them into the clone AFTER it
# exists (docker mounts are created before the entrypoint runs, so they can't
# point inside a directory that git clone creates afterwards).
for name in logs models; do
    if [ -d "${K1_MOUNT_DIR}/${name}" ] && [ ! -e "${K1_WORKDIR}/${name}" ]; then
        ln -s "${K1_MOUNT_DIR}/${name}" "${K1_WORKDIR}/${name}"
        echo "[k1-entrypoint] linked ${name}/ -> ${K1_MOUNT_DIR}/${name}"
    fi
done

# ------------------------------------------------- workspace packages -------
PY="$(cat /etc/k1_python 2>/dev/null || echo python3)"
echo "[k1-entrypoint] python: ${PY}"
# -e (editable) is REQUIRED and applies per-command: `pip install -e A B C`
# would install only A editable! booster_train's setup.py lists only the top
# package (subpackages resolve via the source path), and booster_assets
# computes BOOSTER_ASSETS_DIR from __file__ — both break when non-editable.
$PY -m pip install --no-cache-dir -q -e isaac_tasks/booster_train_ref/source/booster_train
$PY -m pip install --no-cache-dir -q -e isaac_tasks/k1_velocity
$PY -m pip install --no-cache-dir -q -e src/k1_description/assets
$PY -c "import booster_train, k1_velocity; print('[k1-entrypoint] workspace packages import OK')"

# ------------------------------------------------- optional ROS sourcing ----
# NEVER enable for Isaac training processes (RMW/env double-init).
if [ "${K1_SOURCE_ROS:-0}" = "1" ] && [ -f /opt/ros/jazzy/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    echo "[k1-entrypoint] ROS 2 jazzy sourced (non-Isaac workflow)"
fi

# ------------------------------------------------- run ----------------------
if [ "$#" -eq 0 ]; then
    set -- --task Isaac-Velocity-Rough-K1-Teacher-v0 --num_envs 4096 --headless
fi
echo "[k1-entrypoint] exec: ${PY} ${TRAIN_SCRIPT} $*"
exec "${PY}" "${TRAIN_SCRIPT}" "$@"
