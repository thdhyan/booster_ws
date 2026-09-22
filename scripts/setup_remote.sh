#!/usr/bin/env bash
# Idempotent Phase-6 bring-up on a REMOTE server box (dl, …). Run as the remote
# user, headless servers only:
#
#   # bootstrap before the repo exists (script lives in the repo):
#   ssh dl 'bash -s' < scripts/setup_remote.sh
#   # or re-run after clone:
#   ssh dl 'bash $HOME/Projects/booster_ws/scripts/setup_remote.sh'
#
# Creates/repairs, in order (each step skips if already present):
#   1. uv (+ git-lfs hint)
#   2. $HOME/Projects/booster_ws clone + submodules + LFS assets
#   3. $HOME/Projects/IsaacLab worktree at tag v3.0.0-EA (== laptop's ae37b028e)
#   4. .venv via scripts/install_phase6_env.sh  (~30 GB incl. Isaac Sim 6.1.0 pip)
#   5. import sanity checks (isaaclab / rsl_rl / ultralytics)
# Launch afterwards via scripts/tmux_train.sh with the guard env from TRAINING.md
# (GPU_IDX = CUDA_VISIBLE_DEVICES, GPU_MAX_MB, MEM_MAX_GB, DISK_PATH=$HOME).
# Checkpoint handoff (from laptop):  scp model_1500.pt dl:Projects/booster_ws/logs/...
set -euo pipefail
PROJ="$HOME/Projects"
WS="$PROJ/booster_ws"
IL="$PROJ/IsaacLab"
IL_EA="$PROJ/IsaacLab-ea"
export OMNI_KIT_ACCEPT_EULA=YES UV_HTTP_TIMEOUT=900

echo "[setup] host=$(hostname) arch=$(uname -m)"

# 1. tools
export PATH="$HOME/.local/bin:$PATH"   # uv installer target; also skips re-install
if ! command -v uv >/dev/null; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }
command -v git-lfs >/dev/null || echo "WARN: git-lfs missing — LFS assets would be pointer files"
git lfs install >/dev/null 2>&1 || true

# 2. workspace
if [ ! -d "$WS/.git" ]; then
    echo "[setup] cloning booster_ws"
    git clone https://github.com/thdhyan/booster_ws.git "$WS"
fi
cd "$WS"
git fetch origin >/dev/null 2>&1 || true
# Submodule URLs are git@github.com: SSH and dl has no GitHub key; all three
# submodule repos are PUBLIC, so rewrite to https (dl had no working github ssh
# anyway — no regression). Global on purpose: applies to the submodule clones.
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init --recursive
echo "[setup] git lfs pull (assets)"
git lfs pull || echo "WARN: lfs pull failed — check connectivity/git-lfs"

# 3. IsaacLab v3.0.0-EA worktree (matches laptop; no local patches exist there).
#    Tolerates a pre-existing clean clone (dl had ~/Projects/IsaacLab @ beta2).
if [ ! -d "$IL_EA" ]; then
    if [ ! -d "$IL/.git" ]; then
        echo "[setup] cloning IsaacLab"
        git clone https://github.com/isaac-sim/IsaacLab.git "$IL"
    fi
    echo "[setup] fetch v3.0.0-EA tag + worktree"
    git -C "$IL" fetch origin tag v3.0.0-EA --no-tags
    git -C "$IL" worktree add "$IL_EA" v3.0.0-EA
fi

# 4. venv (uv sync with isaacsim extras; re-runable — sync strips pip-added extras)
echo "[setup] install_phase6_env.sh"
bash "$WS/scripts/install_phase6_env.sh"

# 5. sanity
echo "[setup] sanity checks"
# shellcheck disable=SC1091
source "$WS/scripts/phase6_env.sh"
isaaclab --help >/dev/null
python - <<'PY'
import rsl_rl, ultralytics  # noqa: F401
import isaaclab  # noqa: F401
print("[setup] imports OK: isaaclab, rsl_rl, ultralytics")
PY

echo "[setup] DONE — next:"
echo "  scp <laptop>:$WS/logs/rsl_rl/p1_basic_teacher/2026-09-22_08-59-01_p1_basic_teacher/model_1500.pt $WS/logs/rsl_rl/p1_basic_teacher/2026-09-22_08-59-01_p1_basic_teacher/"
echo "  ssh -t $(hostname)   # then scripts/tmux_train.sh per TRAINING.md"
