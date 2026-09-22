#!/usr/bin/env bash
# Reproduce the Phase-6 environment from scratch (idempotent; ~30 GB, uv-cached).
# Re-run this ANY time after `uv sync` — sync removes uv-pip-added extras
# (ultralytics, imageio[ffmpeg]).
#
# Prereq: worktree exists —
#   cd ~/Projects/IsaacLab && git fetch origin tag v3.0.0-EA --no-tags \
#       && git worktree add ~/Projects/IsaacLab-ea v3.0.0-EA
set -euo pipefail
IL_EA="${IL_EA:-$HOME/Projects/IsaacLab-ea}"
export OMNI_KIT_ACCEPT_EULA=YES UV_HTTP_TIMEOUT=900

[ -d "$IL_EA" ] || { echo "missing worktree $IL_EA — see prereq in header" >&2; exit 1; }
cd "$IL_EA"

uv sync --extra isaacsim --extra wandb --extra video --extra importers --extra rsl-rl
uv pip install --python .venv/bin/python ultralytics "imageio[ffmpeg]"

echo "[phase6] done — next: source scripts/phase6_env.sh"
