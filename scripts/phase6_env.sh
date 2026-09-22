#!/usr/bin/env bash
# Phase-6 canonical training environment — Isaac Sim 6.1.0 + IsaacLab v3.0.0-EA.
#
# Usage:  source scripts/phase6_env.sh     (must be SOURCED, not executed)
# Then:   phase6-python <script> ...   or   $PHASE6_VENV/bin/python ...
#
# Our task packages are NOT pip-installed: setuptools egg-info was root-owned
# (removed 2026-09-22) and EA dropped the `omni.isaac.lab.tasks` entry-point
# group, so package metadata isn't consumed anyway — PYTHONPATH is equivalent.
# After any `uv sync` in the worktree, re-run scripts/install_phase6_env.sh
# (sync strips ultralytics/imageio extras).

export PHASE6_IL_EA="${PHASE6_IL_EA:-$HOME/Projects/IsaacLab-ea}"
export PHASE6_VENV="$PHASE6_IL_EA/.venv"
export OMNI_KIT_ACCEPT_EULA=YES
# venv on PATH so the unified `isaaclab` CLI and `python` resolve to the venv
export PATH="$PHASE6_VENV/bin:$PATH"

_phase6_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$_phase6_ws/isaac_tasks/k1_velocity/source:$_phase6_ws/isaac_tasks/booster_train_ref/source/booster_train:$_phase6_ws/src/k1_description/assets/src${PYTHONPATH:+:$PYTHONPATH}"
unset _phase6_ws

phase6-python() { "$PHASE6_VENV/bin/python" "$@"; }

echo "[phase6] venv=$PHASE6_VENV"
echo "[phase6] k1_velocity / booster_train / booster_assets on PYTHONPATH — use phase6-python"
