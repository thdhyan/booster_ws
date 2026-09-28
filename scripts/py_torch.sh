#!/usr/bin/env bash
# Run any python in a torch-safe environment.
#
# WHY THIS EXISTS
# ---------------
# The ROS 2 workspace exports PYTHONPATH (workspace site-packages +
# /opt/ros/jazzy/...) and LD_LIBRARY_PATH (incl. gz_sim_vendor, gz_sensor).
# With those set, importing torch in ANY venv fails:
#
#   AttributeError: module 'torch._C' has no attribute 'OutOfMemoryError'
#   AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'
#
# Those are not version incompatibilities and not a broken install — the ROS
# Gazebo vendor libraries on LD_LIBRARY_PATH shadow torch's own libtorch .so
# files, so `torch._C` loads a mismatched extension. Stripping the two vars
# fixes it completely, verified with torch 2.6.0+cpu / numpy 2.1.3.
#
# USAGE
#   scripts/py_torch.sh python3 -c "import torch; print(torch.__version__)"
#   scripts/py_torch.sh scripts/whatever.py --flag
#
# ROS 2 nodes that do NOT need torch (mujoco-based sim, the feasibility gate,
# the replay recorder) can keep the normal sourced shell.
set -euo pipefail

VENV="${K1_TORCH_VENV:-$HOME/.venvs/torchcpu}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: no interpreter at $PY" >&2
  echo "create it with:" >&2
  echo "  uv venv --python 3.12 $VENV" >&2
  echo "  uv pip install --python $PY 'torch==2.6.0' 'numpy==2.1.3' \\" >&2
  echo "      --index-url https://download.pytorch.org/whl/cpu" >&2
  exit 1
fi

# Keep only PATH; drop the ROS/Gazebo library paths that break torch.
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH \
    -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH -u ROS_DISTRO \
    "$PY" "$@"
