#!/bin/bash
# Spark python probe (container-side, fed via: docker run --entrypoint bash IMG -s < this).
# Finds which python in the image can import isaaclab / rsl_rl; prints PATH.
set +e
echo "PATH=$PATH"
for C in /isaac-sim/python.sh python3 python /opt/conda/bin/python /usr/local/bin/python3; do
  P="$C"
  case "$C" in
    /*) [ -x "$C" ] || { echo "skip(abs) $C"; continue; } ;;
    *) P="$(command -v "$C" 2>/dev/null)"; [ -n "$P" ] || { echo "skip(path) $C"; continue; } ;;
  esac
  echo "-- candidate $P"
  "$P" -c "import isaaclab; print('ISAACLAB_OK', '$P')" 2>&1 | tail -1
  "$P" -c "import rsl_rl; print('RSL_OK', '$P')" 2>&1 | tail -1
  "$P" -c "import torch; print('TORCH_OK', '$P', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
done
echo "PROBE_DONE"
