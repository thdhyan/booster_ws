#!/usr/bin/env bash
# Restart the k1m Gradio app, and verify the NEW process owns the port.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# A plain `pkill -f "k1m/bin/python app.py"` does not reliably stop a previous
# instance, so a relaunch silently fails to bind :7860 and the OLD code keeps
# serving. That happened: a fix was committed, the app was "restarted", and the
# user kept getting the old `IsADirectoryError` because PID 100587 from hours
# earlier still owned the socket. A restart that does not verify who owns the
# port is not a restart.
#
# Also note: pgrep/pkill -f patterns match the *calling shell's* command line
# too, so a naive pattern like "k1m/bin/python app.py" will kill the shell
# running this script. The bracket below keeps the pattern from matching
# itself.
set -uo pipefail

PORT="${K1M_PORT:-7860}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${K1M_PYTHON:-$HOME/.venvs/k1m/bin/python}"
LOG="${K1M_LOG:-/tmp/k1m_app.log}"

# `pytho[n]` matches "python" but not this script's own command line.
PATTERN='k1m/bin/pytho[n] app\.py'

echo "[k1m] stopping any existing app on :$PORT"
for pid in $(pgrep -f "$PATTERN" || true); do
  echo "  killing $pid"
  kill "$pid" 2>/dev/null || true
done
for _ in $(seq 1 15); do
  pgrep -f "$PATTERN" >/dev/null || break
  sleep 1
done
for pid in $(pgrep -f "$PATTERN" || true); do
  echo "  $pid ignored SIGTERM, sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
done

# Refuse to start if something else still holds the port.
if ss -lptn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  echo "[k1m] ERROR: :$PORT is still held by another process:" >&2
  ss -lptn "sport = :$PORT" 2>/dev/null | tail -n +2 >&2
  exit 1
fi

[[ -x "$PY" ]] || { echo "[k1m] ERROR: no interpreter at $PY" >&2; exit 1; }
cd "$REPO"

# Drop PYTHONPATH/LD_LIBRARY_PATH: the ROS 2 workspace exports break torch and
# mujoco imports (see scripts/py_torch.sh).
setsid nohup env -u PYTHONPATH -u LD_LIBRARY_PATH "$PY" app.py \
  > "$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 40); do
  sleep 1
  newpid=$(pgrep -f "$PATTERN" | head -1)
  if [[ -n "${newpid:-}" ]] \
     && ss -lptn "sport = :$PORT" 2>/dev/null | grep -q "pid=$newpid"; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" \
           2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "[k1m] up on http://127.0.0.1:$PORT/  pid=$newpid  (verified: it owns the port and answers 200)"
      echo "[k1m] log: $LOG"
      exit 0
    fi
  fi
done

echo "[k1m] ERROR: did not come up; last 20 log lines:" >&2
tail -20 "$LOG" >&2
exit 1
