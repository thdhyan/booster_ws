#!/usr/bin/env python3
"""Watchdog for the K1 head stereo pair.

WHAT IT DOES
  Periodically subscribes to the rectified left+right topics. If the pair stops
  delivering, it tries to bring it back, in escalating order:
    1. restart booster-video-stream
    2. restart booster-server-perception / booster-daemon-perception
  and records a diagnostic snapshot every time it cannot recover, so we learn
  *why* rather than just seeing a node flapping.

WHY IT CANNOT ALWAYS FIX IT  (read this before trusting a green watchdog)
  booster-video-stream self-describes as "start realsense receiver" and logs
  `Publishing: '0'` indefinitely on this robot: there is no RealSense fitted.
  The /boostercamera/head/* topics are produced by a MIPI pipeline
  (mipi_cam -> StereoNetNode). On A2 the pair came up once after a reboot
  (~7.8 Hz) and then stopped; mipi_cam has no binary on disk and is not
  supervised by any systemd unit. So this watchdog can restart what exists and
  detect failure honestly, but if the underlying driver is absent no amount of
  restarting will conjure frames. That case is reported as UNRECOVERABLE, not
  silently retried forever.

  Verified device state is part of every failure snapshot: no /dev/video*, and
  /dev/v4l/by-path absent, is what a missing MIPI sensor looks like.

USAGE
  ./watchdog_k1_stereo.py --interval 20
  ./watchdog_k1_stereo.py --interval 20 --once     # single check, no loop
  ./watchdog_k1_stereo.py --status                 # report and exit
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

LEFT = "/boostercamera/head/rgb"
RIGHT = "/boostercamera/head/right/rgb"
DEPTH = "/boostercamera/head/depth"
RAW = "/boostercamera/head/raw/rgb"

VIDEO_STREAM = ("/opt/booster/BoosterRos2/install/booster-video-stream/lib/"
                "booster-video-stream/booster-video-stream")
PERCEPTION = ["/opt/booster/ServerPerception/bin/booster-server-perception",
              "/opt/booster/DaemonPerception/bin/booster-daemon-perception"]

LOG = "/tmp/watchdog_k1_stereo.log"
MIN_HZ = 0.5


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:                                # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


# --------------------------------------------------------------- rate check
def measure(seconds: float = 8.0) -> dict:
    """Subscribe to the pair and count messages. Returns Hz per topic."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                               DurabilityPolicy, HistoryPolicy)
        from sensor_msgs.msg import Image
    except Exception as e:                                # noqa: BLE001
        return {"error": f"rclpy unavailable: {e}"}

    rclpy.init()
    node = Node("wd_probe")
    # The publishers offer a durability that rejects a plain default reader, so
    # match RELIABLE + VOLATILE explicitly. This exact combination is what made
    # the pair visible when it was working.
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST)
    counts = {t: 0 for t in (LEFT, RIGHT, DEPTH, RAW)}
    subs = []
    for t in counts:
        def mk(t=t):
            def cb(_msg):
                counts[t] += 1
            return cb
        subs.append(node.create_subscription(Image, t, mk(), qos))
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()
    return {t: counts[t] / seconds for t in counts}


# ------------------------------------------------------------- diagnostics
def snapshot() -> str:
    """Why is the pair down? This is the part that actually teaches us."""
    out = []
    out.append("  device nodes:")
    v = sh(["bash", "-lc", "ls /dev/video* 2>/dev/null || echo '(none)'"]).strip()
    out.append(f"    /dev/video*      : {v or '(none)'}")
    p = sh(["bash", "-lc",
            "ls -A /dev/v4l/by-path/ 2>/dev/null | head -4 || true"]).strip()
    out.append(f"    /dev/v4l/by-path : {p or '(EMPTY - no MIPI sensor enumerated)'}")
    out.append("  processes:")
    for name in ("mipi_cam", "booster-video-stream", "booster-server-perception",
                 "booster-daemon-perception", "StereoNet"):
        r = sh(["bash", "-lc", f"pgrep -af {name} 2>/dev/null | head -1 || true"]).strip()
        out.append(f"    {name:28s}: {r or 'NOT RUNNING'}")
    out.append("  resource pressure:")
    out.append("    " + sh(["bash", "-lc",
                             "free -m | awk 'NR==2{print \"mem used \"$3\"Mi avail \"$7\"Mi\"}'"]).strip())
    out.append("    " + sh(["bash", "-lc", "uptime"]).strip())
    return "\n".join(out)


# ----------------------------------------------------------------- recovery
def restart_video_stream() -> bool:
    if sh(["bash", "-lc", f"pgrep -f booster-video-stream"]).strip():
        sh(["bash", "-lc", "pkill -f booster-video-stream"])
        time.sleep(2)
    subprocess.Popen([VIDEO_STREAM],
                     stdout=open("/tmp/video_stream_wd.log", "ab"),
                     stderr=subprocess.STDOUT,
                     start_new_session=True)
    time.sleep(6)
    return bool(sh(["bash", "-lc", "pgrep -f booster-video-stream"]).strip())


def sensor_present() -> bool:
    """Is there any camera device at all? Decides recoverable vs not."""
    return bool(sh(["bash", "-lc", "ls -A /dev/v4l/by-path/ 2>/dev/null"]).strip()
                or sh(["bash", "-lc", "ls /dev/video* 2>/dev/null"]).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        r = measure()
        log(f"STATUS {r}")
        log("diagnostic:\n" + snapshot())
        return 0

    log(f"watchdog up (interval={a.interval:g}s, min {MIN_HZ} Hz), log={LOG}")
    fails = 0
    while True:
        r = measure()
        if "error" in r:
            log(f"probe error: {r['error']}")
        else:
            lz, rz = r[LEFT], r[RIGHT]
            if lz >= MIN_HZ and rz >= MIN_HZ:
                if fails:
                    log(f"RECOVERED: left={lz:.1f} Hz right={rz:.1f} Hz "
                        f"depth={r[DEPTH]:.1f} raw={r[RAW]:.1f} after {fails} failure(s)")
                fails = 0
            else:
                fails += 1
                log(f"DOWN #{fails}: left={lz:.1f} Hz right={rz:.1f} Hz "
                    f"(threshold {MIN_HZ})")
                if sensor_present():
                    log("sensor device IS present -> attempting restart")
                    ok = restart_video_stream()
                    log(f"  booster-video-stream restart: "
                        f"{'started' if ok else 'FAILED'}")
                else:
                    log("UNRECOVERABLE: no camera device on this robot "
                        "(/dev/v4l/by-path empty). No restart can produce frames.")
                    log("diagnostic:\n" + snapshot())
        if a.once:
            return 0
        time.sleep(max(5.0, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())
