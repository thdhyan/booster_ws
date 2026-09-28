#!/usr/bin/env python3
"""Query A2 for its camera inventory over the Booster DDS interface.

The vendor docs describe a `booster_ros2` ros2_control bridge (that repo is a
404) and never mention cameras. The real interface is the Python SDK: it exposes
CameraClient / CameraModel / CameraPos, and the data-collection page's
head_cam / left_cam / right_cam map onto CameraPos.HEAD / LEFT_HAND / RIGHT_HAND.

READ-ONLY. This queries the camera list and state. It sends no motion, no mode
change and no RPC that could move the robot.

  python query_cameras.py --ip 10.37.11.3
  python query_cameras.py --ip 10.37.11.3 --iface 10.37.11.89
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="robot IP")
    ap.add_argument("--iface", default="",
                    help="local interface/IP for DDS (default: autodetect)")
    ap.add_argument("--timeout", type=float, default=12.0)
    a = ap.parse_args()

    import booster_robotics_sdk_python as b

    # Which camera models this build knows about -- useful even if the robot
    # reports none, because it tells us what firmware could ever report.
    models = [n for n in dir(b) if n.startswith("CAMERA_Model_")]
    print("camera models this SDK build knows:")
    for m in models:
        print(f"    {m} = {getattr(b, m)}")
    print("camera positions:")
    for n in dir(b):
        if n.startswith("CAMERA_POS_"):
            print(f"    {n} = {getattr(b, n)}")
    print()

    iface = a.iface
    if not iface:
        # ChannelFactory wants a local interface address to bind discovery to.
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((a.ip, 9))
            iface = s.getsockname()[0]
        finally:
            s.close()
    print(f"[cam] robot={a.ip}  local iface={iface}")

    try:
        # ChannelFactory.Init is a pybind11 static exposed with an explicit
        # `self`, so it must be called on an instance.
        b.ChannelFactory().Init(0, iface)
    except Exception as e:                              # noqa: BLE001
        print(f"[cam] ChannelFactory.Init FAILED: {type(e).__name__}: {e}")
        print("       DDS discovery may be blocked by the AP (multicast).")
        return 2
    print("[cam] ChannelFactory initialised")

    found: list = []
    try:
        cli = b.CameraClient()
        for attempt in range(2):
            try:
                cli.Init(a.ip)
                found = list(cli.GetCameras() or [])
            except Exception as e:                      # noqa: BLE001
                print(f"[cam] GetCameras (try {attempt + 1}) "
                      f"{type(e).__name__}: {str(e)[:120]}")
                time.sleep(1.5)
                continue
            print(f"[cam] GetCameras -> {len(found)} camera(s)")
            for c in found:
                print(f"     {c}")
            break
    except Exception as e:                              # noqa: BLE001
        print(f"[cam] CameraClient failed: {type(e).__name__}: {str(e)[:160]}")

    if not found:
        print("\n[cam] RESULT: robot reported NO cameras.")
        print("       Consistent with the on-robot probe: no /dev/video*, no")
        print("       RealSense on USB, /boostercamera/head/rgb has 0 publishers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
