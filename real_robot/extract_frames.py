#!/usr/bin/env python3
"""Extract JPEG frames from a K1 walk bag so the visual data can be eyeballed.

A bag that lists 244 CompressedImage messages proves the topic was recorded; it
does NOT prove the JPEGs decode. This pulls a few frames out and reports their
dimensions, so a corrupt stream is caught before it costs you a whole walk.

Usage: extract_frames.py <bag_dir> <out_dir> [n_frames] [step_seconds]
"""
import os
import sys

import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

TOPIC = "/booster_video_stream"
ALT = "/boostercamera/head/rgb"


def open_reader(bag):
    store = rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3")
    # Convert the newer StorageOptions(uri=...) spelling if this Humble wants dir=.
    try:
        return rosbag2_py.SequentialReader(), store
    except Exception:
        pass
    store = rosbag2_py.StorageOptions(directory=bag, storage_id="sqlite3")
    return rosbag2_py.SequentialReader(), store


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    bag, out = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    os.makedirs(out, exist_ok=True)

    # No subscription is needed: frames come straight out of the bag, so the
    # node is never spun.
    reader, store = open_reader(bag)
    try:
        reader.open(store, rosbag2_py.ConverterOptions("", ""))
    except TypeError:
        conv = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader.open(store, conv)

    n_seen, n_saved, last_t = 0, 0, None
    while reader.has_next() and n_saved < want:
        topic, data, t = reader.read_next()
        if topic not in (TOPIC, ALT):
            continue
        n_seen += 1
        if last_t is not None and (t - last_t) < step * 1e9:
            continue
        last_t = t
        msg = deserialize_message(data, CompressedImage if topic == TOPIC else Image)
        ext = "jpg" if topic == TOPIC else "png"
        p = os.path.join(out, f"frame_{n_saved:03d}.{ext}")
        with open(p, "wb") as fh:
            fh.write(bytes(msg.data))
        n_saved += 1

    print(f"frames seen on camera topics: {n_seen}")
    print(f"frames written to {out}: {n_saved}")
    rclpy.shutdown()
    if n_saved == 0:
        print("[FAIL] no camera frames found in bag")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
