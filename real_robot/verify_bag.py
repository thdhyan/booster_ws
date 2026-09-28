#!/usr/bin/env python3
"""Post-hoc bag check: duration, topic coverage, message counts, NaN scan.

Run after recording, or on any bag later:
    python3 verify_bag.py bags/walk_fast_20260928T120000Z
    python3 verify_bag.py bags/* --json

Exits non-zero if a required topic is missing or a numeric field contains NaN.
NaN in an IMU or a TF is the single most common cause of a silently broken
map, and it is invisible in a file listing.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REQUIRED = ["/tf", "/imu/data"]
RECOMMENDED = ["/odom", "/joint_states"]
VISUAL_HINTS = ("image_raw", "camera_info")
NUMERIC_FLOAT_FIELDS = (
    "linear_acceleration", "angular_velocity", "orientation",
    "linear_acceleration_covariance", "angular_velocity_covariance",
    "orientation_covariance",
)


def load_metadata(bag: Path):
    """Read rosbag2 metadata, tolerating the Humble(sqlite3) and Jazzy(mcap) schemas.

    Humble:  duration.nanoseconds, topics_with_message_count[].topic
    Jazzy:   rosbag2_bagfile_information.duration.nanoseconds,
             topics_with_message_count[].topic_metadata.name
    """
    meta = bag / "metadata.yaml"
    if not meta.is_file():
        # Humble writes <bagname>.metadata.yaml beside the db3.
        alts = sorted(bag.glob("*.metadata.yaml"))
        if not alts:
            raise SystemExit(f"no metadata.yaml in {bag} — bag was never finalized")
        meta = alts[0]
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    return yaml.safe_load(meta.read_text())


def extract(meta: dict, storage_hint: str = "sqlite3") -> tuple[dict[str, int], float, int, str]:
    """-> (topic -> message_count, duration_s, total_messages, storage_id)"""
    info = meta.get("rosbag2_bagfile_information", meta)
    dur_ns = (info.get("duration", {}) or {}).get("nanoseconds", 0)
    total = int(info.get("message_count", 0) or 0)

    counts: dict[str, int] = {}
    for entry in info.get("topics_with_message_count", []) or []:
        name = entry.get("topic")
        if name is None:  # mcap / Jazzy nests it
            name = (entry.get("topic_metadata", {}) or {}).get("name")
        if name is None and isinstance(entry.get("name"), str):
            name = entry["name"]
        if name is not None:
            counts[name] = int(entry.get("message_count", 0) or 0)
    return counts, float(dur_ns) / 1e9, total, str(info.get("storage_identifier", storage_hint))


def scan_nan(bag: Path, topics, storage: str = "sqlite3", limit_bytes: int = 200_000_000) -> dict:
    """Best-effort NaN scan over mcap/sqlite payloads.

    Returns per-topic counts. If rosbag2_py is unavailable this reports
    'skipped' rather than silently claiming the data is clean.
    """
    result = {}
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        return {t: "skipped (no rosbag2_py in this interpreter)" for t in topics}

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    seen = {}
    total = 0
    while total < limit_bytes:
        try:
            topic, data, _ = reader.read_next()
        except RuntimeError:
            break  # end of bag
        if topic not in types or topic not in NUMERIC_FLOAT_FIELDS and topic not in (
            "sensor_msgs/msg/Imu", "nav_msgs/msg/Odometry"):
            seen.setdefault(topic, 0)
            continue
        seen.setdefault(topic, 0)
        seen[topic] += 1
        try:
            msg = deserialize_message(data, get_message(types[topic]))
        except Exception:
            continue
        total += len(data)
        for fname in ("orientation", "angular_velocity", "linear_acceleration",
                      "twist", "pose"):
            f = getattr(msg, fname, None)
            vals = []
            if f is not None and hasattr(f, "x"):
                vals = [f.x, f.y, f.z, getattr(f, "w", 0.0)]
            elif f is not None and hasattr(f, "covariance"):
                vals = list(f.covariance)
            for v in vals:
                if isinstance(v, float) and math.isnan(v):
                    seen[topic] = f"NaN in {fname}"
                    break
    result.update(seen)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--nan-scan", action="store_true",
                    help="decode messages and scan for NaN (slower, needs rosbag2_py)")
    args = ap.parse_args()

    rc = 0
    for b in args.bags:
        bag = Path(b)
        try:
            meta = load_metadata(bag)
        except SystemExit as e:
            print(f"[verify] {bag.name}: FAIL — {e}")
            rc = 1
            continue

        counts, dur, total, storage = extract(meta)
        size = sum(f.stat().st_size for f in bag.rglob("*") if f.is_file())

        missing = [t for t in REQUIRED if counts.get(t, 0) == 0]
        missing_rec = [t for t in RECOMMENDED if counts.get(t, 0) == 0]
        visual = [t for t in counts if any(h in t for h in VISUAL_HINTS)]

        print(f"\n[verify] {bag.name}")
        print(f"  duration   : {dur:.1f} s")
        size_s = f"{size/1e6:.1f} MB" if size >= 1e6 else f"{size/1e3:.0f} KB"
        print(f"  size       : {size_s}")
        print(f"  storage    : {storage}")
        print(f"  topics     : {len(counts)}")
        for t, c in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
            print(f"      {c:>9d}  {t}")
        if visual:
            print(f"  visual     : {', '.join(visual[:4])}")
        else:
            print("  visual     : NONE — no map is possible from this bag")
        if missing_rec:
            print(f"  note       : not recorded: {', '.join(missing_rec)}")

        if args.nan_scan:
            for t, s in scan_nan(bag, list(counts), storage=storage).items():
                if isinstance(s, str) and "NaN" in s:
                    print(f"  NaN        : {t} -> {s}")
                    rc = 1

        if missing:
            print(f"  MISSING    : {', '.join(missing)}")
            rc = 1
        elif not visual:
            print("  INCOMPLETE : no camera topics")
            rc = 1
        else:
            print("  OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
