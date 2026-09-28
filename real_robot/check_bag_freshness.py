#!/usr/bin/env python3
"""Detect stale topics in a rosbag2 bag: present but not actually updating.

WHY THIS EXISTS
---------------
The first K1 walk produced a bag that passed every count-based check and was
still worthless. /booster_video_stream logged 2994 messages at a healthy 10 Hz,
but all 2994 were the SAME JPEG with the SAME frozen header stamp
(1790619558.306908794 -- the moment the camera driver started, 11 minutes before
the walk began). The camera had published one frame at startup and then
republished that frame at 10 Hz forever.

So "the topic has messages" and "the data is fresh" are completely different
questions, and only the second one matters. Counting messages cannot tell them
apart; only comparing content can.

WHAT IT CHECKS
--------------
  image topics : unique header stamps AND unique payload hashes. A real camera
                 advances both. One of each means a frozen frame.
  state topics : header stamps must advance across the bag.
  everything   : reports messages/unique-stamps so a frozen topic is obvious.

Usage:  check_bag_freshness.py <bag_dir> [--image-topic T]...
Exit:   0 all topics live, 1 at least one topic is stale/frozen.
"""
import hashlib
import os
import sqlite3
import sys

# Topics whose payload must change from message to message.
IMAGE_TOPICS = ("/booster_video_stream", "/boostercamera/head/rgb", "/boostercamera/head/depth")
# Topics that should advance in time even if the payload is small.
STATE_TOPICS = ("/low_state", "/joint_states", "/odometer_state", "/tf")


def analyse(db, topic_names):
    con = sqlite3.connect(db)
    out = {}
    for name in topic_names:
        row = con.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
        if not row:
            continue
        stamps, hashes, n = set(), set(), 0
        for (blob,) in con.execute("SELECT data FROM messages WHERE topic_id=?", (row[0],)):
            n += 1
            # Full-message hash catches a frozen payload. Cheap enough at these
            # message counts, and correctness beats cleverness here.
            hashes.add(hashlib.blake2b(blob, digest_size=8).digest())
            # CDR: 4-byte encapsulation header, then the header stamp. Both
            # little- and big-endian are read so this works on any host.
            if len(blob) >= 16:
                for endian in ("little", "big"):
                    sec, nsec = int.from_bytes(blob[4:8], endian, signed=True), int.from_bytes(
                        blob[8:12], endian, signed=True
                    )
                    if 0 <= sec < 2**31 and nsec < 10**9:
                        stamps.add((sec, nsec))
                        break
        out[name] = (n, len(stamps), len(hashes))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bag = sys.argv[1]
    extra = [a for i, a in enumerate(sys.argv) if sys.argv[i - 1] == "--image-topic"]

    db = None
    for f in os.listdir(bag):
        if f.endswith(".db3"):
            db = os.path.join(bag, f)
            break
    if not db:
        print(f"[FAIL] no .db3 in {bag}")
        return 1

    stats = analyse(db, list(IMAGE_TOPICS) + extra + list(STATE_TOPICS))
    if not stats:
        print("[FAIL] none of the expected topics are in this bag")
        return 1

    print(f"  freshness of {os.path.basename(bag.rstrip('/'))}:")
    stale = []
    for name, (n, ns, nh) in sorted(stats.items()):
        kind = "IMAGE" if name in IMAGE_TOPICS or name in extra else "state"
        if n == 0:
            status = "EMPTY"
            stale.append(name)
        elif ns <= 1 or (kind == "IMAGE" and nh <= 1):
            status = "STALE  <-- frozen, not real data"
            stale.append(name)
        else:
            status = "live"
        print(f"    {name:<42} msgs={n:<8} unique_stamps={ns:<6} unique_payloads={nh:<5} [{kind}] {status}")

    if stale:
        print(f"  [FAIL] stale topics: {', '.join(stale)}")
        return 1
    print("  [ok] every topic is updating")
    return 0


if __name__ == "__main__":
    sys.exit(main())
