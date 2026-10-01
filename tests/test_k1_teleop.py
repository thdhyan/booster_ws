#!/usr/bin/env python3
"""Checks for k1_teleop (XR -> robot frames, sticks, head, arm clutch, K1 arm
IK against its own FK). No ROS, no headset:

    python3 -m pytest tests/test_k1_teleop.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from k1_teleop.k1_arm_ik import ArmIK  # noqa: E402
from k1_teleop.teleop_mapping import (  # noqa: E402
    BASE_LIMITS, ArmClutch, Edge, base_twist, head_targets, xr_pose_to_robot, yaw_of,
)

IDENT = (0.0, 0.0, 0.0, 1.0)
ARMS_DOWN = {"Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3}   # K1 init pose


def xr(pos, quat=IDENT):
    return xr_pose_to_robot(pos, quat)


def test_frames() -> None:
    assert np.allclose(xr((0, 0, -1))[:3, 3], (1, 0, 0))   # XR forward -> robot +x
    assert np.allclose(xr((0, 1, 0))[:3, 3], (0, 0, 1))    # XR up -> robot +z
    assert np.allclose(xr((1, 0, 0))[:3, 3], (0, -1, 0))   # XR right -> robot -y
    s = math.sin(math.pi / 4)                               # operator turned 90 deg left
    assert abs(yaw_of(xr((0, 0, 0), (0, s, 0, s))[:3, :3]) - math.pi / 2) < 1e-6


def test_sticks_and_buttons() -> None:
    assert base_twist(0, 0, 0, 0) == (0.0, 0.0, 0.0, 0.0)
    vx, vy, vz, wz = base_twist(0.0, 1.0, 0.0, 0.0)          # left stick forward
    assert vx == BASE_LIMITS["vx"] and vy == vz == wz == 0.0
    assert base_twist(1.0, 0, 0, 0)[1] == -BASE_LIMITS["vy"]  # left stick right -> move right (-y)
    assert base_twist(0, 0, 1.0, 0)[3] == -BASE_LIMITS["wz"]  # right stick right -> turn right
    assert base_twist(0, 0, 0, -1.0)[2] == -BASE_LIMITS["vz"]  # right stick down -> lower
    assert base_twist(0.03, float("nan"), 5.0, 0)[:2] == (0.0, 0.0)
    e = Edge()
    assert [e(v) for v in (0, 1, 1, 0, 1)] == [False, True, False, False, True]


def test_head() -> None:
    assert head_targets(xr((0, 1.6, 0)), 0.0) == {"AAHead_yaw": 0.0, "Head_pitch": 0.0}
    s = math.sin(math.radians(15))                          # 30 deg left (about XR +y)
    h = head_targets(xr((0, 1.6, 0), (0, s, 0, math.cos(math.radians(15)))), 0.0)
    assert abs(h["AAHead_yaw"] - math.radians(30)) < 1e-6
    assert abs(head_targets(xr((0, 1.6, 0), (0, s, 0, math.cos(math.radians(15)))), math.radians(30))["AAHead_yaw"]) < 1e-6
    s = math.sin(math.radians(-10))                         # 20 deg down (about XR +x, negative)
    h = head_targets(xr((0, 1.6, 0), (s, 0, 0, math.cos(math.radians(-10)))), 0.0)
    assert abs(h["Head_pitch"] - math.radians(20)) < 1e-6   # positive = looking down
    s = math.sin(math.radians(80))                          # 160 deg left: clamped
    assert head_targets(xr((0, 0, 0), (0, s, 0, math.cos(math.radians(80)))), 0.0)["AAHead_yaw"] == 1.0


def test_clutch() -> None:
    H0 = np.eye(4)
    H0[:3, 3] = (0.1, 0.2, 0.0)
    c = ArmClutch(shoulder=np.array([0.0, 0.15, 0.17]), max_speed=10.0)
    c.engage(xr((0.2, 1.0, -0.3)), 0.0, H0)
    assert np.allclose(c.goal(xr((0.2, 1.0, -0.3)), 0.02), H0)
    g = c.goal(xr((0.2, 1.0, -0.4)), 0.02)                     # 10 cm forward in the room
    assert np.allclose(g[:3, 3], (0.2, 0.2, 0.0)), g[:3, 3]
    c.engage(xr((0.0, 1.0, 0.0)), math.pi / 2, H0)             # operator faces room +y
    g = c.goal(xr((-0.1, 1.0, 0.0)), 0.02)
    assert np.allclose(g[:3, 3], (0.2, 0.2, 0.0)), g[:3, 3]


def test_ik_geometry() -> None:
    ik = ArmIK()
    assert ik.names["left"] == ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]
    L = ik.arm_length("left")
    assert 0.33 < L < 0.40, L          # 0.166 upper arm + 0.20 forearm
    assert abs(L - ik.arm_length("right")) < 1e-9
    # zero pose = T-pose: tip straight out sideways from the shoulder, at full reach
    for side, sign in (("left", 1), ("right", -1)):
        d = ik.fk(side, {})[:3, 3] - ik.shoulder_pos(side, {})
        assert abs(np.linalg.norm(d) - L) < 0.02 and sign * d[1] > 0.9 * L, d
        # init pose: arms hang down
        d = ik.fk(side, ARMS_DOWN)[:3, 3] - ik.shoulder_pos(side, ARMS_DOWN)
        assert d[2] < -0.3 * L, d


def test_ik_vs_fk() -> None:
    ik, rng = ArmIK(), np.random.default_rng(0)
    for side in ("left", "right"):
        lo, hi = ik._lim[side].T
        errs = []
        for _ in range(200):
            q_true = rng.uniform(lo, hi)
            goal = ik.fk(side, {}, q_true)
            # warm start = a nearby pose, as in teleop (the goal moves <= 1 cm per step)
            q0 = np.clip(q_true + rng.normal(0, 0.15, 4), lo, hi)
            state = dict(zip(ik.names[side], q0))
            q, err = ik.solve(side, goal, state)
            assert abs(np.linalg.norm(goal[:3, 3] - ik.fk(side, {}, q)[:3, 3]) - err) < 1e-9
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
            errs.append(err)
        errs = np.array(errs)
        assert np.mean(errs < 0.005) > 0.97, (side, np.sort(errs)[-10:])


def test_ik_tracks_clutch_path() -> None:
    """Hand moving forward from arms-down, 1 cm per step: IK follows within 5 mm."""
    ik = ArmIK()
    state = dict(ARMS_DOWN)
    for side in ("left", "right"):
        start = ik.fk(side, state)
        for k in range(1, 21):
            goal = start.copy()
            goal[:3, 3] += (0.01 * k, 0.0, 0.005 * k)
            q, err = ik.solve(side, goal, state)
            state.update(zip(ik.names[side], q))
            assert err < 0.005, (side, k, err)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
