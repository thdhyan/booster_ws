"""Carry task geometry: the hard-coded arm FK, mirror and hold-pose table vs the URDF.

carry_mdp copies the K1 arm chain by hand so it can run at USD time without a
kinematics library. These tests rebuild the chain from K1_22dof.urdf independently
and check that (a) both FKs agree, (b) the right arm really is the mirror of the
left under MIRROR, and (c) every width in range puts both stub tips on the box
faces with the joints inside their URDF limits.
"""
from __future__ import annotations

import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "booster_assets/robots/K1/K1_22dof.urdf"
MDP = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/carry/carry_mdp.py"


def _load_mdp():
    spec = importlib.util.spec_from_file_location("carry_mdp", MDP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_mdp()


def _urdf_chain(hand: str):
    root = ET.parse(URDF).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], hand
    while link != "Trunk":
        j = by_child[link]
        chain.append(j)
        link = j.find("parent").get("link")
    return chain[::-1]


def _rot(axis, a):
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _urdf_fk(hand: str, q) -> np.ndarray:
    T, i = np.eye(4), 0
    for j in _urdf_chain(hand):
        o = j.find("origin")
        assert o.get("rpy", "0 0 0").split() == ["0", "0", "0"]
        A = np.eye(4)
        A[:3, 3] = [float(v) for v in o.get("xyz").split()]
        T = T @ A
        if j.get("type") == "revolute":
            B = np.eye(4)
            B[:3, :3] = _rot([float(v) for v in j.find("axis").get("xyz").split()], q[i])
            T = T @ B
            i += 1
    return T


def _limits(hand: str):
    return np.array([[float(j.find("limit").get(k)) for k in ("lower", "upper")] for j in _urdf_chain(hand)])


@pytest.mark.parametrize("w", np.linspace(0.25, 0.40, 7))
def test_fk_and_mirror_match_urdf(w):
    g = cm.carry_geometry(w)
    assert np.allclose(cm.arm_fk(g["q_left"], "left"), _urdf_fk(cm.LEFT_EE, g["q_left"]), atol=1e-6)
    assert np.allclose(cm.arm_fk(g["q_right"], "right"), _urdf_fk(cm.RIGHT_EE, g["q_right"]), atol=1e-6)
    # right hand = left hand reflected through the sagittal (xz) plane
    M = np.diag([1.0, -1.0, 1.0])
    T_l, T_r = _urdf_fk(cm.LEFT_EE, g["q_left"]), _urdf_fk(cm.RIGHT_EE, g["q_right"])
    assert np.allclose(M @ T_l[:3, 3], T_r[:3, 3], atol=1e-6)


@pytest.mark.parametrize("w", np.linspace(0.25, 0.40, 13))
def test_stub_tips_sit_on_box_faces_within_limits(w):
    g = cm.carry_geometry(w)
    # weld points on the box faces: +-w/2 laterally, within 1 cm
    assert abs(g["anchor_l"][1] - w / 2) < 0.01
    assert abs(g["anchor_r"][1] + w / 2) < 0.01
    # anchors inside the box depth/height
    for a in (g["anchor_l"], g["anchor_r"]):
        assert abs(a[0]) <= cm.BOX_DEPTH / 2 + 1e-6
        assert abs(a[2]) <= cm.BOX_HEIGHT / 2 + 1e-6
    for hand, q in ((cm.LEFT_EE, g["q_left"]), (cm.RIGHT_EE, g["q_right"])):
        lim = _limits(hand)
        assert np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1]), (hand, q, lim)


def test_box_in_front_of_trunk():
    for w in (0.25, 0.40):
        c = cm.carry_geometry(w)["center"]
        assert c[0] - cm.BOX_DEPTH / 2 > 0.08     # rear face clear of the chest
        assert abs(c[1]) < 1e-9


def test_weld_quaternion_round_trip():
    g = cm.carry_geometry(0.33)
    w, x, y, z = cm._mat_to_quat_wxyz(g["rot_l"])
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(R, g["rot_l"], atol=1e-6)
