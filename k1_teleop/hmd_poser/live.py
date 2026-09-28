"""Live HMD-Poser (CVPR 2024, Pico-AI-Team/HMD-Poser, MIT) on Quest head + controller poses.

HMD-Poser predicts the full SMPL body (22 joints: root orientation + 21 local
rotations) from the head and both hands, 60 Hz, 40-frame windows. Here:

* poses arrive in robot axes (x forward, y left, z up - ``teleop_mapping``),
  which is z-up like the AMASS training data;
* SMPL frames (x = body left, y = up, z = forward) are derived from the devices:
  head = headset (x right, y up, z back) turned 180 deg about y; each hand =
  controller grip times a per-hand offset measured once in a T-pose
  (:meth:`HmdPoserLive.calibrate`), where SMPL wrists equal the body frame;
* only the HMD channels are filled; feet and pelvis IMU channels stay zero
  (the checkpoint's 'HMD' input mode, dataloader.TestDataset);
* joint positions come from our own FK on the SMPL-X neutral rest joints
  (``assets/body_models/smplx_neutral_joints22.npz``, same 22-joint tree as
  SMPL+H), placed so the head joint sits at the headset.

Output: pelvis (root) yaw - the body heading the Quest does not track itself -
plus the skeleton for the headset panel.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "assets/hmd_poser/pretrained_model_protocol1.pt"
JOINTS = REPO / "assets/body_models/smplx_neutral_joints22.npz"
MESH = REPO / "assets/body_models/smplx_neutral_mesh22.npz"      # G1_sim scripts/export_smplx_body_assets.py
FPS = 60.0
WINDOW = 40
HEAD, LHAND, RHAND = 15, 20, 21
HEADSET_TO_SMPL = np.diag([-1.0, 1.0, -1.0])     # (right, up, back) -> (left, up, forward)


def body_frame(yaw: float) -> np.ndarray:
    """SMPL canonical axes (left, up, forward) of an upright body facing ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-s, 0.0, c], [c, 0.0, s], [0.0, 1.0, 0.0]])


def sixd(R: np.ndarray) -> np.ndarray:
    """HMD-Poser 6D: first two columns (utils_transform.matrot2sixd)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def from_sixd(d: np.ndarray) -> np.ndarray:
    a, b = d[..., :3], d[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def features(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(T+1, 3, 3, 3) head/lhand/rhand rotations and (T+1, 3, 3) positions -> (T, 135)."""
    T = len(R) - 1
    f = np.zeros((T, 135))
    cur, prev = R[1:], R[:-1]
    f[:, 0:18] = sixd(cur).reshape(T, 18)
    f[:, 36:54] = sixd(np.einsum("tkji,tkjl->tkil", prev, cur)).reshape(T, 18)     # R_prev^T R_cur
    f[:, 72:81] = p[1:].reshape(T, 9)
    f[:, 81:90] = (p[1:] - p[:-1]).reshape(T, 9)
    Rh = R[:, 0]
    in_head = np.einsum("tji,tkjl->tkil", Rh, R[:, 1:])                        # R_head^T R_hand
    f[:, 90:102] = sixd(in_head[1:]).reshape(T, 12)
    f[:, 102:114] = sixd(np.einsum("tkji,tkjl->tkil", in_head[:-1], in_head[1:])).reshape(T, 12)
    ph = np.einsum("tkj,tji->tki", p[:, 1:] - p[:, :1], Rh)                    # R_head^T (p_hand - p_head)
    f[:, 114:120] = ph[1:].reshape(T, 6)
    f[:, 120:126] = (ph[1:] - ph[:-1]).reshape(T, 6)
    return f


class HmdPoserLive:
    """Feed :meth:`push` every frame; :meth:`start` runs inference on a thread;
    :attr:`result` holds the latest estimate (or None)."""

    def __init__(self, weights: Path = WEIGHTS, joints: Path = JOINTS, rate_hz: float = 15.0) -> None:
        import torch

        from k1_teleop.hmd_poser.network import HMD_imu_HME_Universe

        self.torch = torch
        torch.set_num_threads(2)
        self.net = HMD_imu_HME_Universe(135, 3, 256, 0.05, 8, 2)
        state = torch.load(weights, map_location="cpu")
        self.net.load_state_dict(state.get("state_dict", state))
        self.net.eval()
        j = np.load(joints)
        self.rest, self.parents = j["joints"].astype(np.float64), j["parents"]
        self.mesh = dict(np.load(MESH)) if MESH.exists() else None
        self.offset = {"left": None, "right": None}
        self.z_shift = 0.0
        self.rate_hz = rate_hz
        self._lock = threading.Lock()
        self._R: list[np.ndarray] = []
        self._p: list[np.ndarray] = []
        self._t_next = None
        self.result: dict | None = None
        self.on_result = None                      # called on the inference thread after each step
        self.calibrated = False

    # ── input ──
    def calibrate(self, head: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        """Operator in a T-pose: SMPL wrists = body frame, so offset = grip^T body."""
        yaw = _yaw(head[:3, :3] @ HEADSET_TO_SMPL)
        B = body_frame(yaw)
        self.offset = {"left": left[:3, :3].T @ B, "right": right[:3, :3].T @ B}
        # OpenXR LOCAL space puts the origin at the head; the net expects a floor origin.
        self.z_shift = 1.6 - head[2, 3] if head[2, 3] < 1.0 else 0.0
        with self._lock:
            self._R, self._p, self.calibrated = [], [], True

    def push(self, now: float, head: np.ndarray | None, left: np.ndarray | None, right: np.ndarray | None) -> None:
        """Robot-axes 4x4 device poses; resampled to 60 Hz (zero-order hold)."""
        if not self.calibrated or head is None or left is None or right is None:
            return
        if self._t_next is None or abs(now - self._t_next) > 0.5:      # first sample or clock jump
            self._t_next = now
        R = np.stack([head[:3, :3] @ HEADSET_TO_SMPL, left[:3, :3] @ self.offset["left"],
                      right[:3, :3] @ self.offset["right"]])
        p = np.stack([head[:3, 3], left[:3, 3], right[:3, 3]]) + [0.0, 0.0, self.z_shift]
        with self._lock:
            while self._t_next <= now:
                self._R.append(R)
                self._p.append(p)
                self._t_next += 1.0 / FPS
            del self._R[:-(WINDOW + 1)], self._p[:-(WINDOW + 1)]

    # ── inference ──
    def infer(self) -> dict | None:
        with self._lock:
            if len(self._R) < 2:
                return None
            R, p = np.array(self._R), np.array(self._p)
        t0 = time.perf_counter()
        x = self.torch.from_numpy(features(R, p)).float()[None]
        with self.torch.no_grad():
            pose, _ = self.net(x)
        local = from_sixd(pose[0, -1].numpy().astype(np.float64).reshape(22, 6))
        glob = np.zeros_like(local)
        pos = np.zeros((22, 3))
        for i, par in enumerate(self.parents):
            if par < 0:
                glob[i] = local[i]
            else:
                glob[i] = glob[par] @ local[i]
                pos[i] = pos[par] + glob[par] @ (self.rest[i] - self.rest[par])
        pos += p[-1, 0] - pos[HEAD]                                              # head at the headset
        smpl = local.copy()
        smpl[0] = body_frame(_yaw(glob[0])).T @ glob[0]      # root in SMPL canonical (y up, +z fwd), pelvis yaw removed
        return {"joints": pos, "rot": glob, "smpl": smpl, "pelvis_yaw": _yaw(glob[0]), "head_yaw": _yaw(R[-1, 0]),
                "ms": 1e3 * (time.perf_counter() - t0), "t": time.time()}

    def start(self) -> None:
        def loop():
            while True:
                t = time.time()
                try:
                    self.result = self.infer() or self.result
                except Exception as e:                     # keep teleop alive; surface in the panel
                    self.result = {"error": repr(e), "t": time.time()}
                if self.on_result is not None:
                    self.on_result()
                time.sleep(max(0.0, 1.0 / self.rate_hz - (time.time() - t)))

        threading.Thread(target=loop, daemon=True).start()

    def vertices(self) -> np.ndarray | None:
        """SMPL-X neutral mesh of the latest estimate (linear blend skinning on the
        22 body joints; hands/face follow the wrists/head), or None without the mesh."""
        if self.mesh is None or not self.result or "rot" not in self.result:
            return None
        R, J = self.result["rot"], self.result["joints"]
        v = self.mesh["v_template"].astype(np.float64)
        per_joint = np.einsum("kij,nkj->nki", R, v[:, None, :] - self.rest[None]) + J[None]   # (N, 22, 3)
        return np.einsum("nk,nki->ni", self.mesh["weights"], per_joint)

    def skeleton(self) -> np.ndarray:
        """(21, 2, 3) parent -> child segments of the latest estimate."""
        j = self.result["joints"]
        return np.array([(j[par], j[i]) for i, par in enumerate(self.parents) if par >= 0])


def _yaw(R: np.ndarray) -> float:
    """Heading of an SMPL frame: its forward (z) column in the ground plane."""
    return math.atan2(R[1, 2], R[0, 2])
