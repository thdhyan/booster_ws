"""AGILE-derived gait-shaping rewards for the K1 velocity task.

Is Isaac Lab's locomotion ``mdp`` has no foot-posture or foot-spacing terms, and
NVIDIA's AGILE velocity task for the Booster T1 (``agile/rl_env/mdp/rewards``,
``agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py``) leans on exactly
those instead of an air-time reward.  AGILE deliberately omits ``feet_air_time``
-- air-time is a classic degenerate attractor (hop, or never load one foot) --
and replaces it with foot slip / roll / yaw / spacing shaping.

Every function here returns a *penalty* (>= 0, 0 = ideal) so it can be wired
straight into ``RewTerm`` with a negative weight, matching AGILE's convention.
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Planar yaw (rotation about world +z) of a wxyz quaternion."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Roll (rotation about the body's forward axis) of a wxyz quaternion."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def feet_yaw_mean_vs_base(
    env,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise feet that drift out of the base's heading (AGILE ``feet_yaw_mean``).

    Without this a policy can turn a foot sideways, or track with one foot
    rotated, and still satisfy the velocity tracker.
    """
    robot = env.scene[asset_cfg.name]
    foot_quat = robot.data.body_quat_w[:, feet_asset_cfg.body_ids, :]
    base_quat = robot.data.body_quat_w[:, base_body_cfg.body_ids, :]
    rel = _yaw_from_quat(foot_quat) - _yaw_from_quat(base_quat)
    rel = torch.atan2(torch.sin(rel), torch.cos(rel))  # wrap to (-pi, pi]
    return rel.abs().mean(dim=1)


def feet_yaw_diff_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared difference between the two feet's world yaw (AGILE ``feet_yaw_diff``)."""
    robot = env.scene[asset_cfg.name]
    yaw = _yaw_from_quat(robot.data.body_quat_w[:, asset_cfg.body_ids, :])
    return torch.square(yaw[:, 0] - yaw[:, 1])


def feet_roll_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared foot roll -- keeps the sole flat instead of toe-walking (AGILE ``feet_roll``)."""
    robot = env.scene[asset_cfg.name]
    roll = _roll_from_quat(robot.data.body_quat_w[:, asset_cfg.body_ids, :])
    return torch.square(roll).mean(dim=1)


def feet_distance_from_ref(env, asset_cfg: SceneEntityCfg, ref_distance: float) -> torch.Tensor:
    """Squared error of the two feet's horizontal separation from ``ref_distance``.

    Too narrow -> the legs scissor and collide; too wide -> a wide, splayed,
    robot-waddle gait.  AGILE uses 0.2 m on the T1.
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    sep = torch.linalg.norm(pos[:, 0, :2] - pos[:, 1, :2], dim=-1)
    return torch.square(sep - ref_distance)


def base_height_exp(
    env,
    target_height: float,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Positive reward for holding the trunk at ``target_height`` (AGILE).

    AGILE pays for uprightness (``+2.0``) *and* penalises tilt, rather than
    charging a large L2 bill for height.  That distinction matters: with a
    negative height penalty, every metre travelled has to be paid for out of the
    same budget as the task reward, and "stand still" becomes a strong local
    optimum.  Measured: the first teacher learned to stand flawlessly (0 % falls
    over 900 steps) and walked 0.17 m in 36 s against a 0.4 m/s command.

    ``sensor_cfg`` shifts the target by the terrain height under the trunk so
    this stays correct on rough ground.
    """
    robot = env.scene[asset_cfg.name]
    target = torch.full_like(robot.data.root_pos_w[:, 2], float(target_height))
    if sensor_cfg is not None:
        sensor = env.scene.sensors[sensor_cfg.name]
        target = target + sensor.data.ray_hits_w.torch[..., 2].mean(dim=1)
    err = robot.data.root_pos_w[:, 2] - target
    return torch.exp(-(err * err) / (std * std))


# ---------------------------------------------------------------------------
# Gait structure: cadence, clearance, alternation, stride
# ---------------------------------------------------------------------------
# WHY THESE EXIST
# ---------------
# A 3000-iteration run produced a policy that is stable (0 falls in 750 steps),
# moves (2.66 m net at 0.5 m/s) and yet does not walk. Decoding the play trace
# across 8 envs gave a mean cadence of 5.7 steps/s against 1.8-2.2 for human
# walking, with 5 of 8 envs chattering at 7.9-8.9 steps/s on 3-5 cm strides and
# two more chattering on a single leg while the other was planted.
#
# The reason the reward permitted it: track_lin_vel_xy_yaw_frame_exp uses
# std=0.25, which is nearly flat near the optimum, so a 0.14 m/s tracking error
# costs almost nothing; and nothing in the reward distinguishes a walk from a
# jitter. The policy was being paid to roughly move forward, not to move
# accurately or to move like a human.
#
# These four terms close that gap. All are penalties (>= 0, 0 = ideal) to match
# the convention above, and each targets a specific measured failure.


class _GaitPhase:
    """Per-env low-pass state for gait phase estimation.

    Kept module-level because reward terms are called every step and must not
    allocate. Stores one scalar per env; ``reset`` zeroes it on episode reset so
    a new episode does not inherit phase from the previous one.
    """

    _state: dict = {}

    @classmethod
    def get(cls, key: str, shape, device) -> torch.Tensor:
        """Fetch (or lazily create) a persistent buffer of exactly ``shape``.

        The shape argument matters: per-env phase estimators need a 1-D
        ``(num_envs,)`` scalar, while the jerk term needs the full
        ``(num_envs, num_actions)`` action history. Getting this wrong produced
        "size of tensor a (12) must match the size of tensor b (8)", because a
        1-D ``(8,)`` buffer cannot broadcast against ``(8, 12)``.
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        buf = cls._state.get(key)
        if buf is None or tuple(buf.shape) != shape or buf.device != device:
            buf = torch.zeros(*shape, device=device)
            cls._state[key] = buf
        return buf


def gait_cadence_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    target_hz: float = 2.0,
    dt: float = 0.02,
    alpha: float = 0.15,
    knee_index: int = 3,
    window_s: float = 2.5,
) -> torch.Tensor:
    """Penalise step frequency far from ``target_hz``.

    Cadence is estimated from the dominant frequency of a low-pass filtered
    signal, tracked with a one-pole filter and a running zero-crossing rate. This
    is a cheap proxy rather than a true phase-based gait clock: it is stable
    enough to gate on and needs no contact-timing machinery, which is what
    actually went wrong (the policy exploited "roughly forward" instead of
    stepping).

    The measured failure was ~5.7 steps/s mean against a 2.0 target, with 5 of 8
    envs above 7.9. A penalty that is flat for small errors and grows for large
    ones lets the policy settle at the target without fighting noise.
    """
    # Joint positions live on the articulation's .data, not on the Articulation
    # itself. K1_LEG_JOINTS order is [LHipP,LHipR,LHipY,LKnee,LAnkP,LAnkR, ...],
    # so the knees are indices 3 and 9. Do not derive this from numel(): that
    # gives Right_Hip_Pitch, which does not oscillate once per step.
    joints = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    knee = joints[:, knee_index]
    num_envs = knee.shape[0]

    # Cadence needs a history of the signal, not one scalar per env. A 1-D
    # (num_envs,) buffer made sign[:, 1:] 1-D and mean(dim=1) raised
    # "too many indices for tensor of dimension 1". So keep a ring buffer of the
    # last `window` filtered samples per env: 64 samples is 1.28 s at 50 Hz,
    # which spans ~2.5 cycles of a 2 Hz gait -- enough for a stable estimate.
    window = max(int(window_s / dt), 8)
    key = f"cadence_{asset_cfg.name}"
    hist = _GaitPhase.get(key, (num_envs, window), knee.device)
    filt = _GaitPhase.get(f"{key}_filt", num_envs, knee.device)
    filt.mul_(1.0 - alpha).add_(alpha * knee)
    # Roll rather than an overlapping slice-assign, which is undefined when the
    # source and destination overlap.
    hist.copy_(torch.roll(hist, shifts=1, dims=1))
    hist[:, 0] = filt

    centered = hist - hist.mean(dim=1, keepdim=True)
    sign = torch.sign(centered)
    # fraction of adjacent sample pairs that changed sign
    frac = (sign[:, 1:] * sign[:, :-1] < 0).float().mean(dim=1)
    # Two crossings per gait cycle, and the window spans window*dt seconds:
    #   steps/s = (2 crossings/cycle) / 2 * frac * (window-1) / (window*dt)
    steps_per_s = frac * (window - 1) / (window * dt)

    # Huber-ish: flat near the target, quadratic in the error beyond it, so the
    # policy can sit at 2 Hz without fighting noise.
    err = (steps_per_s - target_hz).abs()
    return torch.where(err <= 0.5, 0.5 * err.pow(2), 0.125 + 0.5 * (err - 0.5)).mean()


def feet_clearance(
    env,
    contact_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.06,
    foot_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise a swing foot that stays below ``target_height`` off the ground.

    Guards against the shuffle directly: with 3-5 cm strides the feet barely
    leave the floor. Only feet that are genuinely swinging are scored, so the
    policy is not penalised for the stance foot staying down, and an env with no
    swing foot at all is skipped rather than punished.
    """
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    forces = env.scene.sensors[contact_cfg.name].data.net_forces_w[
        :, contact_cfg.body_ids
    ]
    swing = forces.norm(dim=-1) <= foot_height_threshold
    # Treat a stance foot as being at target so it contributes nothing.
    scored = torch.where(swing, feet, torch.full_like(feet, target_height))
    short = (target_height - scored).clamp(min=0.0)
    # Only score envs that have at least one swinging foot.
    active = swing.any(dim=1).float()
    return (short.pow(2).mean(dim=1) * active).mean()


def feet_alternation_penalty(
    env,
    contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise both feet being in contact at the same time for long stretches.

    Measured failure: two envs chattered on one leg while the other stayed
    planted, which reads as hopping. Human walking is mostly single-support with
    brief double-support. A sustained both-feet-down or both-feet-up state is the
    signature of a degenerate gait, so penalise the duration of contact
    symmetry rather than trying to reconstruct a full gait clock.
    """
    forces = env.scene.sensors[contact_cfg.name].data.net_forces_w[
        :, contact_cfg.body_ids
    ].norm(dim=-1)
    contact = (forces > 1.0).float()
    both = (contact.sum(dim=1) >= 2).float()
    key = f"alt_{contact_cfg.name}"
    acc = _GaitPhase.get(key, both.shape[0], both.device)
    # Long-run fraction of time spent with both feet planted.
    acc.mul_(0.98).add_(0.02 * both)
    return acc.mean()


def stride_length_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    target_stride: float = 0.35,
) -> torch.Tensor:
    """Penalise horizontal foot separation far below ``target_stride``.

    Directly targets the 3-5 cm strides measured in the chatter mode. Capped
    above the target so a long lunge is not penalised -- that was a different
    observed failure and needs its own term.
    """
    pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, :2]
    sep = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
    short = (target_stride - sep).clamp(min=0.0)
    return short.pow(2).mean()


def action_jerk_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared third difference of the joint targets -- an explicit jerk penalty.

    ``action_rate_l2`` only sees the first difference, which an 8 Hz chatter
    satisfies cheaply. Jerk concentrated in hip-yaw (0.26) and knee (0.19)
    rad/step^2 in the measured failure, and the third difference is what
    penalises that specifically.

    Two previous action buffers are kept so this is a true third difference
    rather than a difference of differences of already-differenced data.
    """
    actions = env.action_manager.action
    flat = actions.reshape(actions.shape[0], -1)          # (num_envs, num_actions)
    prev = _GaitPhase.get(f"jerk_prev1_{asset_cfg.name}", flat.shape, flat.device)
    prev2 = _GaitPhase.get(f"jerk_prev2_{asset_cfg.name}", flat.shape, flat.device)
    third = flat - 3.0 * prev + 3.0 * prev2
    prev2.copy_(prev)
    prev.copy_(flat)
    return third.pow(2).mean()
