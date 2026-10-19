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

import math

import torch

from isaaclab.managers import SceneEntityCfg

from . import gait_clock


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


def _cadence_steps_per_s(
    env,
    asset_cfg: SceneEntityCfg,
    dt: float = 0.02,
    alpha: float = 0.15,
    knee_index: int = 3,
    window_s: float = 2.5,
) -> torch.Tensor:
    """Cadence estimate shared by all cadence terms: (n,) steps/s.

    One-pole filter on knee joint VELOCITY (not position -- position drifts and
    hides the 4-6 Hz chatter, see gait_cadence_penalty's docstring), then the
    zero-crossing rate of a ring buffer. Cached per env step so that several
    cadence terms (Huber + L1) share ONE filter advance per step; without the
    cache the second caller would advance the EMA twice and halve its time
    constant.
    """
    joints = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    knee = joints[:, knee_index]
    num_envs, device = knee.shape[0], knee.device

    window = max(int(window_s / dt), 8)
    key = f"cadence_{asset_cfg.name}"
    hist = _GaitPhase.get(key, (num_envs, window), device)
    filt = _GaitPhase.get(f"{key}_filt", num_envs, device)
    cache = _GaitPhase.get(f"{key}_hz", num_envs, device)
    stepbuf = _GaitPhase.get(f"{key}_hz_step", 1, device)
    step = float(getattr(env, "common_step_counter", -1))
    if step >= 0.0 and stepbuf.item() == step:
        return cache

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
    cache.copy_(steps_per_s)
    if step >= 0.0:
        stepbuf.fill_(step)
    return cache


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

    Cadence is estimated from zero crossings of a low-pass filtered signal,
    tracked with a one-pole filter and a running crossing rate. This is a cheap
    proxy rather than a true phase-based gait clock: it is stable enough to gate
    on and needs no contact-timing machinery, which is what actually went wrong
    (the policy exploited "roughly forward" instead of stepping).

    The signal is **joint velocity**, and that choice is the whole point. The
    first version read ``joint_pos``, whose slow postural drift (0.07-1.0 Hz in
    the measured trace) dominates the knee's 4-6 Hz stepping oscillation, so the
    estimator read ~1-3 steps/s on a policy gait_gate measured at 6.68: the one
    term meant to price the shuffle was blind to it and paid only -0.394 per
    step. Velocity has no drift -- it oscillates about zero at exactly the step
    frequency -- so its crossings are the cadence. A retrain with the blind
    estimator still shuffled 8/8 (cadence 6.68, jerk 0.099).

    A penalty that is flat for small errors and grows for large ones lets the
    policy settle at the 2.0 target without fighting noise; a standing robot
    flickers across zero velocity and reads ~4 steps/s of noise floor rather
    than 0, so standing prices ~0.90, still well under the shuffle's ~2.47 --
    "stop stepping" is not an exit, but it is cheaper than chattering.
    """
    steps_per_s = _cadence_steps_per_s(env, asset_cfg, dt, alpha, knee_index, window_s)

    # Huber-ish: flat near the target, quadratic in the error beyond it, so the
    # policy can sit at 2 Hz without fighting noise.
    err = (steps_per_s - target_hz).abs()
    return torch.where(err <= 0.5, 0.5 * err.pow(2), 0.125 + 0.5 * (err - 0.5)).mean()


def gait_cadence_l1(
    env,
    asset_cfg: SceneEntityCfg,
    target_hz: float = 2.0,
    dt: float = 0.02,
    alpha: float = 0.15,
    knee_index: int = 3,
    window_s: float = 2.5,
) -> torch.Tensor:
    """L1 regulariser on the step cadence: |steps/s - target_hz|.

    Teacher v3 (2026-10-12): the Huber cadence term is flat within +-0.5 steps/s
    of the target, so near the goal cadence there is no gradient at all, and on
    the far side its slope is halved by the flat zone. The L1 term prices every
    step/s of error with the same constant gradient, the same way the linear
    feet_clearance hinge keeps working at 1 cm and at 5 cm of shortfall.
    """
    hz = _cadence_steps_per_s(env, asset_cfg, dt, alpha, knee_index, window_s)
    return (hz - target_hz).abs().mean()


def feet_clearance(
    env,
    contact_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_lift: float = 0.06,
    foot_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise a swing foot that is not lifted clear of the stance foot.

    WHY THIS MEASURES LIFT, NOT HEIGHT
    ----------------------------------
    The previous version compared each swinging foot's world height against
    ``target_height = 0.06`` m. Measured on the seed-43 student, **the foot link
    origin rests at ~0.059 m**, so at rest ``short = 0.06 - 0.059 = 0.001 m`` and
    ``short**2 = 1e-6``. The term was mathematically pinned near zero in every
    512-env run and could only ever charge a foot for being *below its own normal
    standing height* -- i.e. for sinking. It could not charge a policy for failing
    to lift, which is the failure we actually have.

    That measurement is the whole reason the shuffle survived seven reward terms:

    ====== =========== ============= ==============
    cmd     cadence     LIFT (mean)  swing% (foot unloaded)
    ====== =========== ============= ==============
    0.1     9.65/s      +0.0000 m    63.3%
    0.2     9.63/s      +0.0036 m    70.0%
    0.3     8.80/s      +0.0097 m    73.9%
    0.5     8.53/s      +0.0153 m    79.5%
    ====== =========== ============= ==============

    The feet come off the ground and go straight back down. So the quantity to
    charge for is the **lift**: a swinging foot's height relative to the stance
    foot. That is self-referencing, so it needs no absolute height that has to be
    re-tuned when the robot's geometry or pose changes, and it cannot be
    satisfied by a foot that never rises.

    A stance foot is treated as being at target so it contributes nothing, and an
    env with no swinging foot at all is skipped rather than punished -- that gate
    is what let the old term read a clean 0.0000 and hide the problem.

    Returns the per-env mean squared lift shortfall, in m^2.
    """
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    forces = env.scene.sensors[contact_cfg.name].data.net_forces_w[
        :, contact_cfg.body_ids
    ]
    swing = forces.norm(dim=-1) <= foot_height_threshold

    # Reference each foot against the mean of the OTHERS, so with two feet the
    # swinging foot is measured against the stance foot.
    n_feet = z.shape[1]
    others = (z.sum(dim=1, keepdim=True) - z) / max(n_feet - 1, 1)
    lift = z - others

    # Treat a stance foot as being at target so it contributes nothing.
    scored = torch.where(swing, lift, torch.full_like(lift, target_lift))
    short = (target_lift - scored).clamp(min=0.0)
    # Only score envs that have at least one swinging foot.
    active = swing.any(dim=1).float()
    # LINEAR in the shortfall, not squared. The measured shortfall is ~3.3 cm,
    # and (0.033)**2 = 1.1e-3, so a squared penalty at weight -1.6 charged
    # -0.0017/step while action_rate_l2 charged -1.95: the heaviest weight in the
    # budget was 1150x weaker than the lightest. Squaring a *persistent, large*
    # error is what hid it a second time -- the first time it was the wrong
    # reference height, this time it is the curvature. A linear hinge gives the
    # same constant gradient whether the foot is 1 cm or 5 cm short, which is what
    # a failure this persistent needs, and it keeps the weight human-scale.
    return (short.mean(dim=1) * active).mean()


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
    """Penalise fore-aft foot separation far below ``target_stride``.

    Directly targets the 3-5 cm strides measured in the chatter mode. The
    separation is projected onto the body's forward axis, because plain
    horizontal distance counts lateral splay: after the gait-v2 run the reward
    saw 0.21 m of foot separation while the gate measured a 0.07 m stride, so
    the policy had widened its stance instead of stepping. Only the fore-aft
    component is a stride. Capped above the target so a long lunge is not
    penalised -- that was a different observed failure and needs its own term.
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    # Unit forward vector from the root yaw, so the projection follows the body
    # frame as the robot turns instead of the world axes.
    yaw = _yaw_from_quat(robot.data.root_quat_w)
    fwd = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
    along = ((pos[:, 0, :] - pos[:, 1, :]) * fwd).sum(dim=-1).abs()
    short = (target_stride - along).clamp(min=0.0)
    return short.pow(2).mean()


def phase_synced_swing(
    env,
    contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise each foot being grounded during its swing window and vice versa.

    The phase clock was added as gait lever 3 so the policy would have a time
    base to step on, and it works -- the recorded retrain shows an exact 1.001
    Hz ramp -- but nothing in the objective consumed it: max |corr(action,
    clock)| was 0.098 over every joint, so PPO read the clock and ignored it.
    This term is that consumer, and it is the reason to read it.

    The left foot owes contact over phase [0, 0.5) and freedom over [0.5, 1),
    the right the reverse (``body_names`` order is [left, right]). That
    partition makes double support and double flight both expensive and leaves
    alternating single support as the cheap state -- a gait locked to the
    1 Hz reference, i.e. 2.0 steps/s, in the middle of the gate's [1.2, 4.0]
    band by construction. Contact is measured rather than foot height so the
    term holds on slopes, where world z lies about clearance.

    Returns the per-env fraction of feet out of step with their window, in
    [0, 1]: 0 = fully in sync with the clock.
    """
    phase = gait_clock.get_phase(env)
    forces = env.scene.sensors[contact_cfg.name].data.net_forces_w[
        :, contact_cfg.body_ids
    ]
    contact = forces.norm(dim=-1) > 1.0
    # left foot (index 0) swings on the second half-cycle, right on the first.
    left_swing = phase >= 0.5
    swing = torch.stack((left_swing, ~left_swing), dim=1)
    # In contact while owing swing, or airborne while owing stance, is the error.
    return (swing == contact).float().mean(dim=1)


def phase_locked_stride(
    env,
    asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_stride: float = 0.22,
) -> torch.Tensor:
    """Penalise fore-aft foot placement that disagrees with the gait clock.

    WHY THIS EXISTS
    ---------------
    ``gait_cadence_penalty`` cannot fix the measured shuffle, and the 5000-
    iteration run proved it: cadence 4.85 steps/s (gate wants 1.2-4.0), jerk
    0.114, 8/8 fail. The reason is that a *rate* penalty is bistable. Both 0.4
    steps/s (standing) and 11 steps/s (chatter) are "far from 2.0", so the policy
    picks whichever is cheaper given that tracking pays +1.28/step against ~0.7
    of combined gait penalty. No weight fixes an objective with no preferred
    basin.

    Sweeping an independent policy (``model_6321``, gated 8/8 at four speeds)
    reproduces the same bistability and adds the sign that fixes the clock:

    ====== =========== =========== =========
    cmd     cadence      stride      jerk
    ====== =========== =========== =========
    0.1      8.48/s     0.011 m    0.064
    0.2      7.83/s     0.022 m    0.074
    0.3      7.57/s     0.037 m    0.086
    0.5      5.43/s     0.096 m    0.112
    ====== =========== =========== =========

    Cadence *rises* as the command falls, and the stride collapses to 11 mm --
    smaller than the foot -- while speed tracking stays within 0.01 m/s. That is
    a buzz that steps nowhere, and it is why the clock is command-coupled: a
    fixed-rate reference cannot ask for 11 cm of travel at 0.1 m/s without
    fighting the tracking term, which is the failure mode
    ``UniformVelocityGaitBaseHeightCommand`` exists to avoid.

    ``phase_synced_swing`` already locks *contact* to the clock and also failed,
    which is the evidence that contact is too weak a channel: contact is a
    binary predicate, satisfied by a 4 cm twitch as readily as by a real stride.

    So this locks *position*. The reference is a sinusoid in the clock's phase:
    each foot's fore-aft offset from the hip should follow
    ``target_stride * sin(2*pi*phase)``, left and right half a cycle apart. Slow
    and fast are then both expensive -- a foot that does not travel the
    commanded amplitude is wrong regardless of how often it moves -- and only a
    stride at the clock's frequency survives.

    Amplitude is the stride the command implies -- ``v / (2 * hz)`` -- read from
    the same command-coupled clock the phase advances on, so demanding the same
    absolute stride at 0.1 and 0.5 m/s would again fight velocity tracking.
    ``target_stride`` is only the fallback when no command term is readable.

    Returns the per-env mean squared fore-aft placement error, in m^2.
    """
    robot = env.scene[asset_cfg.name]
    phase = gait_clock.get_phase(env)
    hz = gait_clock.get_frequency(env)
    # Reference amplitude per env: the stride the command itself implies.
    #
    # stride = v / steps_per_second, and the clock's hz is half the step rate
    # (one full cycle is two steps), so amp = v / (2 * hz). Dimensionally this is
    # the only correct form: (m/s) / (1/s) = m.
    #
    # This REPLACED ``target_stride / hz``, which was wrong twice over. It is a
    # dimensional error -- m / (1/s) = m*s, not a length -- that looks correct at
    # exactly 1 Hz (0.22/1 = 0.22 m) and is wrong everywhere else. And it inverts
    # the physics: dividing makes a *slower* clock demand a *bigger* step, while a
    # real gait's stride grows with speed.
    #
    # Measured cost of that, 5000 iterations with the old law, gated at 0.1 m/s:
    # mean cadence 10.65 steps/s against a *demanded* 1.2, stride 8-16 mm, mean
    # jerk 0.102 -- worse than the 8.48 / 0.064 of the model it was meant to fix.
    # The mechanism is visible in the numbers: the clock clamps to MIN_HZ=0.6 for
    # every command from 0.1 to 0.3 m/s, so the old law demanded a 0.22/0.6 =
    # 0.367 m excursion there, needing ~0.88 m/s of foot speed while the base
    # moved at 0.1 m/s. The policy ran *faster* trying to chase a reference it
    # could not reach, which is the opposite of the fix.
    speed = gait_clock.command_magnitude(env)
    if speed is None:
        # No readable command (config probing): fall back to a plausible fixed
        # excursion rather than guessing a scale from the clock alone.
        amp = torch.full_like(hz, target_stride)
    else:
        amp = speed / (2.0 * hz.clamp(min=0.1))

    wave = torch.sin(2.0 * math.pi * phase)              # (num_envs,)
    # Left foot leads the cycle, right foot is half a cycle behind.
    reference = torch.stack((wave, -wave), dim=1) * amp.unsqueeze(1)   # (N, 2)

    pos = robot.data.body_pos_w[:, asset_cfg.body_ids]                 # (N, 2, 3)
    hip = robot.data.body_pos_w[:, base_body_cfg.body_ids]             # (N, 1, 3)

    # Yaw-aligned forward axis: a fixed world +x would call "forward" whatever
    # direction the robot happens to face, which breaks the moment it turns.
    # Built in the world frame with a zero z component, so it is (N, 1, 3) and
    # broadcasts against the (N, 2, 3) foot offsets -- a 2-component axis here
    # raises "size of tensor a (3) must match tensor b (2)".
    quat = robot.data.body_quat_w[:, base_body_cfg.body_ids]
    w, x, y, z = quat.unbind(-1)
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    fwd = torch.stack((fwd_x, fwd_y, torch.zeros_like(fwd_x)), dim=-1)   # (N, 1, 3)

    along = ((pos - hip) * fwd).sum(dim=-1)                             # (N, 2)
    return (along - reference).pow(2).mean(dim=1)


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


def _sole_level(foot_quat: torch.Tensor, std: float) -> torch.Tensor:
    """exp(-|g_xy|^2 / std^2) of gravity in each foot frame: 1 for a level sole.

    The K1 foot links have identity joint origins, so the foot frame's xy plane is
    the sole plane; any pitch (heel or toe up) or roll tilts gravity into xy.
    """
    w, x, y, z = foot_quat.unbind(-1)
    # third row of R^T (gravity (0,0,-1) expressed in the foot frame), xy part
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    return torch.exp(-(gx * gx + gy * gy) / std**2)


def feet_flat_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float = 0.15,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward (>= 0, max 1) for loaded feet whose sole is flat on the ground.

    Per foot: [contact force > threshold] * level(sole), averaged over the feet.
    Standing on the heel/ankle edge or the toes pitches the sole and loses the
    credit; ``feet_roll_l2`` only charged roll, so pitch was free. Positive weight.
    Flat-ground assumption: on slopes/stairs edges the level term under-pays a
    little, which is fine for a shaping term.
    """
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hist = sensor.data.net_forces_w_history
    hist = hist.torch if hasattr(hist, "torch") else hist        # Isaac Lab 3 warp arrays
    quat = robot.data.body_quat_w
    quat = quat.torch if hasattr(quat, "torch") else quat
    forces = hist[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    loaded = (forces > force_threshold).to(forces.dtype)
    level = _sole_level(quat[:, asset_cfg.body_ids, :], std)
    return (loaded * level).mean(dim=1)
