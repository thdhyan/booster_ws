"""AGILE T1-parity rewards for the K1 velocity task.

Ports of the NVIDIA AGILE locomotion rewards that Isaac Lab's ``mdp`` module does
not ship, taken from AGILE's own source at
``https://github.com/nvidia-isaac/WBC-AGILE`` (see ``LICENSE``):

* ``agile/rl_env/mdp/rewards/task_rewards.py``
  -- ``track_lin_vel_xy_yaw_frame_exp_weighted_simplified``,
     ``track_ang_vel_z_world_exp_weighted_simplified``
* ``agile/rl_env/mdp/rewards/aestetic_rewards.py`` -- ``jumping``

Weights come from AGILE's Booster T1 locomotion task
(``agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py``), which is the closest
reference NVIDIA ships to a K1: same 22-DoF class, same 12 controlled leg joints,
same ankle-roll actuator caveat, same ``Trunk`` base link name.

WHY THE WEIGHTED TRACKER
------------------------
The unweighted exponential tracker that ``velocity_env_cfg`` used before this
module existed returns ``exp(-err / std**2)`` with no regard for *how fast* the
command asked the robot to go. Measured on the retrain of 2026-10-01, the policy
settled at **0.09-0.31 m/s against a 0.5 m/s command** -- it tracked a large
fraction of the error and was paid for it, because creeping slowly is nearly as
good as arriving.

AGILE's weighted variant multiplies the exponential by a factor interpolated on
the commanded speed, 1.0 at rest to 2.0 at the top of the range, so the fast
commands the curriculum is now asking for are worth roughly twice what the slow
ones are. That is the term that makes "walk at the speed you were told" cheaper
than "walk slowly and collect a partial-credit exponential".

PORT NOTES (deviations from AGILE, all deliberate and all load-bearing)
----------------------------------------------------------------------
1. **``min_vel_norm`` is AGILE-only.** AGILE's
   ``UniformNullVelocityCommandCfg`` defines ``min_vel_norm`` (default 0.1) and
   zeroes commands below it, so AGILE can use it as the low end of the weight
   ramp. We use Isaac Lab's ``UniformVelocityCommandCfg``, which has no such
   field, and we *deliberately* keep standing environments (they carry the
   upright signal that ``base_height`` depends on). ``min_vel_norm`` is therefore
   a parameter here, defaulting to the same 0.1, and must never be used to zero
   commands.
2. **The ramp endpoints follow the live command range, not the initial one.**
   AGILE computes ``vel_max`` from ``cfg.ranges`` once, per call. Our
   ``VelocityRangeCurriculumTerm`` *mutates those same ranges at runtime* to widen
   from +/-0.5 to +/-4.0 m/s, so a ramp pinned to the initial range would go flat
   the moment the curriculum moved and stop rewarding speed -- the exact failure
   this term exists to fix. Reading ``cfg.ranges`` on every call keeps the ramp
   spanning whatever the curriculum currently asks for.
3. **Guarded denominator.** If the commanded range is degenerate (both axes at
   zero) the normalised magnitude divides by zero; this raises instead of
   silently returning weight 1.0 forever.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _cmd_term(env, command_name: str):
    """The CommandTerm itself (not the command tensor).

    ``command_manager.get_command(name)`` returns the tensor; the term carries the
    ``cfg`` the ramp needs. See ``velocity_curriculum.resolve_ranges`` for the
    same distinction.
    """
    mgr = env.command_manager
    getter = getattr(mgr, "get_term", None)
    if getter is not None:
        return getter(command_name)
    return mgr._terms[command_name]


def _weight_from_magnitude(mag: torch.Tensor, lo: float, hi: float,
                           weight_min: float = 1.0, weight_max: float = 2.0) -> torch.Tensor:
    """Linear ramp ``weight_min -> weight_max`` over commanded magnitude ``lo -> hi``."""
    if not hi > lo:
        raise ValueError(
            f"weighted tracking needs hi > lo, got lo={lo}, hi={hi}. The commanded "
            "velocity range is degenerate, so the weight ramp has no span and the "
            "term would silently pin to weight_min."
        )
    clamped = mag.clamp(lo, hi)
    normalized = (clamped - lo) / (hi - lo)
    return weight_min + (weight_max - weight_min) * normalized


def track_lin_vel_xy_exp_weighted(
    env,
    command_name: str,
    std: float = 0.2,
    min_vel_norm: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AGILE ``track_lin_vel_xy_yaw_frame_exp_weighted_simplified``.

    Yaw-frame exponential velocity tracking, scaled 1.0 -> 2.0 by how fast the
    command asks the robot to go.
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3]
    )
    term = _cmd_term(env, command_name)
    vel_cmd = term.vel_command_b[:, :2]

    lin_vel_error = torch.sum(torch.square(vel_cmd - vel_yaw[:, :2]), dim=1)

    # Read the LIVE range: the velocity curriculum rewrites cfg.ranges during
    # training (see module docstring, note 2).
    ranges = term.cfg.ranges
    lo = float(min_vel_norm)
    hi = float(
        torch.linalg.norm(
            torch.tensor([ranges.lin_vel_x[1], ranges.lin_vel_y[1]], dtype=torch.float32)
        ).item()
    )
    weight = _weight_from_magnitude(torch.linalg.norm(vel_cmd, dim=1), lo, hi)
    return weight * torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp_weighted(
    env,
    command_name: str,
    std: float = 0.2,
    min_vel_norm: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AGILE ``track_ang_vel_z_world_exp_weighted_simplified``.

    World-frame yaw-rate tracking, scaled 1.0 -> 2.0 by commanded yaw magnitude.
    """
    asset = env.scene[asset_cfg.name]
    term = _cmd_term(env, command_name)

    ang_vel_cmd = term.vel_command_b[:, 2]
    ang_vel_actual = asset.data.root_ang_vel_w.torch[:, 2]
    ang_vel_error = torch.square(ang_vel_cmd - ang_vel_actual)

    ranges = term.cfg.ranges
    lo = float(min_vel_norm)
    hi = abs(float(ranges.ang_vel_z[1]))
    weight = _weight_from_magnitude(torch.abs(ang_vel_cmd), lo, hi)
    return weight * torch.exp(-ang_vel_error / std**2)


def jumping(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """AGILE ``jumping`` -- penalise both feet leaving the ground at once.

    Returns 1.0 for an env whose feet are *all* out of contact, else 0.0, so it
    is wired with a negative weight. This is the ceiling term on the gait
    structure: ``phase_swing`` already makes double flight expensive by
    construction, but only through the contact channel; this one is independent of
    the clock, so a policy cannot satisfy it by mistiming the reference.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=2)
    return (feet_forces < threshold).all(dim=1).float()