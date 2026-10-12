"""Phase-1 squat-base MDP terms: commanded trunk height (H*) reward, floor, depth curriculum.

Context (TRACK B / P6 next pipeline, Phase-0 degenerate audit 2026-09-28):
the P2 velocity teacher tracks (vx, vy, wz) only — trunk height is a *fixed*
cfg scalar, so the base cannot follow a VR operator squatting down (the teleop
pipeline drives a height channel; BASE_LIMITS in
``~/Projects/wt-vr-teleop/k1_teleop/teleop_mapping.py``). Phase 1 adds a
commanded trunk height H* as a 4th command channel and re-grounds the height
reward/termination on it, so ONE policy walks, stands and holds a squat.

Approved audit defaults baked in here:
* Height reward: Laplacian ``exp(-|z - H*| / std)``, std = 0.05 m. The
  validated ``gait.base_height_exp`` is a Gaussian with std = 0.1 — at 5 cm
  error it still pays 0.78 of the max, far too forgiving to hold a depth
  inside the 2 cm eval gate. The heavier-tailed kernel keeps a real gradient
  the whole way down the ~0.17 m command range.
* Termination: squat-commanded envs get a *relative* floor ``H* - 0.05`` on a
  terrain-relative basis (same basis as the reward). The static 0.35 m floor
  cannot see a squat collapse: commanded 0.45, sinking to 0.40 is already
  failing but still clears 0.35. Stand-commanded envs keep the validated
  world-frame 0.35 m floor bit-for-bit so the velocity-parity gate compares
  against an unchanged baseline.
* Depth progression: iteration floors + EMA gates (see
  :func:`squat_depth_curriculum`) — the audit's "stand back up >= 95%"
  requirement is proxied by the height-error EMA over squat envs: falls and
  failed rises keep that EMA high, so the next, deeper stage stays locked.

Isaac Lab contract notes (verified against BOTH the dl checkout
``IsaacLab-ea`` v3.0.0-EA and the zz-bw SIF ``isaac-lab 3.0.0b2``):
* ``CurriculumManager.compute`` calls ``func(env, env_ids, **params)`` every
  step, and ``manager_base._resolve_common_term_cfg`` REQUIRES class terms to
  subclass ``ManagerTermBase`` (it raises TypeError otherwise). This
  curriculum is therefore a plain function with all-default parameters — the
  only form both builds accept — and its state lives on the command term.
* Reward/termination terms are plain functions called as ``func(env, **params)``.
* No Isaac imports on purpose: the module stays importable/lintable outside
  the simulator (same reasoning as ``velocity_curriculum.py``).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cmd_term(env, command_name: str):
    """Resolve the command term, loudly if it is not the squat command."""
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "height_command"):
        raise AttributeError(
            f"command term '{command_name}' is a {type(term).__name__} without height_command; "
            f"base_height_cmd_exp / root_height_below_commanded_floor / squat_depth_curriculum "
            f"require SquatVelocityHeightCommand"
        )
    return term


def _ray_mean_z(env, sensor_name: str) -> torch.Tensor:
    """(n,) mean world-z of the height-scanner rays: terrain height under the trunk.

    This is exactly the quantity ``gait.base_height_exp`` adds to its scalar
    target, so reward, termination and the command metric share one height
    basis across flat and rough tiles.
    """
    sensor = env.scene.sensors.get(sensor_name)
    if sensor is None:
        raise KeyError(
            f"scene has no sensor '{sensor_name}' (needed for terrain-relative trunk height)"
        )
    return sensor.data.ray_hits_w.torch[..., 2].mean(dim=1)


def trunk_height_rel(env, sensor_name: str = "height_scanner") -> torch.Tensor:
    """(n,) terrain-relative trunk height = world z - mean terrain z under the trunk."""
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] - _ray_mean_z(env, sensor_name)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------
def base_height_cmd_exp(
    env,
    command_name: str = "base_velocity",
    std: float = 0.05,
    sensor_name: str = "height_scanner",
) -> torch.Tensor:
    """Per-env exponential kernel on ``|terrain-relative trunk height - H*|``.

    Drop-in replacement for ``gait.base_height_exp`` (scalar target, Gaussian)
    in which the target comes from the squat command term, so every env is
    scored against ITS commanded height. Weight stays +2.0 (unchanged budget);
    only the target source and the kernel tightness change.
    """
    term = _cmd_term(env, command_name)
    err = trunk_height_rel(env, sensor_name) - term.height_command
    return torch.exp(-torch.abs(err) / float(std))


# ---------------------------------------------------------------------------
# termination
# ---------------------------------------------------------------------------
def root_height_below_commanded_floor(
    env,
    command_name: str = "base_velocity",
    margin: float = 0.05,
    minimum_height: float = 0.35,
    sensor_name: str = "height_scanner",
) -> torch.Tensor:
    """Terminate when the trunk sinks below its commanded floor.

    * Squat-commanded envs: terrain-relative floor
      ``max(H* - margin, minimum_height)`` — a robot told to squat to 0.45 that
      collapses past 0.40 dies immediately instead of surviving all the way
      down to the 0.35 m collapse floor. The terrain basis (not world z) is
      what keeps this correct on raised curriculum tiles, where a world-frame
      ``H* - 0.05`` floor would sit metres above the robot and never fire.
    * Stand-commanded envs: world-frame ``z < minimum_height`` — literally the
      validated ``mdp.root_height_below_minimum(minimum_height=0.35)`` term, so
      baseline fall behaviour is bit-identical for the velocity-parity gate.
      The two bases coincide on the flat training tiles (ray z ~ 0).
    """
    term = _cmd_term(env, command_name)
    z_w = env.scene["robot"].data.root_pos_w[:, 2]
    rel_floor = torch.clamp(term.height_command - float(margin), min=float(minimum_height))
    floor_w = torch.where(
        term.is_squat_env,
        rel_floor + _ray_mean_z(env, sensor_name),
        torch.full_like(z_w, float(minimum_height)),
    )
    return z_w < floor_w


# ---------------------------------------------------------------------------
# curriculum
# ---------------------------------------------------------------------------
def squat_depth_curriculum(
    env,
    env_ids,
    command_name: str = "base_velocity",
    steps_per_iter: int = 24,
    stage_iters: tuple = (500, 1100, 1900, 2700, 3500),
    stage_rel_squat: tuple = (0.10, 0.25, 0.25, 0.25, 0.25),
    stage_height_range: tuple = (
        (0.50, 0.55),
        (0.50, 0.55),
        (0.46, 0.53),
        (0.42, 0.50),
        (0.40, 0.48),
    ),
    stage_gates: tuple = ("vel", "height", "height", "height", "height"),
    height_gate: float = 0.025,
    vel_gate: float = 0.30,
) -> dict | None:
    """Step the squat fraction/depth forward, gated on the command term's EMAs.

    Stage ``k`` (0-based *transition* index) is applied once, when
    ``iteration >= stage_iters[k]`` AND its gate passes:

    * ``"vel"``   — ``vel_err_ema <= vel_gate``: don't stack squats on top of
      an unlearned gait (the first transition is the only one gated on
      velocity; it starts from 1.0 and falls as tracking improves).
    * ``"height"`` — ``height_err_ema <= height_gate``: the previous depth is
      actually being held. Squat envs mid-transition (just resampled from
      stand) spike this EMA, so the gate is 2.5 cm — a hair looser than the
      2 cm *eval* gate, which measures whole episodes.
    * ``None``    — always (reserved).

    State (``squat_stage``, the EMAs) lives on the command term; this stays a
    plain function because the curriculum manager of both target builds
    rejects plain class terms (see module docstring). The returned dict is
    logged as ``Curriculum/squat_depth/*`` in TensorBoard and wandb — watch
    ``stage``/``height_ema`` there to see the gates open.

    The gates self-serialize: before stage 0 applies there are no squat envs,
    so ``height_err_ema`` stays at its pessimistic 1.0 init and every
    height-gated transition stays locked until squats exist and are held.
    """
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "squat_stage"):
        return None

    # Iteration via physics steps — same idiom as push_mdp.goal_dist_curriculum.
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter

    stage = int(term.squat_stage)
    while stage < len(stage_iters):
        if iteration < float(stage_iters[stage]):
            break
        gate = stage_gates[stage]
        if gate == "vel":
            ok = term.vel_err_ema <= float(vel_gate)
        elif gate == "height":
            ok = term.height_err_ema <= float(height_gate)
        else:
            ok = True
        if not ok:
            break
        lo, hi = stage_height_range[stage]
        term.cfg.rel_squat_envs = float(stage_rel_squat[stage])
        term.cfg.squat_height_range = (float(lo), float(hi))
        stage += 1
        term.squat_stage = stage
        print(
            f"[squat-curriculum] iter {iteration:.0f}: -> stage {stage} "
            f"rel_squat={term.cfg.rel_squat_envs:.2f} "
            f"height_range={term.cfg.squat_height_range} "
            f"(vel_ema={term.vel_err_ema:.3f}, height_ema={term.height_err_ema:.4f})",
            flush=True,
        )

    return {
        "stage": int(term.squat_stage),
        "rel_squat": float(term.cfg.rel_squat_envs),
        "h_low": float(term.cfg.squat_height_range[0]),
        "h_high": float(term.cfg.squat_height_range[1]),
        "vel_ema": float(term.vel_err_ema),
        "height_ema": float(term.height_err_ema),
        "iteration": float(iteration),
    }


def feet_unloaded_when_still(
    env,
    sensor_cfg,
    command_name: str = "base_velocity",
    force_threshold: float = 5.0,
    still_speed: float = 0.05,
) -> torch.Tensor:
    """(n,) fraction of feet OFF the ground while the velocity command is ~zero.

    Negative weight. Squat stage A commands zero velocity and wants both feet planted;
    the gait clock never stops (MIN_HZ > 0), so the frozen student would otherwise keep
    stepping in place under the squat.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hist = sensor.data.net_forces_w_history
    hist = hist.torch if hasattr(hist, "torch") else hist
    forces = hist[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    unloaded = (forces <= force_threshold).float().mean(dim=1)
    cmd = env.command_manager.get_command(command_name)
    still = (cmd[:, :2].norm(dim=-1) < still_speed) & (cmd[:, 2].abs() < still_speed)
    return unloaded * still.float()


def squat_height_then_velocity_curriculum(
    env,
    env_ids,
    command_name: str = "base_velocity",
    steps_per_iter: int = 24,
    rel_squat: float = 0.6,
    height_stages: tuple = ((0.50, 0.55), (0.46, 0.55), (0.43, 0.55), (0.40, 0.55)),
    vel_scales: tuple = (0.25, 0.5, 0.75, 1.0),
    vel_full: tuple = ((-0.5, 0.5), (-0.3, 0.3), (-0.8, 0.8)),
    vel_rel_standing: float = 0.1,
    min_stage_iters: int = 200,
    height_gate: float = 0.025,
    vel_gate: float = 0.20,
) -> dict | None:
    """Two curricula in sequence (squat residual).

    A. Height, zero velocity: every env gets a zero velocity command
       (``rel_standing_envs`` = 1), ``rel_squat`` of them a squat H* from
       ``height_stages[k]``. Next depth once ``height_err_ema <= height_gate``.
    B. Velocity: after the deepest height stage is held, the velocity ranges grow
       ``vel_scales[k] * vel_full`` (VR limits) with ``vel_rel_standing`` standing envs,
       gated on BOTH height and velocity EMAs, so squat-and-walk is learned on top of a
       working squat.

    Every stage lasts at least ``min_stage_iters`` (lets the 0.99 EMAs see the new
    stage). State lives on the command term, as for :func:`squat_depth_curriculum`.
    """
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "squat_stage"):
        return None
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    n_h = len(height_stages)
    n_total = n_h + len(vel_scales)

    def apply(stage: int) -> None:
        c = term.cfg
        c.rel_squat_envs = float(rel_squat)
        lo, hi = height_stages[min(stage, n_h - 1)]
        c.squat_height_range = (float(lo), float(hi))
        s = 0.0 if stage < n_h else float(vel_scales[stage - n_h])
        (x0, x1), (y0, y1), (z0, z1) = vel_full
        c.ranges.lin_vel_x = (s * x0, s * x1)
        c.ranges.lin_vel_y = (s * y0, s * y1)
        c.ranges.ang_vel_z = (s * z0, s * z1)
        c.rel_standing_envs = 1.0 if stage < n_h else float(vel_rel_standing)
        term.squat_stage = stage
        term.hv_stage_since = iteration
        print(f"[squat-curriculum] iter {iteration:.0f}: stage {stage} "
              f"({'height' if stage < n_h else 'velocity'}) H*~U{c.squat_height_range} "
              f"vel_scale={s:.2f} rel_standing={c.rel_standing_envs:.2f} "
              f"(vel_ema={term.vel_err_ema:.3f}, height_ema={term.height_err_ema:.4f})", flush=True)

    if not getattr(term, "hv_init", False):
        term.hv_init = True
        apply(0)
    stage = int(term.squat_stage)
    if stage < n_total - 1 and iteration - term.hv_stage_since >= min_stage_iters:
        ok = term.height_err_ema <= float(height_gate)
        if stage >= n_h:                  # in B: velocity must track too
            ok = ok and term.vel_err_ema <= float(vel_gate)
        if ok:
            apply(stage + 1)
    c = term.cfg
    return {
        "stage": int(term.squat_stage),
        "velocity_phase": float(term.squat_stage >= n_h),
        "h_low": float(c.squat_height_range[0]),
        "vel_x_max": float(c.ranges.lin_vel_x[1]),
        "vel_ema": float(term.vel_err_ema),
        "height_ema": float(term.height_err_ema),
        "iteration": float(iteration),
    }
