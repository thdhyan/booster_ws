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
