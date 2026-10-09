"""Phase-clock observation for the K1 velocity task (gait lever 3 of 3).

Measured failure (``scripts/gait_gate.py`` on ``p2_gaitshuffle_2999.npz``, 8/8
gates failed): cadence 8.25 steps/s against a wanted 1.2-4.0, stride 0.03 m,
jerk 0.10 -- the policy shuffled instead of stepping.

The cause is that nothing in the observation tells it *when* to step. P2's
policy group is pure proprioception (48 dims) plus the velocity command, so a
step is only "something that happens", never "something due now". Raising the
ankle gains (lever 1) gives the foot the authority to hold a rigid lever;
domain-randomising friction (lever 2) stops the sole from sliding; this clock
gives the policy a time base to swing that lever on.

One full cycle is one gait cycle -- two steps, left then right -- so a
``frequency_hz`` of 1.0 asks for 2.0 steps/s, the middle of the gate's
[1.2, 4.0] band and an ordinary walking cadence.

The output is ``stack([sin(2*pi*phase), cos(2*pi*phase)], -1)``: a continuous,
wrap-free encoding of phase. A raw ``phase`` in [0, 1) jumps 1.0 -> 0.0 at the
wrap and looks like two different inputs to an MLP; the sin/cos pair is the
same point on the circle either side of it, so the network can interpolate
through the seam instead of memorising both endpoints.

Observation terms are stateless in Isaac Lab -- the manager calls
``term_cfg.func(env)`` fresh every step -- so the phase itself lives in a
module-level dict keyed by ``id(env)``. The manager calls each term once per
control step for the group that contains it (twice: the policy and the teacher
both carry this term), and both groups must observe the *same* phase. That is
why the state carries ``prev_len``: advancing on ``episode_length_buf``
changing, rather than on being called, makes exactly one of the calls advance
the clock and the others no-ops. ``get_phase()`` exposes the raw phase to
consumers outside this module -- the phase-synced swing reward needs the same
reference the policy observes, and the idempotent read guarantees they agree.
"""
from __future__ import annotations

import math

import torch

# One gait cycle == two steps, so 1.0 Hz == 2.0 steps/s (see module docstring).
PHASE_FREQUENCY_HZ = 1.0

# ---------------------------------------------------------------------------
# Command-coupled gait frequency (2026-10-01)
# ---------------------------------------------------------------------------
# The fixed clock was the blocker for a phase-locked *stride* reward. A fixed
# 1.001 Hz reference is demanded identically at 0.1 m/s and 0.5 m/s, so a term
# that rewards placing the foot where the clock says makes the robot step on
# our schedule rather than the command's, and velocity tracking wins because it
# pays more (+1.28/step against ~0.7 of gait penalty).
#
# That is visible in the measured two-mode split on the 5000-iteration run: the
# chattering envs were moving ~0.44 m/s effective (11.5 steps/s x 4 cm strides),
# i.e. the top of the command range, while the under-steppers sat near zero.
# Chatter was the policy's solution to "go fast".
#
# AGILE avoids this by exposing BOTH ``gait_process`` and ``gait_frequency`` from
# its command term (``UniformVelocityGaitBaseHeightCommand``) so gait-cycle
# rewards enforce timing at the speed actually being asked for. We had the phase
# without the frequency coupling.
#
# Frequency now follows commanded speed: cadence scales with the command, and is
# clamped to a band that stays inside the gate's [1.2, 4.0] steps/s. A near-zero
# command gives MIN_HZ, not zero -- a stopped robot still needs a reference to
# stand against, and frequency 0 would freeze the phase and make the stride term
# uninformative exactly when the policy is least confident.
HZ_PER_MPS = 2.0            # 0.5 m/s -> 1.0 Hz -> 2.0 steps/s, mid-gate-band
MIN_HZ = 0.6                # ~1.2 steps/s: the gate's lower bound
MAX_HZ = 2.0                # 4.0 steps/s: the gate's upper bound
COMMAND_SMOOTHING = 0.15    # EMA coefficient on the commanded magnitude

# Per-env clock state: {"phase": (num_envs,) float in [0,1), "prev_len": long}.
# Keyed by id(env) because a process holds more than one env during config
# probing, and each needs its own phase. Shape/device are validated on read so
# a rebuilt env (different num_envs) re-seeds rather than indexing stale
# storage; an env that dies and is replaced at the same id self-heals too,
# because its inherited prev_len is larger than the fresh env's and therefore
# reads as a reset.
_CLOCK_STATE: dict[int, dict] = {}


def command_magnitude(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Per-env magnitude of the planar velocity command, or None if unavailable.

    Public because it is the physical scale a gait reference needs: the stride a
    command implies is ``v / steps_per_second``, so any reward that asks "where
    should this foot be" has to know ``v`` in m/s. Deriving that from the clock
    frequency alone is what produced the inverted amplitude in
    ``phase_locked_stride`` (see the note there).

    Read from the command *term* rather than the manager's tensor because the
    term is what carries the ranges the frequency clamp is expressed against.
    """
    mgr = getattr(env, "command_manager", None)
    if mgr is None:
        return None
    getter = getattr(mgr, "get_term", None)
    try:
        term = getter("base_velocity") if getter is not None else mgr._terms["base_velocity"]
        cmd = term.vel_command_b
    except Exception:  # noqa: BLE001 - probing envs may lack commands
        return None
    return torch.linalg.norm(cmd[:, :2], dim=1)


def get_frequency(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Per-env gait frequency in Hz, following the commanded speed.

    EMA-smoothed so a command resample (every 8-12 s) does not step the
    reference, which would ask the policy to change cadence mid-stride.

    Returns a ``(num_envs,)`` float32 tensor in ``[MIN_HZ, MAX_HZ]``. Falls back
    to ``PHASE_FREQUENCY_HZ`` when no command term is reachable (config probing).
    """
    device = env.episode_length_buf.device
    num_envs = env.num_envs
    state = _CLOCK_STATE.get(id(env))
    if state is None or state.get("freq_hz") is None or state["freq_hz"].shape != (num_envs,):
        if state is not None:
            state["freq_hz"] = torch.full(
                (num_envs,), PHASE_FREQUENCY_HZ, device=device
            )
        else:
            return torch.full((num_envs,), PHASE_FREQUENCY_HZ, device=device)

    mag = command_magnitude(env, command_name)
    if mag is None:
        return state["freq_hz"]
    return get_frequency_from_magnitude(env, mag)


def get_frequency_from_magnitude(env, mag: torch.Tensor) -> torch.Tensor:
    """``get_frequency`` with the caller supplying the speed instead of the env.

    The EMA, the ``[MIN_HZ, MAX_HZ]`` clamp and the reset handling live HERE, in one
    place, because they are the whole of the clock's behaviour and a second copy of
    them is how the frequency ends up subtly different between the phase a policy
    trained on and the phase it is served.

    Needed by the box-push task, where the commanded speed is not in a command term
    at all: push drives the frozen base through the action's ``[vx, vy, wz, H*]``
    slice and records it in ``push_state.last_vel_cmd``. Reading
    ``command_magnitude`` there returns None -- ``base_velocity`` is a wrist-target
    term with no ``vel_command_b`` -- so the clock would silently freeze at
    ``PHASE_FREQUENCY_HZ`` and the frozen base would be served a cadence it was never
    trained against.
    """
    device = env.episode_length_buf.device
    num_envs = env.num_envs
    state = _CLOCK_STATE.get(id(env))
    if state is None or state.get("freq_hz") is None or state["freq_hz"].shape != (num_envs,):
        if state is not None:
            state["freq_hz"] = torch.full(
                (num_envs,), PHASE_FREQUENCY_HZ, device=device
            )
        else:
            return torch.full((num_envs,), PHASE_FREQUENCY_HZ, device=device)

    mag = mag.to(device=device, dtype=state["freq_hz"].dtype)
    target = (HZ_PER_MPS * mag).clamp(MIN_HZ, MAX_HZ)
    prev = state["prev_len"]
    cur = env.episode_length_buf
    just_reset = cur < prev
    stepped = (cur > prev).to(state["freq_hz"].dtype)
    # Reset envs jump straight to the new command's frequency; only stepping
    # envs get the EMA, so a resample cannot be smoothed away by a stale value.
    fresh = just_reset.to(state["freq_hz"].dtype)
    blend = torch.clamp(COMMAND_SMOOTHING * stepped + fresh, max=1.0)
    state["freq_hz"].lerp_(target, blend)
    return state["freq_hz"]


def phase_clock_from_magnitude(env, mag: torch.Tensor) -> torch.Tensor:
    """``phase_clock`` driven by a caller-supplied speed. See the above."""
    return phase_clock(env, get_frequency_from_magnitude(env, mag))


def get_phase(env, frequency_hz: float | None = None) -> torch.Tensor:
    """Return the raw phase (``[0, 1)``) for every env, advancing the clock.

    This is the same state ``phase_clock`` encodes; the sin/cos observation and
    any consumer that needs the linear phase (the phase-synced swing reward)
    therefore can never drift apart. The advance-on-``episode_length_buf``
    scheme makes the read idempotent within a control step, so it is safe to
    call from both the observation and the reward manager regardless of which
    runs first.

    Args:
        env: The manager-based env (observation or reward manager's env instance).
        frequency_hz: Gait cycles per second, or None to follow the commanded
            speed (the default since 2026-10-01). A float pins the clock, which
            is what the command-coupling tests do to isolate the arithmetic.

    Returns:
        A ``(num_envs,)`` float32 tensor in ``[0, 1)``.
    """
    device = env.episode_length_buf.device
    num_envs = env.num_envs
    state = _CLOCK_STATE.get(id(env))
    if (
        state is None
        or state["phase"].shape != (num_envs,)
        or state["phase"].device != device
    ):
        # First read for this env (config probing and the real run both land
        # here): seed phase uniformly so the first episode does not start every
        # env mid-stride at the same point.
        state = {
            "phase": torch.rand(num_envs, device=device),
            "prev_len": env.episode_length_buf.clone(),
            "freq_hz": torch.full((num_envs,), PHASE_FREQUENCY_HZ, device=device),
        }
        _CLOCK_STATE[id(env)] = state

    if frequency_hz is None:
        hz = get_frequency(env)
    elif isinstance(frequency_hz, torch.Tensor):
        # A per-env tensor frequency, for callers that already resolved it (the push
        # task derives it from the action's velocity slice rather than a command term).
        # Kept here rather than in a second phase implementation so the advance,
        # the reset reseed and the in-place update stay in one place.
        hz = frequency_hz.to(device=device, dtype=torch.float32).reshape(num_envs)
    else:
        hz = torch.full((num_envs,), float(frequency_hz), device=device)

    cur = env.episode_length_buf
    # "<" means this env was reset this step (episode_length_buf drops to 0),
    # ">" a normal step, "=" a second caller reading the same step (the policy
    # and teacher groups both carry the observation, and the reward reads it too).
    just_reset = cur < state["prev_len"]
    if bool(just_reset.any()):
        state["phase"][just_reset] = torch.rand(int(just_reset.sum()), device=device)
    stepped = (cur > state["prev_len"]).to(state["phase"].dtype)
    # In-place: called every step for several consumers, must not allocate or grow.
    # Phase advances by the *per-env* frequency, so a fast command really does
    # demand a faster cadence.
    state["phase"].add_(stepped * hz * env.step_dt).remainder_(1.0)
    state["prev_len"].copy_(cur)
    return state["phase"]


def phase_clock(env, frequency_hz: float | None = None) -> torch.Tensor:
    """Return ``[sin(2*pi*phase), cos(2*pi*phase)]`` for every env.

    The phase advances ``frequency_hz * env.step_dt`` per control step, where the
    frequency follows the commanded speed by default, and is resampled uniformly
    for any env whose episode reset since the last read, so a policy cannot lean
    on a phase the previous episode happened to leave behind.

    Args:
        env: The manager-based env (policy or teacher group's env instance).
        frequency_hz: Gait cycles per second; a float pins the clock, a
            ``(num_envs,)`` tensor supplies a per-env frequency, or None follows
            the commanded speed.

    Returns:
        A ``(num_envs, 2)`` float32 tensor, concatenated into the group.
    """
    angle = get_phase(env, frequency_hz) * (2.0 * math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def reset_clock_state(env) -> None:
    """Drop cached clock state for this env (tests, and re-probing a new env)."""
    _CLOCK_STATE.pop(id(env), None)
