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

# Per-env clock state: {"phase": (num_envs,) float in [0,1), "prev_len": long}.
# Keyed by id(env) because a process holds more than one env during config
# probing, and each needs its own phase. Shape/device are validated on read so
# a rebuilt env (different num_envs) re-seeds rather than indexing stale
# storage; an env that dies and is replaced at the same id self-heals too,
# because its inherited prev_len is larger than the fresh env's and therefore
# reads as a reset.
_CLOCK_STATE: dict[int, dict] = {}


def get_phase(env, frequency_hz: float = PHASE_FREQUENCY_HZ) -> torch.Tensor:
    """Return the raw phase (``[0, 1)``) for every env, advancing the clock.

    This is the same state ``phase_clock`` encodes; the sin/cos observation and
    any consumer that needs the linear phase (the phase-synced swing reward)
    therefore can never drift apart. The advance-on-``episode_length_buf``
    scheme makes the read idempotent within a control step, so it is safe to
    call from both the observation and the reward manager regardless of which
    runs first.

    Args:
        env: The manager-based env (observation or reward manager's env instance).
        frequency_hz: Gait cycles per second. 1.0 Hz = 2 steps/s.

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
        }
        _CLOCK_STATE[id(env)] = state

    cur = env.episode_length_buf
    # "<" means this env was reset this step (episode_length_buf drops to 0),
    # ">" a normal step, "=" a second caller reading the same step (the policy
    # and teacher groups both carry the observation, and the reward reads it too).
    just_reset = cur < state["prev_len"]
    if bool(just_reset.any()):
        state["phase"][just_reset] = torch.rand(int(just_reset.sum()), device=device)
    stepped = (cur > state["prev_len"]).to(state["phase"].dtype)
    # In-place: called every step for several consumers, must not allocate or grow.
    state["phase"].add_(stepped * (frequency_hz * env.step_dt)).remainder_(1.0)
    state["prev_len"].copy_(cur)
    return state["phase"]


def phase_clock(env, frequency_hz: float = PHASE_FREQUENCY_HZ) -> torch.Tensor:
    """Return ``[sin(2*pi*phase), cos(2*pi*phase)]`` for every env.

    The phase advances ``frequency_hz * env.step_dt`` per control step and is
    resampled uniformly for any env whose episode reset since the last read, so
    a policy cannot lean on a phase the previous episode happened to leave
    behind.

    Args:
        env: The manager-based env (policy or teacher group's env instance).
        frequency_hz: Gait cycles per second. 1.0 Hz = 2 steps/s.

    Returns:
        A ``(num_envs, 2)`` float32 tensor, concatenated into the group.
    """
    angle = get_phase(env, frequency_hz) * (2.0 * math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1)
