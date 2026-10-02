# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Physics backend selection for K1 tasks.

Selection via env var ``K1_PHYSICS``:

* ``physx`` (default) — leave ``sim.physics`` unset; the launcher's automatic
  selector resolves it to PhysX (the proven local path).
* ``newton`` (opt-in) — ``sim.physics = NewtonCfg()`` (isaaclab_newton).

**Decision 2026-09-22 (user-approved): PhysX for ALL local runs.** Measured on
the RTX 4060 @256 envs: PhysX 1.6–2.1 s/iter vs Newton 4.8–5.0 s/iter (~2.5–3×
slower), and PhysX→Newton checkpoint transfer explodes to NaN within one
iteration (fresh Newton policies are stable). Newton remains available via
``K1_PHYSICS=newton`` for probes or stronger boxes.

Called from each env cfg's ``__post_init__`` as ``apply_physics_backend(self)``.
``isaaclab_newton`` imports are pxr-free (verified), so this is safe before
SimulationApp starts.

Newton workaround: ``ContactSensorCfg.filter_prim_paths_expr=["/World/ground"]``
crashes Newton sensor init ("No bodies matched the counterpart pattern(s)" —
the ground plane is not a body label in the Newton model).  The filter is
semantically inert for our tasks: every consumer reads **net** forces
(``net_normal_forces_w_history`` — all contacts in PhysX too) and air/contact
timers; ``filtered_forces``/``force_matrix``/``contact_pos`` are never read
(verified in k1_velocity).  So we clear filter exprs under Newton only.
"""
from __future__ import annotations

import os

from isaaclab.sim import SimulationCfg


#: Aggregate pairs to budget per environment. The observed demand at 8192 envs is
#: ~256 pairs/env (2 099 817 / 8192); 512 is 2x that, so the buffer does not sit
#: one contact short of the cap.
PAIRS_PER_ENV = 512
#: Floor regardless of env count, so small runs are not raised to a silly size.
MIN_AGGREGATE_PAIRS = 2**22  # 4 194 304


def ensure_physx_gpu_capacity(env_cfg) -> int:
    """Raise the GPU broadphase pair buffer so contacts are not silently dropped.

    PhysX allocates its GPU aggregate-pair buffer once, from
    ``gpu_total_aggregate_pairs_capacity``, which defaults to ``2**21``.  When
    demand exceeds it the broadphase does NOT fail loudly -- it logs, once per
    physics step::

        PhysX error: The application needs to increase
        PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 2099817
        , otherwise, the simulation will miss interactions

    and then **drops contacts**, so feet pass through the ground and the policy
    trains against a wrong contact model.

    This is not hypothetical and it was badly misdiagnosed.  At 8192 envs with the
    corrected ankle gains the K1 generates 2 099 817 pairs against the 2 097 152
    default -- short by 2 665, i.e. 0.13% -- and the resulting error flood cost
    roughly 6x wall-clock (3.1-5.0 s/iter with 1 logged error, versus 20-32 s/iter
    with 1392).  The overflow tracks *env count and contact activity*, not the
    gains: a 2048-env A/B of the same two gain settings showed a 0.3% difference
    because 2048 envs never reach the cap.  That is why the gains looked innocent
    at 2048 and guilty at 8192.

    Hence sizing from ``num_envs`` rather than hardcoding a constant: a run at
    16 384 envs would otherwise silently start dropping contacts again.

    Returns the capacity now configured, or 0 when not applicable (Newton).
    """
    physx = getattr(env_cfg.sim, "physx", None)
    if physx is None:
        return 0
    if not hasattr(physx, "gpu_total_aggregate_pairs_capacity"):
        return 0
    num_envs = int(getattr(env_cfg.scene, "num_envs", 0) or 0)
    want = max(MIN_AGGREGATE_PAIRS, num_envs * PAIRS_PER_ENV)
    if physx.gpu_total_aggregate_pairs_capacity < want:
        physx.gpu_total_aggregate_pairs_capacity = want
    return int(physx.gpu_total_aggregate_pairs_capacity)


def apply_physics_backend(env_cfg) -> str:
    """Select the physics backend on ``env_cfg`` from ``$K1_PHYSICS``.

    Sets ``env_cfg.sim.physics`` and, under Newton, clears contact-sensor
    per-partner filters (see module docstring).  Returns the backend name.
    """
    backend = os.environ.get("K1_PHYSICS", "physx").lower()
    if backend == "newton":
        from isaaclab_newton.physics import NewtonCfg

        env_cfg.sim.physics = NewtonCfg()
        _clear_contact_filters(env_cfg.scene)
    elif backend != "physx":
        raise ValueError(f"K1_PHYSICS must be 'newton' or 'physx', got {backend!r}")
    return backend


def _clear_contact_filters(scene_cfg) -> None:
    """Clear ``filter_prim_paths_expr`` on every sensor cfg in the scene cfg.

    Duck-typed walk over the scene configclass fields so each task family
    (velocity/basic/kick) is covered without per-cfg edits.

    Fields are read from the instance ``__dict__`` directly: plain ``getattr``
    on e.g. the scene's ``class_type: ResolvableString`` triggers its lazy
    ``__getattr__`` template resolution, which imports 43 pxr modules
    pre-SimulationApp (the Kit-poisoning failure mode from the G2 video bug).
    """
    for value in vars(scene_cfg).values():
        if isinstance(value, list):
            continue
        fields = getattr(value, "__dict__", None)
        if not isinstance(fields, dict):
            continue
        if fields.get("filter_prim_paths_expr"):
            value.filter_prim_paths_expr = []
