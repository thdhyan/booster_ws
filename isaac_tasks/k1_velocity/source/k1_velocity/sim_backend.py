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
