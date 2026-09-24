# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Force-variant (P1f / P2f) custom MDP terms.

The ``*-F-v0`` task family adds Isaac Lab's built-in
``envs.mdp.apply_external_force_torque`` randomization (random 3-D force + torque
shoves riding the Trunk's permanent-wrench buffer). This module exposes that
wrench back to the **teacher** as a privileged observation so *the teacher knows
the shove while the deployable student does not*:

* ``teacher`` group gains :func:`applied_shove_wrench` (6-dim) — privileged,
  exempt in ``tests/test_deployability.py``.
* ``policy`` / ``student`` groups stay blind (proprioception only) — the real
  robot has no external-wrench sensor; the student must infer the disturbance
  from history, with the teacher's anticipation transferred via distillation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    # Keep runtime imports free of pxr: importing assets/envs eagerly at module
    # load breaks Kit extension loading when the cfg module is imported before
    # SimulationApp starts (same pattern as tasks/basic/mdp.py).
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def applied_shove_wrench(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current composed force+torque on the listed bodies -> ``(num_envs, 6)``.

    Reads the articulation's permanent-wrench composer *output* buffers — the
    exact buffer Isaac Lab's ``apply_external_force_torque`` event and the custom
    ``random_body_push`` impulse shove both write into. World-frame input
    wrenches are composed into the body frame, so the result is the wrench
    (``[force_xyz, torque_xyz]``) actually applied to those bodies this step.

    Zeros while no shove is active (composer buffers are zero-initialised and
    cleared by ``composer.reset()``).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    body_sel: slice | list = slice(None) if ids is None or ids == slice(None) else ids
    composer = asset.permanent_wrench_composer
    force = composer.out_force_b.torch[:, body_sel, :].sum(dim=1)
    torque = composer.out_torque_b.torch[:, body_sel, :].sum(dim=1)
    return torch.cat([force, torque], dim=-1)
