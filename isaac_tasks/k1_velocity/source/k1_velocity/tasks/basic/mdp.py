# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""P1 BASIC custom MDP terms (PLAN §3.2 push, §4 P1 rewards/obs).

Standard terms (flat_orientation_l2, joint_deviation_l1, feet_slide, ...) come from
``isaaclab_tasks.core.velocity.mdp`` and are referenced there directly by the env cfg;
only K1/plan-specific terms live here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    # NOTE: keep these under TYPE_CHECKING — importing the runtime classes eagerly
    # pulls pxr into sys.modules before SimulationApp starts, which breaks Kit's
    # extension loading at launch (observed as `omni.usd has no attribute
    # 'get_context'` on --video runs). Annotations only; never needed at runtime.
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


# ---------------------------------------------------------------------------
# Stand-still rewards (P1 R2 / R3)
# ---------------------------------------------------------------------------
def still_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """||v_xy||^2 — horizontal drift penalty (weight -1.0)."""
    robot: Articulation = env.scene["robot"]
    return torch.sum(robot.data.root_vel_w.torch[:, :2] ** 2, dim=-1)


def still_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """||omega||^2 — rotation penalty (weight -0.5)."""
    robot: Articulation = env.scene["robot"]
    return torch.sum(robot.data.root_vel_w.torch[:, 3:6] ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Teacher observation (P1 teacher input: +4 dims)
# ---------------------------------------------------------------------------
def foot_contact_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-foot (contact flag, slip speed) -> (num_envs, 4).

    contact: net normal force > 1 N on that foot;  slip: foot horizontal speed while
    in contact (0 while airborne). Feet order follows ``body_ids`` resolution.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # EA names the PhysX `get_net_contact_forces` history `net_normal_forces_w_history`;
    # the isaac-lab image renamed it `net_forces_w_history` (same quantity, value-identical).
    hist = getattr(sensor.data, "net_normal_forces_w_history", None)
    if hist is None:
        hist = sensor.data.net_forces_w_history
    forces = hist.torch[:, -1, sensor_cfg.body_ids, :].norm(dim=-1)
    contact = (forces > 1.0).float()
    asset: Articulation = env.scene[asset_cfg.name]
    slip = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.cat([contact, slip * contact], dim=-1)


# ---------------------------------------------------------------------------
# Auto-stability push (PLAN §3.2 — P1/P2 required)
# ---------------------------------------------------------------------------
class random_body_push(ManagerTermBase):
    """Localized force shove: horizontal 20-80 N for 0.05-0.15 s on a random body
    part, every 3-8 s, active in ~60 % of envs.

    Registered as ``mode="interval"`` with ``interval_range_s=(0.01, 0.01)``: the
    manager interval is only the *tick* (100 Hz); push start/duration scheduling is
    tracked inside this term because the manager cannot time a *persistent* force.
    The force itself rides the permanent wrench buffer (applied every physics step
    until cleared) — set at push start via ``permanent_wrench_composer`` reset+add,
    cleared at push end with ``reset``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.tick = float(p.get("tick", 0.01))
        self.force_lo, self.force_hi = p.get("force_range", (20.0, 80.0))
        self.dur_lo, self.dur_hi = p.get("duration_range", (0.05, 0.15))
        self.iv_lo, self.iv_hi = p.get("interval_range", (3.0, 8.0))
        self.active_frac = float(p.get("active_env_fraction", 0.6))

        robot: Articulation = env.scene["robot"]
        self.body_ids, _ = robot.find_bodies(list(p["body_names"]))

        n = env.scene.num_envs
        dev = env.device
        self.enabled = torch.rand(n, device=dev) < self.active_frac
        self.wait = self._sample_wait(n, dev)  # countdown to next push start
        self.push_left = torch.zeros(n, device=dev)  # > 0 while force is active

    def _sample_wait(self, n: int, device: torch.device | str) -> torch.Tensor:
        return self.iv_lo + torch.rand(n, device=device) * (self.iv_hi - self.iv_lo)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        body_names: list[str] | tuple = (),
        force_range: tuple[float, float] = (20.0, 80.0),
        duration_range: tuple[float, float] = (0.05, 0.15),
        interval_range: tuple[float, float] = (3.0, 8.0),
        active_env_fraction: float = 0.6,
    ):
        # NOTE: the manager invokes class terms as func(env, env_ids, **cfg.params) and
        # statically checks the signature against cfg.params (manager_base._resolve_common_term_cfg),
        # so every cfg param must appear as a named arg with a default here (built-in class-event
        # pattern). Values are already read from cfg.params in __init__ — the runtime copies are
        # intentionally ignored.
        robot: Articulation = env.scene["robot"]
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        ids = env_ids.long()
        dt = self.tick

        # 1) finish active pushes: clear the wrench for envs whose duration expired
        active = self.push_left[ids] > 0
        self.push_left[ids] = (self.push_left[ids] - dt).clamp(min=0.0)
        finished = ids[active & (self.push_left[ids] == 0)]
        if finished.numel() > 0:
            robot.permanent_wrench_composer.reset(env_ids=finished)

        # 2) idle envs: count down to their next push start
        idle = ids[self.push_left[ids] == 0]
        if idle.numel() == 0:
            return
        self.wait[idle] -= dt
        overdue = idle[self.wait[idle] <= 0.0]
        if overdue.numel() == 0:
            return
        # reschedule every overdue env (disabled envs just roll the dice again)
        self.wait[overdue] = self._sample_wait(overdue.numel(), env.device)
        start = overdue[self.enabled[overdue]]
        if start.numel() > 0:
            self._start_push(robot, start, env.device)

    def _start_push(self, robot: Articulation, ids: torch.Tensor, device) -> None:
        k = ids.numel()
        # one body per env, uniform over the configured candidates
        sel_body = torch.randint(0, len(self.body_ids), (k,), device=device)
        # horizontal random direction (±180°), magnitude force_range
        mag = self.force_lo + torch.rand(k, device=device) * (self.force_hi - self.force_lo)
        az = torch.rand(k, device=device) * 2.0 * torch.pi
        forces = torch.zeros(k, 1, 3, device=device)
        forces[:, 0, 0] = mag * torch.cos(az)
        forces[:, 0, 1] = mag * torch.sin(az)
        torques = torch.zeros_like(forces)
        dur = self.dur_lo + torch.rand(k, device=device) * (self.dur_hi - self.dur_lo)

        # clear any stale wrench for the starting envs, then add this push
        # (exactly one body per env -> each env lands in a single group below)
        robot.permanent_wrench_composer.reset(env_ids=ids)
        for i, body_id in enumerate(self.body_ids):
            sel = torch.nonzero(sel_body == i).squeeze(-1)
            if sel.numel() == 0:
                continue
            robot.permanent_wrench_composer.add_forces_and_torques_index(
                forces=forces[sel],
                torques=torques[sel],
                body_ids=[body_id],
                env_ids=ids[sel],
            )
        self.push_left[ids] = dur
