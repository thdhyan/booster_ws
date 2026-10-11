"""Tiered carry rewards: pure-tensor pieces (no Isaac imports, unit-tested directly).

The first carry run fell in ~15 % of episodes: every reward it could earn needed
walking, so standing still with the box earned nothing and falling while walking
cost little more than standing. The tiers fix the ordering:

1. ``hold_quality`` in [0, 1]: upright trunk AND the box at its nominal spot in
   front of the trunk. Paid on its own every step (``stand_hold``), and doubled
   while the command asks for zero velocity and the robot is still (``stand_still``).
2. Walking/goal rewards are multiplied by ``hold_quality``, so they only pay while
   tier 1 holds: walking with the box slipping or the trunk tipping earns nothing.
3. ``goal_reached`` scales with how far the goal was (``reach_scale``), so
   stand-only goals (radius 0) cannot farm the success bonus.
4. ``RadiusSchedule`` only widens the goal radius while the fall rate is low.
"""
from __future__ import annotations

import torch


def hold_quality_from(box_b: torch.Tensor, nominal_b: torch.Tensor, gravity_b: torch.Tensor,
                      pos_std: float, tilt_std: float) -> torch.Tensor:
    """(n,) exp(-|box - nominal|^2 / pos_std^2) * exp(-|g_xy|^2 / tilt_std^2).

    ``box_b``/``nominal_b``: box centre in the root frame (n,3); ``gravity_b``: unit
    projected gravity in the root frame (n,3); its xy part is 0 when upright.
    """
    pos_err = (box_b - nominal_b).square().sum(-1)
    tilt = gravity_b[:, :2].square().sum(-1)
    return torch.exp(-pos_err / pos_std**2) * torch.exp(-tilt / tilt_std**2)


def stand_still_kernel(cmd_xy_wz: torch.Tensor, lin_vel_xy: torch.Tensor, ang_vel_z: torch.Tensor,
                       cmd_eps: float, sigma: float) -> torch.Tensor:
    """(n,) exp(-(|v_xy|^2 + w_z^2) / sigma^2) where the command is ~zero, else 0."""
    idle = (cmd_xy_wz.norm(dim=-1) < cmd_eps).to(lin_vel_xy.dtype)
    motion = lin_vel_xy.square().sum(-1) + ang_vel_z.square()
    return idle * torch.exp(-motion / sigma**2)


def reach_scale(goal_r: torch.Tensor, full_r: float) -> torch.Tensor:
    """(n,) goal-reached bonus scale: 0 for a stand goal, 1 from ``full_r`` metres out."""
    return (goal_r / full_r).clamp(0.0, 1.0)


class RadiusSchedule:
    """Grow the max goal radius by ``step`` whenever a window of resets falls rarely.

    A window closes after ``window`` resets; if falls / resets < ``fall_threshold``
    the radius grows (capped at ``target``). Never shrinks: a regression shows up
    as a rising fall_rate in the logs, not as a silent rollback.
    """

    def __init__(self, start: float, step: float, target: float, fall_threshold: float, window: int):
        self.start = float(start)
        self.max_r = float(start)
        self.step = float(step)
        self.target = float(target)
        self.fall_threshold = float(fall_threshold)
        self.window = int(window)
        self.resets = 0
        self.falls = 0
        self.last_rate = -1.0   # no window closed yet

    def update(self, n_resets: int, n_falls: int) -> bool:
        """Count resets; return True when this call widened the radius."""
        self.resets += int(n_resets)
        self.falls += int(n_falls)
        if self.resets < self.window:
            return False
        self.last_rate = self.falls / self.resets
        self.resets = self.falls = 0
        if self.last_rate < self.fall_threshold and self.max_r < self.target:
            self.max_r = min(self.max_r + self.step, self.target)
            return True
        return False
