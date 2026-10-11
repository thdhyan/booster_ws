"""Tiered carry rewards: the pure-tensor pieces in carry_tiers, plus cfg wiring.

carry_tiers has no Isaac imports, so it is loaded straight from its source file.
The env cfg needs Isaac, so its wiring is checked on the source text.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CARRY = Path(__file__).resolve().parents[1] / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/carry"


def _tiers():
    spec = importlib.util.spec_from_file_location("carry_tiers", CARRY / "carry_tiers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _tiers()
UP = torch.tensor([[0.0, 0.0, -1.0]])
NOMINAL = torch.tensor([[0.25, 0.0, 0.05]])


def test_hold_quality_is_one_when_upright_and_box_on_spot():
    q = T.hold_quality_from(NOMINAL.clone(), NOMINAL, UP, 0.08, 0.3)
    assert q.item() == pytest.approx(1.0)


def test_hold_quality_drops_with_box_slip_and_with_tilt():
    slipped = T.hold_quality_from(NOMINAL + torch.tensor([[0.08, 0.0, 0.0]]), NOMINAL, UP, 0.08, 0.3)
    assert slipped.item() == pytest.approx(math.exp(-1.0))
    tilt = 0.3
    g = torch.tensor([[math.sin(tilt), 0.0, -math.cos(tilt)]])
    tipped = T.hold_quality_from(NOMINAL.clone(), NOMINAL, g, 0.08, 0.3)
    assert tipped.item() == pytest.approx(math.exp(-(math.sin(tilt) / 0.3) ** 2))
    # a fallen robot (gravity in the trunk's xy plane) earns ~nothing
    fallen = T.hold_quality_from(NOMINAL.clone(), NOMINAL, torch.tensor([[1.0, 0.0, 0.0]]), 0.08, 0.3)
    assert fallen.item() < 1e-4


def test_stand_still_pays_only_for_zero_command_and_no_motion():
    zero_cmd = torch.zeros(1, 3)
    still = T.stand_still_kernel(zero_cmd, torch.zeros(1, 2), torch.zeros(1), 0.05, 0.2)
    assert still.item() == pytest.approx(1.0)
    drifting = T.stand_still_kernel(zero_cmd, torch.tensor([[0.2, 0.0]]), torch.zeros(1), 0.05, 0.2)
    assert drifting.item() == pytest.approx(math.exp(-1.0))
    walking_cmd = torch.tensor([[0.4, 0.0, 0.0]])
    asked_to_walk = T.stand_still_kernel(walking_cmd, torch.zeros(1, 2), torch.zeros(1), 0.05, 0.2)
    assert asked_to_walk.item() == 0.0


def test_reach_scale_zero_for_stand_goals_and_saturates():
    s = T.reach_scale(torch.tensor([0.0, 0.5, 1.0, 3.0]), 1.0)
    assert s.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_radius_schedule_waits_for_a_full_window():
    sched = T.RadiusSchedule(start=1.0, step=0.25, target=4.0, fall_threshold=0.08, window=100)
    assert not sched.update(60, 0)
    assert sched.max_r == 1.0 and sched.last_rate == -1.0
    assert sched.update(40, 2)           # 2 % falls over the window
    assert sched.max_r == 1.25 and sched.last_rate == pytest.approx(0.02)


def test_radius_schedule_holds_while_falls_are_frequent_and_caps_at_target():
    sched = T.RadiusSchedule(start=3.9, step=0.25, target=4.0, fall_threshold=0.08, window=10)
    assert not sched.update(10, 5)       # 50 % falls
    assert sched.max_r == 3.9
    assert sched.update(10, 0)
    assert sched.max_r == 4.0
    assert not sched.update(10, 0)       # at target: never exceeds it
    assert sched.max_r == 4.0


def test_cfg_gates_goal_terms_and_keeps_tier_one_ungated():
    src = (CARRY / "carry_env_cfg.py").read_text()
    for name in ("goal_progress", "goal_reached"):
        assert f'"{name}"' in src.split("GATED_REWARDS = ")[1].split("\n")[0]
    gated_line = src.split("GATED_REWARDS = ")[1].split("\n")[0]
    assert "stand_hold" not in gated_line
    assert "stand_hold = RewTerm" in src
    assert "goal_radius_curriculum" in src
