"""Squat residual two-stage curriculum: height at zero velocity first, then velocity.

squat_mdp has no Isaac imports, so it is loaded straight from its source file and driven
with a fake env/command term.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

VEL = Path(__file__).resolve().parents[1] / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"


def _mod():
    spec = importlib.util.spec_from_file_location("squat_mdp", VEL / "squat_mdp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Env:
    def __init__(self, term):
        self.steps = 0
        self.cfg = SimpleNamespace(decimation=1)
        self.sim = SimpleNamespace(get_physics_step_count=lambda: self.steps)
        self.command_manager = SimpleNamespace(get_term=lambda name: term)

    def at_iter(self, it):
        self.steps = it * 24


def _term():
    ranges = SimpleNamespace(lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8))
    cfg = SimpleNamespace(rel_squat_envs=0.0, squat_height_range=(0.50, 0.55), ranges=ranges,
                          rel_standing_envs=0.02)
    return SimpleNamespace(cfg=cfg, squat_stage=0, vel_err_ema=1.0, height_err_ema=1.0)


def test_stage_a_zero_velocity_then_deeper_on_height_gate():
    m, term = _mod(), _term()
    env = _Env(term)
    f = m.squat_height_then_velocity_curriculum
    out = f(env, None)
    assert term.cfg.ranges.lin_vel_x == (0.0, 0.0) and term.cfg.rel_standing_envs == 1.0
    assert out["stage"] == 0 and term.cfg.rel_squat_envs > 0
    env.at_iter(500)
    f(env, None)                      # height not held yet -> stays
    assert term.squat_stage == 0
    term.height_err_ema = 0.01
    f(env, None)
    assert term.squat_stage == 1 and term.cfg.squat_height_range[0] < 0.50
    env.at_iter(550)                  # min_stage_iters not elapsed
    f(env, None)
    assert term.squat_stage == 1


def test_stage_b_velocity_needs_both_gates_and_reaches_full_range():
    m, term = _mod(), _term()
    env = _Env(term)
    f = m.squat_height_then_velocity_curriculum
    f(env, None)
    term.height_err_ema = 0.01
    it = 0
    for _ in range(3):                # through the 4 height stages
        it += 200
        env.at_iter(it)
        f(env, None)
    assert term.squat_stage == 3 and term.cfg.ranges.lin_vel_x == (0.0, 0.0)
    it += 200
    env.at_iter(it)
    f(env, None)                      # last height stage held -> first velocity stage
    assert term.squat_stage == 4 and term.cfg.ranges.lin_vel_x[1] == pytest.approx(0.125)
    assert term.cfg.rel_standing_envs < 1.0
    it += 200
    env.at_iter(it)
    f(env, None)                      # vel_err_ema 1.0 blocks
    assert term.squat_stage == 4
    term.vel_err_ema = 0.1
    for _ in range(5):
        it += 200
        env.at_iter(it)
        f(env, None)
    assert term.squat_stage == 7
    assert term.cfg.ranges.lin_vel_x == (-0.5, 0.5) and term.cfg.ranges.ang_vel_z == (-0.8, 0.8)
    assert term.cfg.squat_height_range == (0.40, 0.55)
