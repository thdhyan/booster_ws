"""Behaviour tests for the velocity-range curriculum.

Pure Python: the module under test has no Isaac imports precisely so this logic
can be checked on a laptop with no GPU. Run without any simulator.

Each test was confirmed to fail against a broken version of the logic, because a
curriculum that silently does nothing is the exact failure mode being guarded
against.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_curriculum.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("velocity_curriculum", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["velocity_curriculum"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# --- fakes -------------------------------------------------------------------
class FakeTerm:
    """Stands in for Isaac Lab's UniformVelocityCommand."""

    def __init__(self, lin=0.5, ang=1.0):
        self.lin_vel_x_range = (-lin, lin)
        self.lin_vel_y_range = (-lin, lin)
        self.ang_vel_z_range = (-ang, ang)

    @property
    def lin(self):
        return self.lin_vel_x_range[1]

    @property
    def ang(self):
        return self.ang_vel_z_range[1]


class FakeValue:
    def __init__(self, v):
        self._v = v

    def mean(self):
        return self._v


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


class FakeEnv:
    def __init__(self, tracking=0.9, term=None):
        self.term = term or FakeTerm()
        self.logger = FakeLogger()
        self._tracking = tracking
        self.tracking_calls = 0

        outer = self

        class CM:
            def get_command(self, name):
                assert name == "base_velocity"
                return outer.term

        class RM:
            active_terms = ["track_lin_vel_xy_exp", "other"]

            def get_term(self, name):
                outer.tracking_calls += 1
                return FakeValue(outer._tracking)

        self.command_manager = CM()
        self.reward_manager = RM()


def make(env=None, **kw):
    kw.setdefault("interval_steps", 1)
    kw.setdefault("patience", 3)
    return mod.VelocityRangeCurriculum(env or FakeEnv(), **kw)


def prime(cur, env):
    """Spend the first call, which discovers the range attribute and installs it."""
    cur(env, None)


def run(cur, env, intervals=1):
    """Run `intervals` measurement windows (each interval_steps calls)."""
    for _ in range(intervals * cur.interval_steps):
        cur(env, None)


# --- tests -------------------------------------------------------------------
def test_isaac_import_is_guarded_not_absent():
    """The logic must stay importable without Isaac Lab, so any isaac import has
    to sit inside a try/except ImportError. An unguarded import would make this
    file untestable on a laptop."""
    src = MODULE_PATH.read_text()
    assert "import torch" not in src, "torch is not needed and drags in a GPU dep"
    assert "import omni" not in src

    tree = ast.parse(src)

    # Collect every import that appears inside a try block whose handlers catch
    # ImportError. Those are the guarded ones.
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            isinstance(h.type, ast.Name) and h.type.id == "ImportError"
            for h in node.handlers
        )
        if not catches_import_error:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                guarded.add(id(sub))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = [a.name for a in node.names] + [node.module or ""]
        if not any("isaaclab" in n for n in names):
            continue
        assert id(node) in guarded, (
            f"unguarded Isaac import at line {node.lineno}. The ManagerTermBase "
            "adapter import must sit inside try/except ImportError so the "
            "curriculum logic stays testable without Isaac Sim."
        )

    # The real proof is that this module was imported at the top of the file in an
    # environment with no Isaac Lab on the path: if the guard were broken the whole
    # module would have failed to load before a single test ran.


def test_first_call_installs_range_without_measuring():
    env = FakeEnv()
    cur = make(env)
    cur(env, None)
    # The first call discovers the attribute; it must not also gate on a reward.
    assert env.tracking_calls == 0
    assert env.term.lin == pytest.approx(0.5)


def test_expands_only_after_patience_of_success():
    env = FakeEnv(tracking=0.99)
    cur = make(env, patience=3)
    prime(cur, env)
    run(cur, env, intervals=2)
    assert cur.expansions == 0, "expanded before patience was satisfied"
    assert env.term.lin == pytest.approx(0.5)
    run(cur, env, intervals=1)
    assert cur.expansions == 1
    assert env.term.lin == pytest.approx(0.75)


def test_patience_resets_when_tracking_dips():
    env = FakeEnv(tracking=0.99)
    cur = make(env, patience=3)
    prime(cur, env)
    run(cur, env, intervals=2)
    env._tracking = 0.1
    run(cur, env, intervals=1)
    assert cur.expansions == 0
    env._tracking = 0.99
    run(cur, env, intervals=2)
    assert cur.expansions == 0, "streak was not reset by the dip"
    run(cur, env, intervals=1)
    assert cur.expansions == 1


def test_contracts_when_tracking_collapses():
    env = FakeEnv(tracking=0.99)
    cur = make(env, patience=2)
    prime(cur, env)
    run(cur, env, intervals=2)
    widened = env.term.lin
    assert widened > 0.5
    env._tracking = 0.2
    run(cur, env, intervals=1)
    assert env.term.lin < widened
    assert cur.contractions == 1


def test_never_exceeds_target_ceiling():
    """1.5 m/s is the real-robot-evidenced ceiling; 3 m/s must not be reachable."""
    env = FakeEnv(tracking=1.0)
    cur = make(env, patience=1, target_max_lin_vel=1.5, step_lin_vel=0.25)
    prime(cur, env)
    run(cur, env, intervals=60)
    assert env.term.lin == pytest.approx(1.5)
    assert cur.peak_lin <= 1.5


def test_contracts_back_to_initial_floor():
    env = FakeEnv(tracking=1.0)
    cur = make(env, patience=1)
    prime(cur, env)
    run(cur, env, intervals=8)
    assert env.term.lin > 0.5
    env._tracking = 0.0
    run(cur, env, intervals=60)
    assert env.term.lin == pytest.approx(0.5), "must not contract below the start range"


def test_range_stays_symmetric():
    env = FakeEnv(tracking=1.0)
    cur = make(env, patience=1)
    prime(cur, env)
    run(cur, env, intervals=6)
    lo, hi = env.term.lin_vel_x_range
    assert lo == pytest.approx(-hi), "backward range must widen with forward"
    lo, hi = env.term.lin_vel_y_range
    assert lo == pytest.approx(-hi)


def test_missing_reward_term_never_widens():
    """An absent signal must not read as success."""
    env = FakeEnv()
    cur = make(env, patience=1)
    prime(cur, env)
    env.reward_manager.active_terms = ["something_else"]
    run(cur, env, intervals=50)
    assert cur.expansions == 0
    assert env.term.lin == pytest.approx(0.5)


def test_raises_instead_of_silently_doing_nothing():
    """A renamed Isaac attribute must be loud, not a no-op curriculum."""

    class Renamed(FakeTerm):
        def __init__(self):
            super().__init__()
            del self.lin_vel_x_range
            self.something_else = (-0.5, 0.5)

    env = FakeEnv()
    env.term = Renamed()
    cur = make(env)
    with pytest.raises(AttributeError, match="lin_vel_x"):
        cur(env, None)


def test_interval_gates_measurement_rate():
    env = FakeEnv(tracking=1.0)
    cur = make(env, patience=1, interval_steps=10)
    prime(cur, env)
    run(cur, env, intervals=1)
    assert env.tracking_calls == 1, "did not measure once the interval elapsed"


def test_default_ceiling_matches_real_robot_evidence():
    """Guards the 1.5 m/s default against being silently raised to 3.0."""
    import inspect

    sig = inspect.signature(mod.VelocityRangeCurriculum.__init__)
    assert sig.parameters["target_max_lin_vel"].default == pytest.approx(1.5)


def test_config_points_at_the_managerterm_adapter():
    """Isaac Lab rejects a bare class here, so the config must use the adapter."""
    cfg = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py"
    src = cfg.read_text()
    assert "velocity_curriculum.VelocityRangeCurriculumTerm" in src, (
        "CurrTerm func must be the ManagerTermBase adapter; a bare class raises "
        "TypeError: not of type ManagerTermBase"
    )
    assert "velocity_curriculum.VelocityRangeCurriculum," not in src


# --- cfg plumbing -----------------------------------------------------------
class FakeCfg:
    """Shape Isaac Lab actually hands a term: the values live under .params."""

    def __init__(self, **params):
        self.params = params
        self.func = object()
        self._private = 1


def test_params_arrive_nested_under_cfg_params():
    """Isaac passes the whole ManagerTermBaseCfg, so kwargs are nested.

    Getting this wrong is only visible after a full Isaac boot:
    TypeError: ... got an unexpected keyword argument 'params'
    """
    cfg = FakeCfg(init_lin_vel=0.5, target_max_lin_vel=1.5, patience=5)
    kw = mod._as_kwargs(cfg)
    assert kw == {"init_lin_vel": 0.5, "target_max_lin_vel": 1.5, "patience": 5}
    # And they must be acceptable to the real constructor.
    cur = mod.VelocityRangeCurriculum(FakeEnv(), **kw)
    assert cur.target_max_lin == pytest.approx(1.5)


def test_params_nested_in_a_plain_dict():
    kw = mod._as_kwargs({"params": {"init_lin_vel": 0.75}})
    assert kw == {"init_lin_vel": 0.75}


def test_flat_dict_still_accepted():
    kw = mod._as_kwargs({"init_lin_vel": 0.25, "patience": 2})
    assert kw == {"init_lin_vel": 0.25, "patience": 2}


def test_private_and_callable_cfg_fields_are_dropped():
    """cfg.func and friends must not be forwarded to __init__."""
    cfg = FakeCfg(init_lin_vel=0.5)
    kw = mod._as_kwargs(cfg)
    assert "func" not in kw
    assert mod.VelocityRangeCurriculum(FakeEnv(), **kw) is not None


def test_none_cfg_is_tolerated():
    assert mod._as_kwargs(None) == {}
