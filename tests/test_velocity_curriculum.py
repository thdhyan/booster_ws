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
class FakeRanges:
    """Matches UniformVelocityCommandCfg.Ranges, which is what the term reads."""

    def __init__(self, lin=0.5, ang=1.0):
        self.lin_vel_x = (-lin, lin)
        self.lin_vel_y = (-lin, lin)
        self.ang_vel_z = (-ang, ang)


class FakeCfg:
    def __init__(self, lin=0.5, ang=1.0):
        self.ranges = FakeRanges(lin, ang)


class FakeTerm:
    """Stands in for Isaac Lab's UniformVelocityCommand.

    The ranges live at term.cfg.ranges, NOT on the term as attributes: the real
    command samples with r.uniform_(*self.cfg.ranges.lin_vel_x). A fake that put
    them on the term is what made the first two launch attempts look plausible
    and fail in the container.
    """

    def __init__(self, lin=0.5, ang=1.0):
        self.cfg = FakeCfg(lin, ang)

    @property
    def lin(self):
        return self.cfg.ranges.lin_vel_x[1]

    @property
    def ang(self):
        return self.cfg.ranges.ang_vel_z[1]


class FakeTensor:
    """Stand-in for a torch scalar: supports .mean() and .item()."""

    def __init__(self, v):
        self._v = float(v)

    def mean(self):
        return self

    def item(self):
        return self._v


class FakeStepReward:
    """Stand-in for the (num_envs, num_terms) tensor, so `[:, idx]` works.

    Values are produced from a live callable, not a snapshot: the real
    RewardManager recomputes _step_reward every step, so a fake that captured
    the value at construction would silently ignore later changes and make the
    contraction path untestable.
    """

    def __init__(self, value_fn, reads=None):
        self._value_fn = value_fn
        self._reads = reads if reads is not None else []

    def __getitem__(self, key):
        self._reads.append(key)
        col = key[1] if isinstance(key, tuple) else key
        return FakeTensor(self._value_fn(col))


class FakeRewardTermCfg:
    def __init__(self, weight):
        self.weight = weight


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
        self.reward_reads = []

        class FakeCommandTensor:
            """get_command returns a TENSOR, not the term. Using it is the bug."""

        class CM:
            def get_command(self, name):
                return FakeCommandTensor()

            def get_term(self, name):
                assert name == "base_velocity"
                return outer.term

        class RM:
            # Shaped like Isaac Lab's RewardManager: NO get_term, per-step values
            # in a (num_envs, num_terms) matrix indexed by _term_names, and the
            # stored values are func * weight.
            active_terms = ["track_lin_vel_xy_exp", "other"]
            _term_names = ["track_lin_vel_xy_exp", "other"]
            WEIGHT = 10.0

            def __init__(self):
                # func * weight, recomputed on every read like the real manager.
                self._step_reward = FakeStepReward(
                    lambda col: (
                        outer._tracking * self.WEIGHT if col == 0 else 0.0
                    ),
                    reads=outer.reward_reads,
                )

            def get_term_cfg(self, name):
                return FakeRewardTermCfg(self.WEIGHT)

            def get_active_iterable_terms(self, env_idx):
                outer.tracking_calls += 1
                return [
                    ("track_lin_vel_xy_exp", [outer._tracking * self.WEIGHT]),
                    ("other", [0.0]),
                ]

        self.command_manager = CM()
        self.reward_reads = []
        self.reward_manager = RM()
        self.tracking_calls = 0


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
    # The first call installs the range; it must not also gate on a reward.
    assert len(env.reward_reads) == 0
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
    lo, hi = env.term.cfg.ranges.lin_vel_x
    assert lo == pytest.approx(-hi), "backward range must widen with forward"
    lo, hi = env.term.cfg.ranges.lin_vel_y
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
    """A term without reachable ranges must be loud, not a no-op curriculum."""

    class NoRanges:
        def __init__(self):
            self.cfg = object()

    env = FakeEnv()
    env.term = NoRanges()
    cur = make(env)
    with pytest.raises(AttributeError, match="ranges"):
        cur(env, None)


def test_raises_when_ranges_lack_a_velocity_axis():
    class PartialRanges:
        lin_vel_x = (-0.5, 0.5)

    class Term:
        def __init__(self):
            self.cfg = type("C", (), {"ranges": PartialRanges()})()

    env = FakeEnv()
    env.term = Term()
    cur = make(env)
    with pytest.raises(AttributeError, match="lin_vel_y"):
        cur(env, None)


def test_detects_a_write_that_does_not_take_effect():
    """A silently-ignored write must be caught, not run 3000 iterations inert."""

    class FrozenRanges(FakeRanges):
        def __setattr__(self, name, value):
            if name == "lin_vel_x" and getattr(self, "_frozen", False):
                return  # pretend the config is immutable
            object.__setattr__(self, name, value)

    # The term must start somewhere else, otherwise the first write is a no-op
    # (0.5 -> 0.5) and the read-back legitimately matches.
    class Term:
        def __init__(self):
            r = FrozenRanges(lin=0.1, ang=0.2)
            object.__setattr__(r, "_frozen", True)
            self.cfg = type("C", (), {"ranges": r})()

    env = FakeEnv()
    env.term = Term()
    cur = make(env)
    with pytest.raises(RuntimeError, match="inert"):
        cur(env, None)


def test_uses_get_term_not_get_command():
    """get_command returns a Tensor; using it is how the container run failed."""
    seen = {"get_term": 0, "get_command": 0}
    env = FakeEnv()
    inner = env.command_manager

    class CM:
        def get_command(self, name):
            seen["get_command"] += 1
            return object()

        def get_term(self, name):
            seen["get_term"] += 1
            return inner._terms["base_velocity"] if hasattr(inner, "_terms") else env.term

    env.command_manager = CM()
    cur = make(env)
    cur(env, None)
    assert seen["get_term"] == 1
    assert seen["get_command"] == 0


def test_interval_gates_measurement_rate():
    env = FakeEnv(tracking=1.0)
    cur = make(env, patience=1, interval_steps=10)
    prime(cur, env)
    run(cur, env, intervals=1)
    assert len(env.reward_reads) == 1, "did not measure once the interval elapsed"


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
class FakeTermCfg:
    """Shape Isaac Lab hands a *term* at construction: values under .params."""

    def __init__(self, **params):
        self.params = params
        self.func = object()
        self._private = 1


def test_params_arrive_nested_under_cfg_params():
    """Isaac passes the whole ManagerTermBaseCfg, so kwargs are nested.

    Getting this wrong is only visible after a full Isaac boot:
    TypeError: ... got an unexpected keyword argument 'params'
    """
    cfg = FakeTermCfg(init_lin_vel=0.5, target_max_lin_vel=1.5, patience=5)
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
    cfg = FakeTermCfg(init_lin_vel=0.5)
    kw = mod._as_kwargs(cfg)
    assert "func" not in kw
    assert mod.VelocityRangeCurriculum(FakeEnv(), **kw) is not None


def test_none_cfg_is_tolerated():
    assert mod._as_kwargs(None) == {}


def test_call_signature_is_not_generic():
    """A **kwargs signature is rejected by Isaac's static check.

    manager_base._resolve_common_term_cfg compares the declared parameter names
    against cfg.params and treats **kwargs as a mandatory parameter named
    'kwargs', so it fails. The concrete signature is checked against the config
    in the two tests below.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__call__":
            assert node.args.kwarg is None, "**kwargs is not accepted by the manager"


def test_adapter_is_instantiated_once_so_state_persists():
    """A fresh instance per step would reset the streak every call.

    Isaac replaces term_cfg.func with the instance in _prepare_terms, so the
    logic object's counters must live on the instance, not in __call__ locals.
    """
    import inspect as _inspect

    sig = _inspect.signature(mod.VelocityRangeCurriculum.__init__)
    # Everything the curriculum needs is set up in __init__ so it can persist.
    assert "env" in sig.parameters
    src = MODULE_PATH.read_text()
    assert "self._streak = 0" in src
    assert "self._lin = float(init_lin_vel)" in src


def _adapter_call_params():
    """Parameter names of VelocityRangeCurriculumTerm.__call__, via AST.

    There are two __call__ definitions in the module -- the logic class's and the
    adapter's -- so this has to select by enclosing class name, not just by
    function name.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VelocityRangeCurriculumTerm":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__call__":
                    a = sub.args
                    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    raise AssertionError("VelocityRangeCurriculumTerm.__call__ not found")


def _velocity_range_cfg_params():
    """Keys of the velocity_range CurrTerm params dict in the env config.

    The config has two CurrTerms (terrain_levels and velocity_range), so this
    selects the one whose func points at the velocity curriculum.
    """
    cfg = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py"
    tree = ast.parse(cfg.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name != "CurrTerm":
            continue
        if "velocity_curriculum" not in ast.dump(node):
            continue
        for kw in node.keywords:
            if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                return [k.value for k in kw.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("no velocity_range CurrTerm(params={...}) found in velocity_env_cfg.py")


def test_call_signature_matches_config_params_exactly():
    """Isaac compares the __call__ signature to cfg.params by name, statically.

    manager_base._resolve_common_term_cfg does
        set(args[min_argc:]) != set(term_params + args_with_defaults)
    and does not understand **kwargs, so a generic signature is rejected with
    "expects mandatory parameters: ['kwargs']". Every config param must appear by
    name in __call__ with a default. This test is what stops that duplication
    from drifting out of sync without a 4-minute Isaac boot.
    """
    call_params = set(_adapter_call_params()) - {"self", "env", "env_ids"}
    cfg_params = set(_velocity_range_cfg_params())
    # Every config param must be declared by name with a default. The reverse is
    # NOT required: the check compares set(args[3:]) against
    # set(term_params + args_with_defaults), and an extra declared-with-default
    # parameter appears on both sides, so it passes. Omitting a config param does
    # not, and that is the direction that breaks at runtime.
    missing = cfg_params - call_params
    assert not missing, (
        f"config passes {sorted(missing)} but __call__ does not declare them; "
        f"declared={sorted(call_params)}"
    )


def test_call_params_all_have_defaults():
    """Params without defaults are counted as mandatory and fail the same check."""
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VelocityRangeCurriculumTerm":
            for sub in node.body:
                if not (isinstance(sub, ast.FunctionDef) and sub.name == "__call__"):
                    continue
                a = sub.args
                positional = [x.arg for x in a.posonlyargs + a.args]
                required = positional[: len(positional) - len(a.defaults)]
                assert set(required) <= {"self", "env", "env_ids"}, (
                    f"these have no default and Isaac will treat them as mandatory: {required}"
                )
                assert a.kwarg is None, (
                    "**kwargs breaks Isaac's static signature check; list the params "
                    "explicitly with defaults instead"
                )
                return
    raise AssertionError("VelocityRangeCurriculumTerm.__call__ not found")


def test_unknown_reward_weight_raises_rather_than_mis_gating():
    """A wrong weight would compare a weighted value to a raw threshold.

    The gate threshold (0.85) is expressed against the RAW exponential, but
    RewardManager stores func * weight. Guessing the weight would silently
    mis-gate the curriculum, so an unreadable weight is an error.
    """
    env = FakeEnv(tracking=0.99)
    # get_term_cfg is a bound method on the class, so shadow it on the instance
    # rather than deleting it.
    env.reward_manager.get_term_cfg = None
    cur = make(env, patience=1)
    prime(cur, env)
    with pytest.raises(AttributeError, match="get_term_cfg"):
        run(cur, env, intervals=1)


def test_zero_weight_raises():
    class ZeroCfg:
        weight = 0.0

    env = FakeEnv(tracking=0.99)
    env.reward_manager.get_term_cfg = lambda name: ZeroCfg()
    cur = make(env, patience=1)
    prime(cur, env)
    with pytest.raises(ValueError, match="weight"):
        run(cur, env, intervals=1)


def test_raw_value_is_weight_normalised():
    """With weight 10 the stored value is 10x the raw one; the gate must see the raw."""
    env = FakeEnv(tracking=0.95)
    assert env.reward_manager.WEIGHT == 10.0
    cur = make(env, patience=1)
    assert cur._reward_mean() == pytest.approx(0.95), (
        "must divide by the weight, otherwise 0.95*10=9.5 would sail past any "
        "threshold on the raw scale"
    )
