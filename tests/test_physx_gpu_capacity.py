"""The GPU broadphase pair buffer must scale with env count, or contacts vanish.

PhysX sizes ``PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity`` once, at
``2**21``.  Past that capacity it does not raise -- it logs once per physics step
and **drops contacts**:

    PhysX error: The application needs to increase
    PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 2099817
    , otherwise, the simulation will miss interactions

Measured on the 8192-env K1 run: demand 2 099 817 against the 2 097 152 default,
short by 2 665 pairs (0.13%).  The error flood cost ~6x wall clock and, worse,
the feet passed through the ground so the policy trained against a wrong contact
model.

Three traps this test exists to close, all of which were hit in turn:

* It is invisible at small env counts.  A 2048-env A/B of the same two gain
  settings differed by 0.3%, because 2048 envs never approach the cap -- so the
  overflow looks like a *gains* problem at 8192 and like nothing at all at 2048.
* There is no ``sim.physx``.  ``SimulationCfg`` has one ``physics: PhysicsCfg |
  None`` field, still ``None`` during ``__post_init__``; ``SimulationContext``
  substitutes a stock ``PhysxCfg()`` later.  A fix that mutates ``sim.physx``
  is silently a no-op -- and a test written against that invented attribute
  passes happily while the box keeps overflowing.
* It is not a constant.  Hardcoding a bigger number fixes 8192 and silently
  re-breaks at 16 384.

``sim_backend`` imports ``isaaclab.sim`` at module scope, so Isaac is stubbed here
to keep this a laptop test; the sizing arithmetic under test is pure.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SIM_BACKEND = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/sim_backend.py"
CFG_ROOT = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks"

#: What the 8192-env K1 actually demanded, verbatim from the PhysX error.
OBSERVED_DEMAND_8192 = 2_099_817
#: The PhysX default we must exceed.
PHYSX_DEFAULT = 2**21  # 2_097_152


def _load_sim_backend():
    """Import sim_backend with stubbed isaaclab/isaaclab_physx so no Isaac is needed."""
    if "isaaclab" not in sys.modules:
        pkg = types.ModuleType("isaaclab")
        sim = types.ModuleType("isaaclab.sim")

        class SimulationCfg:  # minimal stand-in; only the type is referenced
            pass

        sim.SimulationCfg = SimulationCfg
        pkg.sim = sim
        sys.modules["isaaclab"] = pkg
        sys.modules["isaaclab.sim"] = sim
    if "isaaclab_physx" not in sys.modules:
        physx_pkg = types.ModuleType("isaaclab_physx")
        physx_mod = types.ModuleType("isaaclab_physx.physics")

        class PhysxCfg:
            """Stands in for the real PhysxCfg, defaults included."""

            def __init__(self):
                self.gpu_total_aggregate_pairs_capacity = PHYSX_DEFAULT

        physx_mod.PhysxCfg = PhysxCfg
        physx_pkg.physics = physx_mod
        sys.modules["isaaclab_physx"] = physx_pkg
        sys.modules["isaaclab_physx.physics"] = physx_mod
    spec = importlib.util.spec_from_file_location("_k1_sim_backend_test", SIM_BACKEND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakePhysxCfg:
    def __init__(self, capacity=PHYSX_DEFAULT):
        self.gpu_total_aggregate_pairs_capacity = capacity


class _FakeEnvCfg:
    """Mirrors SimulationCfg: one ``physics`` field, no ``physx`` attribute."""

    def __init__(self, num_envs, physics=None):
        self.scene = types.SimpleNamespace(num_envs=num_envs)
        self.sim = types.SimpleNamespace(physics=physics)


def test_capacity_clears_the_measured_8192_demand():
    mod = _load_sim_backend()
    cfg = _FakeEnvCfg(num_envs=8192, physics=_FakePhysxCfg())
    got = mod.ensure_physx_gpu_capacity(cfg)
    assert got >= OBSERVED_DEMAND_8192, (
        f"8192 envs still overflow: capacity {got} < observed demand "
        f"{OBSERVED_DEMAND_8192}; contacts will be silently dropped"
    )
    assert cfg.sim.physics.gpu_total_aggregate_pairs_capacity > PHYSX_DEFAULT


def test_supplies_the_physx_cfg_rather_than_mutating_a_nonexistent_physx_field():
    """The whole fix hinges on this: under PhysX, sim.physics starts as None.

    ``SimulationContext`` would otherwise substitute a stock ``PhysxCfg()`` at
    construction time, which is where the 2**21 default comes from. So the env cfg
    has to supply its own, already-raised, or the change is a silent no-op.
    """
    mod = _load_sim_backend()
    cfg = _FakeEnvCfg(num_envs=8192, physics=None)
    assert cfg.sim.physics is None
    got = mod.ensure_physx_gpu_capacity(cfg)
    assert cfg.sim.physics is not None, "the PhysxCfg must be attached to sim.physics"
    assert cfg.sim.physics.gpu_total_aggregate_pairs_capacity == got
    assert not hasattr(cfg.sim, "physx"), (
        "SimulationCfg has no `physx` attribute; anything reading one is a no-op"
    )


def test_capacity_scales_with_env_count_not_a_constant():
    """A 16384-env run must not re-break the way 8192 did."""
    mod = _load_sim_backend()
    small = _FakeEnvCfg(num_envs=8192, physics=_FakePhysxCfg())
    large = _FakeEnvCfg(num_envs=16384, physics=_FakePhysxCfg())
    mod.ensure_physx_gpu_capacity(small)
    mod.ensure_physx_gpu_capacity(large)
    assert (
        large.sim.physics.gpu_total_aggregate_pairs_capacity
        > small.sim.physics.gpu_total_aggregate_pairs_capacity
    ), "capacity did not grow with num_envs; a larger run would overflow again"


def test_never_shrinks_an_already_larger_buffer():
    mod = _load_sim_backend()
    generous = 2**24
    cfg = _FakeEnvCfg(num_envs=1024, physics=_FakePhysxCfg(capacity=generous))
    mod.ensure_physx_gpu_capacity(cfg)
    assert cfg.sim.physics.gpu_total_aggregate_pairs_capacity == generous


def test_small_run_gets_the_floor_and_not_a_silly_size():
    mod = _load_sim_backend()
    cfg = _FakeEnvCfg(num_envs=64, physics=_FakePhysxCfg())
    assert mod.ensure_physx_gpu_capacity(cfg) == mod.MIN_AGGREGATE_PAIRS


def test_newton_style_cfg_without_the_knob_is_a_no_op_not_a_crash():
    mod = _load_sim_backend()

    class _Newtonish:
        pass

    cfg = _FakeEnvCfg(num_envs=8192, physics=_Newtonish())
    assert mod.ensure_physx_gpu_capacity(cfg) == 0


def test_every_env_cfg_family_raises_the_capacity():
    """The overflow is a property of the scene, so no task family may skip it.

    Inheritance counts: several families (``partial``, ``squat``) subclass
    ``K1VelocityRoughEnvCfg`` and pick the fix up through
    ``super().__post_init__()``. Requiring a literal call in every file flagged
    them as broken when they are fine, so ancestry is followed instead. The check
    still has teeth: ``head`` derives straight from ``ManagerBasedRLEnvCfg`` and
    was genuinely missed until this test existed.
    """
    calls: dict[str, bool] = {}
    bases: dict[str, list[str]] = {}
    for cfg_file in sorted(CFG_ROOT.rglob("*_env_cfg.py")):
        for node in ast.walk(ast.parse(cfg_file.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            calls[node.name] = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "ensure_physx_gpu_capacity"
                for n in ast.walk(node)
            )
            bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)] + [
                b.attr for b in node.bases if isinstance(b, ast.Attribute)
            ]

    def reaches(cls: str, seen: set[str] | None = None) -> bool:
        seen = seen or set()
        if cls in seen or cls not in calls:
            return False
        if calls[cls]:
            return True
        seen.add(cls)
        return any(reaches(b, seen) for b in bases.get(cls, ()))

    offenders = sorted(
        name
        for name in calls
        if name.endswith("EnvCfg") and not name.endswith("PlayEnvCfg") and not reaches(name)
    )
    assert not offenders, f"env cfgs that never raise the GPU pair capacity: {offenders}"


def test_the_mechanism_and_the_field_name_are_documented_where_it_lives():
    """The 2048-vs-8192 trap and the missing ``sim.physx`` are why this misfired."""
    src = SIM_BACKEND.read_text()
    assert "2099817" in src.replace(" ", ""), (
        "the measured 8192-env demand should be recorded in the docstring"
    )
    assert "sim.physx" in src, "the nonexistent-attribute trap should be recorded"