"""Import-ordering guard for the Isaac entrypoint scripts.

The failure this prevents produced the single most misleading error in this project:

    ValueError: Unknown asset config type for height_scanner: RayCasterCfg(...)

`height_scanner` IS a `RayCasterCfg`; `RayCasterCfg` IS a `SensorBaseCfg` subclass; and
`SensorBaseCfg` is checked at line 947 of interactive_scene, well before the `raise` at
994. So the message is unreachable by the reasoning it invites, and the real cause --
importing isaaclab modules before `SimulationApp` exists, which splits module identity so
`interactive_scene` tests the scene cfgs against a different `SensorBaseCfg` object than
the one they were built from -- appears nowhere in it.

That cost four wrong diagnoses (class identity, branch order, GPU contention, a stale
clone) before a probe that instantiated the launcher first passed the identical build.
The rule is simple enough to enforce mechanically: no isaaclab import above the line that
constructs the AppLauncher, except `isaaclab.app` itself.

These are static AST checks on purpose. Nothing here imports Isaac, so the guard runs in
the fast unit suite -- which is the only place a check can catch the mistake BEFORE a
4-minute container run does.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "isaac_tasks/k1_velocity/scripts"

# Modules that must not be imported before the app exists. `isaaclab.app` is the one
# exception: the launcher itself has to be importable to be constructed.
GUARDED_PREFIXES = ("isaaclab.", "isaaclab_tasks", "isaaclab_rl", "rsl_rl", "booster_train")


def _imported_modules(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (lineno, module) for every `import x` / `from x import ...` in the tree."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, node.module))
    return out


def _app_launcher_line(tree: ast.Module) -> int | None:
    """Line of the `AppLauncher(...)` call, if the script constructs one."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "AppLauncher":
                return node.lineno
    return None


def _scripts() -> list[pathlib.Path]:
    return sorted(p for p in SCRIPTS.glob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.name)
def test_no_isaaclab_import_precedes_the_app_launcher(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text())
    launch_line = _app_launcher_line(tree)
    if launch_line is None:
        pytest.skip(f"{path.name} does not construct an AppLauncher")

    offenders = [
        (lineno, module)
        for lineno, module in _imported_modules(tree)
        if lineno < launch_line
        and module.startswith(GUARDED_PREFIXES)
        and module != "isaaclab.app"
    ]
    assert not offenders, (
        f"{path.name} imports {offenders} before constructing AppLauncher on line "
        f"{launch_line}. SimulationApp must exist first: importing isaaclab modules "
        f"early splits module identity, so interactive_scene tests the scene cfgs "
        f"against a different SensorBaseCfg object than the one they were built from and "
        f"raises 'Unknown asset config type for height_scanner'."
    )


@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.name)
def test_app_launcher_is_not_constructed_twice(path: pathlib.Path) -> None:
    """Two launchers means two apps; the second is always a mistake."""
    tree = ast.parse(path.read_text())
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "AppLauncher"
    ]
    assert len(lines) <= 1, f"{path.name} constructs AppLauncher {len(lines)} times (lines {lines})"
