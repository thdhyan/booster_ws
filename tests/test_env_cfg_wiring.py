"""Static audit that every Isaac Lab manager group is actually loadable.

ManagerBase._prepare_terms() iterates ``self.cfg.__dict__.items()``.  For a class
decorated with ``@configclass`` those terms are dataclass *fields*, so they land
in the instance ``__dict__`` and get picked up.  For a plain class the terms are
only *class* attributes, the instance ``__dict__`` is empty, and the manager
silently loads **zero** terms -- no warning, no exception.

That happened to the P2 velocity task: ``TerminationsCfg`` was missing its
``@configclass`` decorator, so the fall terminations never ran.  The robot could
sink to the floor mid-episode and training never saw it, which is the same
blind spot that let P3 lie face-down for 500 steps.

This is a pure source check -- no Isaac Sim required.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_ROOT = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks"

# Group classes the managers load from cfg.<attr>.  Nesting depth is used to
# ignore the inner classes (e.g. ObservationsCfg.PolicyCfg).
GROUP_RE = re.compile(r"^(\s*)class\s+(\w*Cfg)\s*[:(]")
DECORATOR_RE = re.compile(r"^\s*@configclass\b")


def _cfg_files() -> list[pathlib.Path]:
    return sorted(p for p in CFG_ROOT.rglob("*cfg.py") if p.is_file())


def test_cfg_files_found():
    files = _cfg_files()
    assert files, f"no env config files under {CFG_ROOT}"


@pytest.mark.parametrize("path", _cfg_files(), ids=lambda p: p.name)
def test_manager_groups_are_configclass(path: pathlib.Path):
    lines = path.read_text().splitlines()
    missing = []
    for i, line in enumerate(lines):
        m = GROUP_RE.match(line)
        if not m:
            continue
        indent, name = m.group(1), m.group(2)
        # only top-level (indent 0) classes are manager groups
        if indent:
            continue
        prev = lines[i - 1].strip() if i else ""
        if not DECORATOR_RE.match(prev):
            missing.append(name)
    assert not missing, (
        f"{path.name}: manager group(s) {missing} are missing @configclass -- "
        "the manager will load ZERO terms for them, silently"
    )
