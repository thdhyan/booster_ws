"""Stages that run on the local machine: the feasibility gate and the render.

Both wrap the existing scripts via subprocess rather than reimplementing their
logic. `motion_feasibility_gate.py` is the pre-flight authority on whether a
motion is safe to send to hardware, and `record_k1_motion.py` is a validated
dynamic PD replay -- both have accumulated fixes (geom groups, quaternion
order, the K1 sole-frame offset) that would be easy to lose by forking them
into a library. Subprocess also keeps their single source of truth intact.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from . import schema

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _py() -> str:
    """Interpreter with mujoco + numpy. Honours K1M_PYTHON."""
    exe = os.environ.get("K1M_PYTHON") or sys.executable
    try:
        import mujoco  # noqa: F401
    except ImportError:
        alt = os.path.expanduser("~/.venvs/gmr/bin/python")
        if os.path.exists(alt):
            return alt
    return exe


def _env() -> dict:
    """A clean env.

    The ROS 2 workspace exports PYTHONPATH and LD_LIBRARY_PATH that shadow
    libtorch and break `import torch` in any venv (see scripts/py_torch.sh), so
    both are stripped for every child process.
    """
    e = dict(os.environ)
    for k in ("PYTHONPATH", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH",
              "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH", "ROS_DISTRO"):
        e.pop(k, None)
    return e


def gate(motion_csv: str, fps: float = 30.0,
         urdf: str = schema.DEFAULT_K1_URDF,
         report_json: str | None = None) -> dict:
    """Run the feasibility gate. Returns the parsed report.

    The gate is the authority on hardware-readiness. A motion that fails it is
    not "probably fine" -- it is a kinematic solve with no foot locking, no
    dynamics and no actuator model behind it, which is exactly what breaks a
    real robot.
    """
    script = os.path.join(_REPO, "scripts", "motion_feasibility_gate.py")
    report_json = report_json or (os.path.splitext(motion_csv)[0] + "_gate.json")
    p = subprocess.run(
        [sys.executable, script, motion_csv, "--fps", str(fps),
         "--urdf", urdf, "--json", report_json],
        capture_output=True, text=True, env=_env(), cwd=_REPO)
    out = (p.stdout or "") + (p.stderr or "")
    if not os.path.exists(report_json):
        raise RuntimeError(f"gate produced no report (exit {p.returncode})\n"
                           + "\n".join(out.strip().splitlines()[-20:]))
    with open(report_json) as f:
        rep = json.load(f)
    rep["_stdout"] = out
    rep["_report_path"] = report_json
    return rep


def render(motion_csv: str, out_mp4: str, fps: float = 30.0,
           title: str = "", sim_fps: int = 200, width: int = 1280,
           gl: str = "egl") -> str:
    """Render the 3-panel MuJoCo replay to `out_mp4`. Returns the path."""
    script = os.path.join(_REPO, "scripts", "record_k1_motion.py")
    cmd = [sys.executable, script, motion_csv, "--out", out_mp4,
           "--fps", str(fps), "--sim-fps", str(sim_fps),
           "--width", str(width), "--gl", gl]
    if title:
        cmd += ["--title", title]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(out_mp4):
        raise RuntimeError(f"render failed (exit {p.returncode})\n"
                           + "\n".join(((p.stdout or "") + (p.stderr or ""))
                                       .strip().splitlines()[-20:]))
    return out_mp4


def summarize_gate(rep: dict) -> str:
    """One-line human summary of a gate report.

    The report schema is::

        {"reports": [{"path", "fps", "n_frames", "duration_s",
                      "checks": [{"name", "passed", "detail", "worst"}, ...]}]}
    """
    reports = rep.get("reports") or []
    if not reports:
        return "no reports in gate output"
    out = []
    for r in reports:
        checks = r.get("checks", [])
        fails = [c for c in checks if not c.get("passed", False)]
        verdict = "FAIL" if fails else "PASS"
        out.append(f"{verdict}: {r.get('path', '?')} -- "
                   f"{len(checks) - len(fails)}/{len(checks)} checks pass, "
                   f"{len(fails)} blocking")
        for c in fails:
            out.append(f"  - {c.get('name', '?')}: {c.get('detail', '')}")
    return "\n".join(out)
