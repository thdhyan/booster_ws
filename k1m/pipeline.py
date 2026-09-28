"""End-to-end pipelines: video or text in, joint angles + replay video out.

    video ─► GVHMR ─► SMPL-X ─┐
                              ├─► GMR ─► K1 CSV ─► gate ─► MuJoCo replay
    text  ─► Kimodo ──────────┘

Every path ends at the gate, and the gate's verdict is reported rather than
swallowed. That is deliberate: the retargeters are kinematic IK solvers with no
foot locking, no dynamics and no actuator model, so a visually great render can
still be physically untrackable -- an early Macarena render asked `AAHead_yaw`
for 9.58x its effort limit and put a sole 91 mm through the floor. A tool that
only returned a pretty video would hide that.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

from . import local_stages, remote_stages, schema


@dataclass
class Result:
    """Everything one run produced, all in the output directory."""
    outdir: str
    motion_csv: str = ""
    gate_json: str = ""
    gate_summary: str = ""
    replay_mp4: str = ""
    source_npz: str = ""
    fps: float = 30.0
    n_frames: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        if not self.gate_json or not os.path.exists(self.gate_json):
            return False
        with open(self.gate_json) as f:
            return not any(not c.get("passed", False)
                           for r in json.load(f).get("reports", [])
                           for c in r.get("checks", []))

    def to_dict(self) -> dict:
        return {
            "outdir": self.outdir, "motion_csv": self.motion_csv,
            "gate_json": self.gate_json, "replay_mp4": self.replay_mp4,
            "gate_passed": self.gate_passed, "fps": self.fps,
            "n_frames": self.n_frames, "warnings": self.warnings,
            "gate_summary": self.gate_summary,
        }


def _write_fps_sidecar(csv_path: str, fps: float) -> None:
    """Record the sample rate next to a motion CSV, which cannot carry it."""
    with open(os.path.splitext(csv_path)[0] + ".fps.txt", "w") as f:
        f.write(f"{float(fps):.6f}\n")


def _finish(npz: str, outdir: str, stem: str, fps: float,
            do_gate: bool, do_render: bool, render_gl: str,
            title: str) -> Result:
    """Retarget the npz, then gate and render. Shared by every entry point."""
    os.makedirs(outdir, exist_ok=True)
    res = Result(outdir=outdir, source_npz=npz, fps=fps)

    csv_path = os.path.join(outdir, f"{stem}_k1.csv")
    remote_stages.gmr(npz, csv_path)
    # The CSV format carries no fps, so record it alongside; otherwise every
    # velocity/accel number read back from the file is wrong by a constant
    # factor (squared, for acceleration).
    _write_fps_sidecar(csv_path, fps)
    motion, _ = schema.read_csv(csv_path)
    res.motion_csv = csv_path
    res.n_frames = int(motion.shape[0])

    # Cheap structural checks before the expensive ones.
    v = schema.validate(motion)
    if not v["ok"]:
        res.warnings.append("CSV contract: " + "; ".join(v["errors"]))

    if do_gate:
        rep = local_stages.gate(csv_path, fps=fps,
                                report_json=os.path.join(outdir, f"{stem}_gate.json"))
        res.gate_json = rep["_report_path"]
        res.gate_summary = local_stages.summarize_gate(rep)
    else:
        res.warnings.append("gate SKIPPED -- not validated for hardware")

    if do_render:
        res.replay_mp4 = local_stages.render(
            csv_path, os.path.join(outdir, f"{stem}_replay.mp4"),
            fps=fps, title=title or f"K1 - {stem}", gl=render_gl)
    return res


def from_text(prompt: str, outdir: str = "k1m_out", stem: str = "text",
              seconds: float = 8.0, seed: int = 7, fps: float = 30.0,
              do_gate: bool = True, do_render: bool = True,
              render_gl: str = "egl", **kw) -> Result:
    """Text prompt -> K1 joint angles + replay video."""
    t0 = time.time()
    npz = remote_stages.kimodo(prompt, seconds=seconds, seed=seed)
    r = _finish(npz, outdir, stem, fps, do_gate, do_render, render_gl,
                f"K1 - text: {prompt[:40]}")
    r.warnings.append(f"total {time.time() - t0:.0f}s")
    return r


def from_video(video: str, outdir: str = "k1m_out", stem: str = "",
               fps: float = 30.0, do_gate: bool = True, do_render: bool = True,
               render_gl: str = "egl", max_seconds: float = 0.0,
               **kw) -> Result:
    """Video -> GVHMR -> K1 joint angles + replay video."""
    stem = stem or os.path.splitext(os.path.basename(video))[0]
    t0 = time.time()
    npz = remote_stages.gvhrm(video, fps=int(fps), max_seconds=max_seconds)
    r = _finish(npz, outdir, stem, fps, do_gate, do_render, render_gl,
                f"K1 - video: {os.path.basename(video)}")
    r.warnings.append(f"total {time.time() - t0:.0f}s")
    return r


def from_smplx(npz: str, outdir: str = "k1m_out", stem: str = "smplx",
               fps: float = 30.0, do_gate: bool = True,
               do_render: bool = True, render_gl: str = "egl",
               **kw) -> Result:
    """An existing SMPL-X AMASS npz -> K1 joint angles + replay video."""
    stem = stem or os.path.splitext(os.path.basename(npz))[0]
    return _finish(npz, outdir, stem, fps, do_gate, do_render, render_gl,
                   f"K1 - {stem}")


def from_csv(csv_path: str, outdir: str = "k1m_out", fps: float | None = None,
             do_gate: bool = True, do_render: bool = True,
             render_gl: str = "egl", **kw) -> Result:
    """Already-retargeted K1 CSV -> gate + replay. No GPU needed."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"{stem}_k1.csv")
    if os.path.abspath(csv_path) != os.path.abspath(dst):
        shutil.copy(csv_path, dst)
    # Carry the rate across if the source had a sidecar, else record what the
    # caller asserted so the copy is not silently ambiguous.
    src_fps = schema.read_csv(csv_path)[1]
    if fps is None:
        fps = src_fps
    _write_fps_sidecar(dst, fps)
    motion, _ = schema.read_csv(dst)
    r = Result(outdir=outdir, motion_csv=dst, fps=fps,
               n_frames=int(motion.shape[0]))
    v = schema.validate(motion)
    if not v["ok"]:
        r.warnings.append("CSV contract: " + "; ".join(v["errors"]))
    if do_gate:
        rep = local_stages.gate(dst, fps=fps,
                                report_json=os.path.join(outdir, f"{stem}_gate.json"))
        r.gate_json = rep["_report_path"]
        r.gate_summary = local_stages.summarize_gate(rep)
    if do_render:
        r.replay_mp4 = local_stages.render(
            dst, os.path.join(outdir, f"{stem}_replay.mp4"),
            fps=fps, title=f"K1 - {stem}", gl=render_gl)
    return r
