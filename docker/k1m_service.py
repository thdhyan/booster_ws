"""HTTP front end for the K1 video -> motion retarget pipeline.

WHY THIS EXISTS
---------------
The pipeline already works as a CLI (``k1m video dance.mp4``) and the Gradio UI in
``app.py`` wraps it. Neither is usable as a service: the UI is a single-user
Python process on one workstation, and the stages it drives (GVHMR on
aim_spark02, GMR/KIMODO on aim_spark03) are remote over SSH. So the useful unit
is not a UI, it is an endpoint that accepts a video and returns retargeted K1
joint angles plus the gate verdict.

The heavy lifting stays where it is. GVHMR and GMR are not installed in this
image -- ``k1m.remote_stages`` reaches them over ssh, and ``remote.py`` resolves
the host aliases through ~/.ssh/config. So this image needs no CUDA, no torch and
no GVHMR checkout; it is a thin orchestrator. That is the whole reason it is
small enough to build in a couple of minutes.

Endpoints
---------
GET  /health              liveness
GET  /doctor              which stages are reachable, and their versions
POST /retarget            multipart video -> CSV + gate verdict + replay mp4
GET  /artifact/{name}     fetch a file this service produced

The gate verdict is returned as data, not as a 200. ``k1m`` is a kinematic IK
solver with no foot locking, no dynamics and no actuator model, so a clean-looking
replay can still be untrackable: an early Macarena render looked fine while asking
AAHead_yaw for 9.58x its effort limit. Callers must read ``gate.pass``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

# The repo is mounted at /workspace/booster_ws; make the package importable even
# if the image was built from a different context root.
REPO = Path(os.environ.get("K1M_REPO", "/workspace/booster_ws")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUTDIR = Path(os.environ.get("K1M_OUTDIR", "/data/k1m_out"))
MAX_UPLOAD_MB = int(os.environ.get("K1M_MAX_UPLOAD_MB", "512"))

app = FastAPI(title="K1 motion retarget", version="1.0")

# One retarget at a time. GVHMR is minutes per video and the remote stages are
# shared, so unbounded concurrency would just queue on the far side while holding
# uploaded videos in tmpfs. Callers get 429 instead of a silent hang.
_lock = threading.Lock()
ALLOWED_SUFFIX = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _fail(msg: str, code: int = 500) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=code)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "repo": str(REPO), "outdir": str(OUTDIR)}


@app.get("/doctor")
def doctor() -> JSONResponse:
    """Report stage reachability. Never raises: a dead stage is data, not a 500."""
    from k1m import remote, remote_stages

    rows: list[dict] = []
    for label, host in (
        ("gvhrm", remote.DEFAULT_GVHMR_HOST),
        ("gmr", remote.DEFAULT_GMR_HOST),
        ("kimodo", remote.DEFAULT_KIMODO_HOST),
    ):
        entry = {"stage": label, "host": host}
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host,
                 "echo ok; nvidia-smi --query-gpu=name --format=csv,noheader | head -1"],
                capture_output=True, text=True, timeout=40,
            )
            lines = [x for x in r.stdout.splitlines() if x.strip()]
            entry["reachable"] = bool(lines and lines[0].strip() == "ok")
            entry["gpu"] = lines[1].strip() if len(lines) > 1 else ""
            if not entry["reachable"]:
                entry["error"] = (r.stderr or "").strip()[:200]
        except Exception as exc:  # noqa: BLE001 - report, never raise
            entry["reachable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
    return JSONResponse({"ok": any(r["reachable"] for r in rows), "stages": rows})


@app.get("/artifact/{name:path}")
def artifact(name: str) -> FileResponse:
    """Serve a produced file.

    The path is relative to OUTDIR and may include the per-run subdirectory
    (``tennis4/tennis4_k1.csv``), because that is where the pipeline writes.
    Confinement uses Path.relative_to rather than a string prefix: a prefix test
    would accept ``/data_evil/x`` for an OUTDIR of ``/data``, since
    ``"/data_evil".startswith("/data")`` is True.
    """
    root = OUTDIR.resolve()
    try:
        path = (root / name).resolve()
        path.relative_to(root)          # raises ValueError if it escapes
    except (ValueError, OSError):
        raise HTTPException(400, "artifact path escapes the output directory")
    if not path.is_file():
        raise HTTPException(404, f"no such artifact: {name}")
    return FileResponse(path)


@app.post("/retarget")
async def retarget(
    video: UploadFile = File(..., description="human video, ideally full-body side or front view"),
    fps: float = Form(30.0),
    seconds: float = Form(0.0, description="0 = whole clip"),
    do_gate: bool = Form(True),
    do_render: bool = Form(False),
    stem: str = Form(""),
) -> JSONResponse:
    t0 = time.time()
    name = (video.filename or "clip").strip() or "clip"
    if Path(name).suffix.lower() not in ALLOWED_SUFFIX:
        return _fail(f"unsupported video type {Path(name).suffix!r}; "
                     f"want one of {sorted(ALLOWED_SUFFIX)}", 400)

    if not _lock.acquire(blocking=False):
        return _fail("a retarget is already running; this pipeline is minutes per "
                     "video and the remote stages are shared", 429)

    tmpdir = Path(tempfile.mkdtemp(prefix="k1m_up_"))
    try:
        src = tmpdir / Path(name).name
        nbytes = 0
        with src.open("wb") as fh:
            while chunk := await video.read(4 << 20):
                nbytes += len(chunk)
                if nbytes > MAX_UPLOAD_MB * 1024 * 1024:
                    return _fail(f"video exceeds {MAX_UPLOAD_MB} MB", 413)
                fh.write(chunk)
        if nbytes == 0:
            return _fail("uploaded video is empty", 400)

        OUTDIR.mkdir(parents=True, exist_ok=True)
        out_stem = stem or f"{Path(name).stem}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTDIR / out_stem
        run_dir.mkdir(parents=True, exist_ok=True)

        from k1m import pipeline

        # Run the blocking pipeline off the event loop.
        #
        # do_gate/do_render MUST be forwarded: pipeline.from_video runs the gate
        # AND the replay renderer internally, so accepting these flags and not
        # passing them means do_render=false is silently ignored and every request
        # dies in mujoco's EGL init inside a CPU-only image.
        def _work():
            return pipeline.from_video(
                str(src), outdir=str(run_dir), stem=out_stem,
                fps=float(fps), max_seconds=float(seconds),
                do_gate=bool(do_gate), do_render=bool(do_render),
                render_gl=os.environ.get("K1M_RENDER_GL", "egl"),
            )

        res = await __import__("asyncio").to_thread(_work)

        # The pipeline already gated, so do NOT run it a second time. Read the
        # report it wrote.
        csvp = getattr(res, "motion_csv", "") or ""
        gate = None
        if do_gate and getattr(res, "gate_json", "") and os.path.exists(res.gate_json):
            with open(res.gate_json) as fh:
                rep = json.load(fh)
            gate = {
                "pass": bool(getattr(res, "gate_passed", False)),
                "summary": getattr(res, "gate_summary", ""),
                "report": os.path.basename(res.gate_json),
            }
            failing = [
                {"check": c.get("name", c.get("check", "?")),
                 "passed": c.get("passed"), "detail": c.get("detail", "")}
                for r in rep.get("reports", []) for c in r.get("checks", [])
                if not c.get("passed", False)
            ][:20]
            if failing:
                gate["failing_checks"] = failing
        replay = getattr(res, "replay_mp4", "") or ""

        payload = {
            "ok": True,
            "stem": out_stem,
            "seconds": round(time.time() - t0, 1),
            "n_frames": int(getattr(res, "n_frames", 0)),
            "csv": os.path.basename(csvp) if csvp else None,
            "replay_mp4": os.path.basename(replay) if replay and os.path.exists(replay) else None,
            "gate": gate,
            "warnings": list(getattr(res, "warnings", []) or []),
            # Paths are relative to OUTDIR and INCLUDE the per-run subdirectory,
            # because that is where the pipeline writes. Serving bare basenames
            # 404s.
            "download": {
                "csv": f"/artifact/{out_stem}/{os.path.basename(csvp)}" if csvp else None,
                "replay_mp4": f"/artifact/{out_stem}/{os.path.basename(replay)}"
                if replay and os.path.exists(replay) else None,
                "gate_report": f"/artifact/{out_stem}/{os.path.basename(res.gate_json)}"
                if do_gate and getattr(res, "gate_json", "") else None,
            },
        }
        # A gate failure is a 200 with ok=false: the CSV exists and the caller
        # needs it. Only transport problems are 4xx/5xx.
        if gate is not None and not gate.get("pass", True):
            payload["ok"] = False
            payload["reason"] = "gate failed: retargeted motion is not trackable"
        return JSONResponse(payload)
    except Exception as exc:  # noqa: BLE001 - surface the reason, keep the service up
        traceback.print_exc()
        return _fail(f"{type(exc).__name__}: {exc}", 500)
    finally:
        _lock.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse(
        "K1 motion retarget service.\n"
        "GET  /doctor            stage reachability\n"
        "POST /retarget          multipart: video (+fps,seconds,do_gate,do_render)\n"
        "GET  /artifact/{name}   fetch a produced CSV or replay mp4\n"
    )
