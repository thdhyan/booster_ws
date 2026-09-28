"""Stages that need a GPU, and so run on a remote DGX Spark.

  gvhmr    video  -> SMPL-X AMASS npz      (GVHMR, monocular world-frame)
  kimodo   text   -> SMPL-X AMASS npz      (Kimodo-SMPLX)
  gmr      npz    -> Booster K1 CSV        (mink + MuJoCo IK)

The laptop is a client only, so these cannot run there. Each one is a plain
script on the remote host, invoked over ssh and copied back with scp; nothing
is installed by this module. If a stage's environment is missing, the error
names the exact setup command instead of a stack trace from a missing module.
"""
from __future__ import annotations

import os
import posixpath
import shlex

from . import remote
from .remote import RemoteError

# Host defaults. GMR and Kimodo already live on spark03; GVHMR is being
# installed on spark02 because spark03 is short on free RAM.
GMR_HOST = remote.DEFAULT_GMR_HOST
GMR_DIR = "~/Projects/GMR"
GMR_VENV = "~/Projects/.venv-gmr"

KIMODO_HOST = remote.DEFAULT_KIMODO_HOST
KIMODO_DIR = "~/Projects/kimodo_ws/kimodo"
KIMODO_VENV = "~/Projects/kimodo_ws/kimodo/.venv"  # note: inside the repo

GVHMR_HOST = remote.DEFAULT_GVHMR_HOST
GVHMR_DIR = "~/Projects/GVHMR"
GVHMR_VENV = "~/Projects/.venv-gvhrm"


def _q(host: str, path: str) -> str:
    """Quote a remote path, expanding a leading `~` first.

    `shlex.quote` single-quotes, which stops the remote shell expanding `~`, so
    every `cd ~/Projects/...` would fail with "No such file or directory".
    """
    return shlex.quote(remote.expand(host, path))


def _cd(host: str, path: str) -> str:
    return f"cd {_q(host, path)} && "


class StageUnavailable(RuntimeError):
    """A stage's remote environment is not set up. Carries the fix command."""


def _venv_py(host: str, venv: str, setup_hint: str) -> str:
    py = remote.which_python(host, venv)
    if py is None:
        raise StageUnavailable(
            f"{host}: no interpreter at {venv}/bin/python\n  fix: {setup_hint}")
    return py


# ---------------------------------------------------------------- GVHMR ----

def gvhrm(video_local: str, host: str = GVHMR_HOST, fps: int = 30,
          max_seconds: float = 0.0, workdir: str | None = None,
          static_cam: bool = True) -> str:
    """Monocular video -> SMPL-X AMASS npz. Returns the local npz path.

    GVHMR estimates world-frame, gravity-aligned motion, which is what makes
    it usable for retargeting (screen-space estimates drift and cannot be
    foot-planted).

    Two steps, because GVHMR does not export AMASS itself: `tools/demo/demo.py`
    writes a `pred` torch dict, and `k1m/gvhmr_amass_export.py` reshapes it into
    the AMASS layout GMR consumes.
    """
    py = _venv_py(host, GVHMR_VENV, "see docs/k1m.md for the GVHMR setup")
    if not os.path.exists(video_local):
        raise FileNotFoundError(video_local)
    base = os.path.splitext(os.path.basename(video_local))[0]
    wd = workdir or f"~/Projects/kimodo_ws/runs/gvhrm/{base}"
    remote.run(host, f"mkdir -p {_q(host, wd)}")
    rv = posixpath.join(wd, posixpath.basename(video_local))
    remote.put(host, video_local, rv)

    if max_seconds:
        trimmed = posixpath.join(wd, f"clip_{int(max_seconds)}s.mp4")
        remote.run(host, f"ffmpeg -y -v error -i {_q(host, rv)} -t {max_seconds} "
                         f"-an -c:v libx264 -crf 20 {_q(host, trimmed)}")
        rv = trimmed

    conv = posixpath.join(wd, "gvhmr_amass_export.py")
    remote.put(host, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "gvhmr_amass_export.py"), conv)

    # Running `python tools/demo/demo.py` puts tools/demo on sys.path, not the
    # cwd, so the in-repo `hmr4d` package is not importable without this.
    cmd = (_cd(host, GVHMR_DIR) + "export PYTHONPATH=. && "
           f"{shlex.quote(py)} tools/demo/demo.py "
           f"--video={shlex.quote(rv)} --output_root={shlex.quote(_q(host, wd))}")
    if static_cam:
        # Skip DPVO SLAM: right for a mostly-static camera, and much cheaper.
        cmd += " -s"
    remote.run(host, cmd)

    # demo.py writes hmr4d_results.pt under <output_root>/<clip stem>/
    pred = None
    for cand in (posixpath.join(wd, base, "hmr4d_results.pt"),
                 posixpath.join(wd, "hmr4d_results.pt")):
        if remote.have(host, cand):
            pred = cand
            break
    if pred is None:
        raise RemoteError(
            f"GVHMR wrote no hmr4d_results.pt under {wd}. Its layout is "
            f"<output_root>/<clip stem>/hmr4d_results.pt; list {wd} on {host} "
            f"to see what it actually produced.")
    rnpz = posixpath.join(wd, f"{base}_amass.npz")
    remote.run(host, f"{shlex.quote(py)} {_q(host, conv)} "
                     f"--pred {shlex.quote(remote.expand(host, pred))} "
                     f"--out {shlex.quote(remote.expand(host, rnpz))} --fps {fps}")
    if not remote.have(host, rnpz):
        raise RemoteError(f"AMASS export produced no {rnpz}")
    return remote.get(host, rnpz,
                      os.path.splitext(video_local)[0] + "_amass.npz")


# --------------------------------------------------------------- Kimodo ----

def kimodo(prompt: str, host: str = KIMODO_HOST, seed: int = 7,
           seconds: float = 8.0, model: str = "Kimodo-SMPLX-RP-v1",
           stem: str = "k1m") -> str:
    """Text prompt -> SMPL-X AMASS npz. Returns the local npz path.

    The real Kimodo entry point is `python -m kimodo.scripts.generate`. It
    takes the prompt as a positional arg, and `--output STEM` writes
    `STEM_amass.npz` -- the AMASS file, which is what GMR consumes. Verified
    against the working Macarena invocation on spark03.
    """
    py = _venv_py(host, KIMODO_VENV, "see docs/k1m.md for Kimodo setup")
    ro = f"~/Projects/kimodo_ws/kimodo/out/{stem}_amass.npz"
    remote.run(host, _cd(host, KIMODO_DIR) + "mkdir -p out && "
                     f"{shlex.quote(py)} -m kimodo.scripts.generate "
                     f"{shlex.quote(prompt)} --model {model} "
                     f"--duration {seconds} --num_samples 1 --seed {seed} "
                     f"--no-postprocess --output out/{stem}")
    if not remote.have(host, ro):
        raise RemoteError(f"Kimodo produced no {ro}")
    return remote.get(host, ro, os.path.join("k1m_out", f"{stem}_amass.npz"))


# ------------------------------------------------------------------ GMR ----

def gmr(smplx_npz: str, out_csv: str, host: str = GMR_HOST,
        robot: str = "booster_k1", fps: int = 30,
        resample_fps: int = 0) -> str:
    """SMPL-X AMASS npz -> Booster K1 motion CSV. Returns the local CSV path.

    `smplx_npz` may be local (it is pushed) or already a path on `host`.
    """
    py = _venv_py(host, GMR_VENV, "see docs/k1m.md for GMR setup")
    wd = "~/Projects/kimodo_ws/runs/gmr"
    remote.run(host, f"mkdir -p {_q(host, wd)}")
    if os.path.exists(smplx_npz):
        rnpz = posixpath.join(wd, posixpath.basename(smplx_npz))
        remote.put(host, smplx_npz, rnpz)
    else:
        rnpz = smplx_npz
    # Argument to the remote *interpreter*, not the shell, so `~` must already
    # be expanded -- python never expands it.
    rnpz = remote.expand(host, rnpz)
    rcsv = posixpath.join(wd, "out.csv")
    rpkl = _q(host, posixpath.join(wd, "out.pkl"))
    out = (_cd(host, GMR_DIR) + "export PYTHONPATH=. && "
           f"{shlex.quote(py)} scripts/smplx_to_robot.py "
           f"--smplx_file {shlex.quote(rnpz)} --robot {robot} --headless "
           f"--save_path {rpkl}")
    remote.run(host, out)
    conv = ["pkl_to_csv.py",
            "--in", remote.expand(host, posixpath.join(wd, "out.pkl")),
            "--out", remote.expand(host, rcsv),
            "--xml", "assets/booster_k1/K1_serial.xml"]
    if resample_fps:
        conv += ["--fps", str(resample_fps)]
    # Ship the converter from the package rather than relying on it being
    # present in the remote GMR checkout, so k1m works on a fresh clone.
    conv_path = posixpath.join(wd, "pkl_to_csv.py")
    remote.put(host, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "pkl_to_csv.py"), conv_path)
    remote.run(host, _cd(host, GMR_DIR) + f"{shlex.quote(py)} "
                     + _q(host, conv_path) + " "
                     + " ".join(shlex.quote(str(a)) for a in conv[1:]))
    if not remote.have(host, rcsv):
        raise RemoteError(f"GMR produced no {rcsv}")
    return remote.get(host, rcsv, out_csv)


def _imports_ok(host: str, py: str, mods: list[str], cwd: str = "") -> str:
    """Import-check on the remote venv. Returns '' if all fine, else the error.

    `cwd` matters: stage packages (kimodo, gvhmr) live inside their repo, so
    `import <pkg>` only resolves from that directory.
    """
    code = "; ".join(f"import {m}" for m in mods)
    pre = f"cd {shlex.quote(remote.expand(host, cwd))} && " if cwd else ""
    try:
        out = remote.run(host, f"{pre}{shlex.quote(py)} -c {shlex.quote(code)}",
                         timeout=240, quiet=True)
        if "Traceback" not in out and "Error" not in out:
            return ""
        # Surface the actual missing module, not a truncated "ModuleNo..."
        for line in out.splitlines():
            if "No module named" in line:
                return line.strip()
        lines = out.strip().splitlines()
        return lines[-1][:200] if lines else "unknown"
    except Exception as e:                            # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:160]}"


def preflight(host: str = GMR_HOST) -> list[str]:
    """Report which remote stages are actually usable right now. Never raises.

    Checks the venv *and* that its key modules import -- an empty venv is a
    common state after a failed install and would otherwise look fine.
    """
    out = []
    stages = (
        ("gmr", GMR_HOST, GMR_VENV, ["torch", "mujoco", "mink"], GMR_DIR),
        ("kimodo", KIMODO_HOST, KIMODO_VENV, ["torch", "kimodo"], KIMODO_DIR),
        ("gvhrm", GVHMR_HOST, GVHMR_VENV,
         ["torch", "cv2", "smplx", "detectron2"], GVHMR_DIR),
    )
    for name, h, venv, mods, cwd in stages:
        try:
            py = remote.which_python(h, venv)
            if py is None:
                out.append(f"{name:7s} {h:12s} MISSING venv {venv}")
                continue
            err = _imports_ok(h, py, mods, cwd)
            out.append(f"{name:7s} {h:12s} "
                       + ("ready" if not err else f"venv present, import failed: {err}"))
        except Exception as e:                        # noqa: BLE001
            out.append(f"{name:7s} {h:12s} UNREACHABLE ({type(e).__name__})")
    return out
