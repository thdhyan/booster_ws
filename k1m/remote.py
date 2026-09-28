"""Run pipeline stages on a remote GPU host, or locally.

The laptop is a client only -- the motion models (GVHMR, Kimodo) and the GMR
IK solve need a real GPU, and on aarch64 boxes that means a DGX Spark. The
gate and the MuJoCo render are cheap enough to run locally, so the pipeline
splits across both rather than round-tripping every stage.

This module is deliberately thin: it runs a shell command on a host and
streams files back. All the real logic lives in the stage scripts, which are
ordinary files on the remote host and can also be run by hand.
"""
from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import sys

DEFAULT_GMR_HOST = "aim_spark03"
DEFAULT_KIMODO_HOST = "aim_spark03"
DEFAULT_GVHMR_HOST = "aim_spark02"


class RemoteError(RuntimeError):
    pass


def _base(host: str) -> list[str]:
    return ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host]


def run(host: str, cmd: str, timeout: int = 3600,
        check: bool = True, quiet: bool = False) -> str:
    """Run `cmd` on `host` via non-interactive ssh. Returns stdout.

    stderr is folded into stdout so remote tracebacks are not lost, which is
    the usual reason a stage "silently" produces no output.
    """
    full = " ".join(_base(host)) + f" {shlex.quote(cmd)}"
    if not quiet:
        print(f"[k1m] {host}$ {cmd}", file=sys.stderr, flush=True)
    p = subprocess.run(full, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    if check and p.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        raise RemoteError(f"{host}: exit {p.returncode}\n{tail}")
    return out


def put(host: str, local: str, remote: str) -> str:
    """Copy a local file to `remote` (a full path on `host`)."""
    remote = expand(host, remote)
    d = posixpath.dirname(remote)
    if d:
        run(host, f"mkdir -p {shlex.quote(d)}")
    p = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", local,
                        f"{host}:{remote}"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RemoteError(f"scp {local} -> {host}:{remote} failed: {p.stderr}")
    return remote


def get(host: str, remote: str, local: str) -> str:
    """Copy a file back from `host` to a local path.

    `remote` is expanded to an absolute path rather than relying on scp's
    tilde handling, which is not consistent across implementations.
    """
    remote = expand(host, remote)
    d = os.path.dirname(os.path.abspath(local))
    if d:
        os.makedirs(d, exist_ok=True)
    p = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10",
                        f"{host}:{remote}", local],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RemoteError(f"scp {host}:{remote} -> {local} failed: {p.stderr}")
    return local


_HOME_CACHE: dict[str, str] = {}


def home(host: str) -> str:
    """Remote $HOME, cached.

    Stage paths are written as `~/Projects/...` for readability, but `shlex.quote`
    single-quotes them, which stops the remote shell expanding `~`. Resolving
    `~` here means the paths can stay readable and still work.
    """
    if host not in _HOME_CACHE:
        p = subprocess.run(" ".join(_base(host)) + " 'echo $HOME'",
                           shell=True, capture_output=True, text=True,
                           timeout=30)
        _HOME_CACHE[host] = (p.stdout or "").strip() or "~"
    return _HOME_CACHE[host]


def expand(host: str, path: str) -> str:
    """`~/x` -> `/home/user/x`, leaving everything else alone."""
    if path.startswith("~/"):
        return home(host) + path[1:]
    return path


def have(host: str, path: str) -> bool:
    """True if `path` exists on `host`."""
    try:
        run(host, f"test -e {shlex.quote(expand(host, path))}",
            check=True, quiet=True)
        return True
    except (RemoteError, subprocess.TimeoutExpired):
        return False


def which_python(host: str, venv: str) -> str | None:
    """Path to a venv interpreter on `host`, or None if absent."""
    p = posixpath.join(expand(host, venv), "bin", "python")
    return p if have(host, p) else None
