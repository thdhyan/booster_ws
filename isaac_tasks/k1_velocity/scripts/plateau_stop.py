"""Reward-plateau early stopping for a running RSL-RL log.

WHY
---
Every PPO run here is launched with a large ``max_iterations`` as a CAP, not a
target, and ``save_interval`` writes a checkpoint every 100 iterations. That means
the real question is never "did it finish" but "when has it stopped improving" --
and answering it by eye means tailing a log and eyeballing a number for hours.

This is a separate watchdog rather than a hook in ``train.py`` on purpose: it must
be able to stop a run that is already in flight (including one started before this
existed), and it must not be able to corrupt a training step by raising inside the
step loop. A wrong answer from this script costs one run; a wrong answer from inside
the runner costs the run.

THE TEST
--------
Plateau is judged on the MEAN reward of consecutive windows, not on a single
iteration. RSL-RL's per-iteration reward is extremely noisy -- an iteration where
one env falls early swings the mean far more than any policy improvement -- so a
naive ``reward[-1] <= reward[-2]`` triggers within tens of iterations and stops
every run.

A window is "improved" only if its mean beats the best previous window by
``min_delta``. Requiring ``patience`` consecutive non-improving windows is what
separates a plateau from a dip.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from statistics import mean

DEFAULT_KEY = "Reward/total"


def parse_series(path: str, key: str) -> list[float]:
    """All values of ``key: <float>`` in the log, in order, de-duplicated by position.

    RSL-RL prints the metrics table every ``log_interval`` iterations, so the same
    iteration can appear twice if the log is re-read; taking the values in file
    order and letting the caller window them is enough and avoids needing to parse
    the iteration counter alongside every value.
    """
    pat = re.compile(re.escape(key) + r"[:=]\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    out: list[float] = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out.append(float(m.group(1)))
    return out


def window_means(series: list[float], window: int) -> list[float]:
    """Non-overlapping consecutive means; the trailing partial window is dropped."""
    n = len(series) // window
    return [mean(series[i * window:(i + 1) * window]) for i in range(n)]


def is_plateau(
    series: list[float],
    window: int = 200,
    min_delta: float = 0.5,
    patience: int = 3,
) -> tuple[bool, str]:
    """True once ``patience`` consecutive windows failed to beat the best by ``min_delta``.

    Returns ``(plateaued, reason)``. Deliberately requires at least
    ``patience + 1`` windows, so a run cannot be declared flat before it has shown
    any improvement at all -- an early flat patch is normal PPO warmup.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    wm = window_means(series, window)
    if len(wm) < patience + 1:
        return False, f"only {len(wm)} window(s) of {window}; need {patience + 1}"
    best = wm[0]
    stale = 0
    for w in wm[1:]:
        if w > best + min_delta:
            best = w
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                return True, (
                    f"{stale} consecutive windows failed to beat {best:.3f} "
                    f"by {min_delta} (last={wm[-1]:.3f}, windows={len(wm)})"
                )
    return False, f"{len(wm)} windows, best {best:.3f}, stale {stale}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="training log to tail")
    ap.add_argument("--key", default=DEFAULT_KEY, help=f"metric key (default {DEFAULT_KEY})")
    ap.add_argument("--window", type=int, default=200, help="iterations per window")
    ap.add_argument("--min-delta", type=float, default=0.5,
                    help="min improvement in window-mean reward to count as progress")
    ap.add_argument("--patience", type=int, default=3,
                    help="consecutive non-improving windows before stopping")
    ap.add_argument("--tmux", default="", help="tmux session to kill on plateau (optional)")
    ap.add_argument("--marker", default="", help="file to touch when plateau is detected")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit")
    args = ap.parse_args()

    def check() -> int:
        if not os.path.exists(args.log):
            return 0
        series = parse_series(args.log, args.key)
        done, why = is_plateau(series, args.window, args.min_delta, args.patience)
        print(f"[plateau] n={len(series)} plateau={done} ({why})", flush=True)
        if done:
            if args.marker:
                with open(args.marker, "w") as fh:
                    fh.write(why + "\n")
            if args.tmux:
                subprocess_kill_tmux(args.tmux)
        return 1 if done else 0

    if args.once:
        return check()

    while True:
        if check():
            return 0
        time.sleep(args.poll)


def subprocess_kill_tmux(session: str) -> None:
    """Kill a tmux session without importing subprocess at module scope."""
    import subprocess

    subprocess.run(["tmux", "kill-session", "-t", session], check=False)


if __name__ == "__main__":
    sys.exit(main())