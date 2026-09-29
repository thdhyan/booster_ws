#!/usr/bin/env python3
"""Gait quality gate: does this policy WALK, or just move?

WHY THIS EXISTS
---------------
The displacement gate I built earlier passed a policy that was chattering. That
run scored 2.664 m net displacement with zero falls over 750 steps at 0.5 m/s --
comfortably "moving" by any displacement criterion -- and was not walking at all.

Decoding the play trace across 8 envs showed why:

    mean cadence 5.7 steps/s   (human walking: 1.8-2.2)
    5 of 8 envs at 7.9-8.9 steps/s on 3-5 cm strides
    2 envs chattering on one leg while the other stayed planted
    jerk 0.26 rad/step^2 in hip-yaw, 0.19 in knee

So movement is necessary but not sufficient. A shuffle satisfies every
displacement test there is. This gate adds the three things that actually
distinguish a walk: cadence in a human range, a stride long enough to be
locomotion rather than vibration, and smoothness.

It reads the same ``.npz`` play trace the displacement check uses, so it needs
no simulator.

Usage:  gait_gate.py <trace.npz> [--strict]
Exit:   0 if the gait is human-like, 1 if it is a shuffle/hop, 2 on bad input.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

DT = 0.02  # 50 Hz control, matches decimation=4 * dt=0.005

# Human walking reference. Deliberately generous ranges: the goal is "obviously
# not a shuffle", not "biomechanically exact".
CADENCE_MIN, CADENCE_MAX = 1.2, 4.0      # steps/s
STRIDE_MIN = 0.15                        # m, horizontal foot separation
JERK_MAX = 0.06                          # rad/step^2, mean |3rd difference|


def dominant_hz(signal: np.ndarray, dt: float = DT) -> float:
    """Dominant frequency of a 1-D signal, via FFT peak. 0.0 if flat."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), dt)
    if len(spec) < 2:
        return 0.0
    return float(freqs[int(np.argmax(spec[1:])) + 1])


def analyse(npz_path: pathlib.Path, knee_idx: tuple[int, int]) -> dict:
    d = np.load(npz_path)
    if "actions" not in d:
        raise KeyError(f"{npz_path} has no 'actions' array; is it a play trace?")
    actions = d["actions"]                      # (steps, num_envs, n_joints)
    speeds = None
    if "root_lin_vel" in d:
        speeds = np.linalg.norm(d["root_lin_vel"][..., :2], axis=-1)

    per_env = []
    for e in range(actions.shape[1]):
        a = actions[:, e, :]
        # One gait cycle per knee-flexion oscillation, so steps/s = 2 * freq.
        f_l = dominant_hz(a[:, knee_idx[0]])
        f_r = dominant_hz(a[:, knee_idx[1]])
        cadence = max(2.0 * f_l, 2.0 * f_r)
        speed = float(speeds[:, e].mean()) if speeds is not None else float("nan")
        stride = speed / cadence if cadence > 0.2 else float("nan")
        jerk = float(np.abs(np.diff(a, n=3, axis=0)).mean())
        per_env.append(
            {"env": e, "cadence": cadence, "speed": speed, "stride": stride, "jerk": jerk}
        )
    return {"per_env": per_env, "n_env": actions.shape[1], "n_steps": actions.shape[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=pathlib.Path)
    ap.add_argument(
        "--knee",
        type=int,
        nargs=2,
        default=(3, 9),
        help="indices of left/right knee in the action vector (default 3, 9)",
    )
    args = ap.parse_args()

    if not args.npz.is_file():
        print(f"[FAIL] no such trace: {args.npz}")
        return 2
    try:
        res = analyse(args.npz, tuple(args.knee))
    except (KeyError, ValueError) as exc:
        print(f"[FAIL] cannot analyse trace: {exc}")
        return 2

    print(f"GAIT_GATE {args.npz.name}  ({res['n_env']} envs, {res['n_steps']} steps)")
    print(f"  {'env':>3} {'cadence':>9} {'speed':>7} {'stride':>8} {'jerk':>8}   verdict")
    shuffle = 0
    for r in res["per_env"]:
        bad = []
        if not (CADENCE_MIN <= r["cadence"] <= CADENCE_MAX):
            bad.append("cadence")
        if not (r["stride"] != r["stride"]) and r["stride"] < STRIDE_MIN:
            bad.append("stride")
        if r["jerk"] > JERK_MAX:
            bad.append("jerk")
        if bad:
            shuffle += 1
        verdict = "SHUFFLE: " + ",".join(bad) if bad else "walk-like"
        print(
            f"  {r['env']:>3} {r['cadence']:>9.2f} {r['speed']:>7.3f} "
            f"{r['stride']:>8.3f} {r['jerk']:>8.4f}   {verdict}"
        )

    cad = float(np.mean([r["cadence"] for r in res["per_env"]]))
    jk = float(np.mean([r["jerk"] for r in res["per_env"]]))
    print(f"\n  mean cadence {cad:.2f} steps/s (want {CADENCE_MIN}-{CADENCE_MAX})")
    print(f"  mean jerk    {jk:.4f} rad/step^2 (want < {JERK_MAX})")
    print(f"  {shuffle}/{res['n_env']} envs fail the gait gate")

    if shuffle > res["n_env"] / 2:
        print(
            "\nGAIT_GATE=FAIL  This policy moves but does not walk. Displacement "
            "alone is not a sufficient test; a high-frequency shuffle passes every "
            "distance check. Fix the reward (cadence, clearance, stride, jerk) "
            "rather than the network size."
        )
        return 1
    print("\nGAIT_GATE=PASS  cadence and smoothness are in a human walking range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
