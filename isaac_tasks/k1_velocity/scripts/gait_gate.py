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

# Swing/clearance, measured from the foot arrays play_record.py now records.
# feet_clearance's own target_height; kept here so the gate and the reward agree.
SWING_FOOT_HEIGHT = 0.06                 # m, a swing foot should clear this
# A trace without these arrays predates the change; the gate degrades to the
# action-derived cadence rather than refusing to run, because old traces are the
# only record we have of the runs that failed.
FOOT_KEYS = ("foot_pos", "foot_contact_n")


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


def foot_swing_stats(foot_pos: np.ndarray, contact_n: np.ndarray,
                     env: int, max_n: float) -> dict:
    """Measured swing/clearance for one env, straight from the foot arrays.

    ``foot_pos`` is (steps, n_feet, 3) world-frame and ``contact_n`` is
    (steps, n_feet) contact-sensor force norm. A foot is *swinging* when its
    force is at or below ``max_n`` -- the same predicate ``feet_clearance`` uses,
    so the two agree by construction.

    This exists because the action-derived metrics cannot answer the question.
    ``cadence`` is the FFT peak of the knee ACTION, so a policy that oscillates
    its knees while both feet stay planted scores a healthy cadence and a
    "stride" of a few millimetres. Reading the feet removes the ambiguity.
    """
    z = foot_pos[:, env, :, 2]                       # (steps, n_feet) height
    swing = contact_n[:, env, :] <= max_n            # (steps, n_feet)
    has_swing = swing.any(axis=1)                    # (steps,)
    n_swing_feet = swing.sum(axis=1)

    swing_z = z[swing]
    clearance = swing_z - SWING_FOOT_HEIGHT if swing_z.size else np.array([])
    # Step events: a rising edge into swing. Counting both feet gives steps/s.
    rising = np.zeros(n_swing_feet.shape[0], dtype=bool)
    rising[1:] = (n_swing_feet[1:] > n_swing_feet[:-1])
    step_hz = float(rising.sum()) / max(len(n_swing_feet) * DT, 1e-9)

    # Fore-aft excursion of whichever foot swings most, in the body frame would
    # be better but the heading is not in the trace; magnitude is still a measure
    # of step length that does not divide by cadence.
    horiz = np.linalg.norm(np.diff(foot_pos[:, env, :, :2], axis=0), axis=-1)  # (steps-1, n_feet)
    swing_horiz = horiz[swing[1:]] if horiz.shape[0] == swing.shape[0] - 1 else horiz[:]

    return {
        "swing_frac": float(has_swing.mean()),
        "step_hz": step_hz,
        "swing_clear": float(clearance.mean()) if clearance.size else float("nan"),
        "swing_h_max": float(swing_z.max()) if swing_z.size else float("nan"),
        "swing_h_mean": float(swing_z.mean()) if swing_z.size else float("nan"),
        "swing_horiz": float(swing_horiz.mean()) if swing_horiz.size else float("nan"),
        "min_foot_z": float(z.min()),
    }


def analyse(npz_path: pathlib.Path, knee_idx: tuple[int, int]) -> dict:
    d = np.load(npz_path)
    if "actions" not in d:
        raise KeyError(f"{npz_path} has no 'actions' array; is it a play trace?")
    actions = d["actions"]                      # (steps, num_envs, n_joints)
    speeds = None
    if "root_lin_vel" in d:
        speeds = np.linalg.norm(d["root_lin_vel"][..., :2], axis=-1)
    has_feet = all(k in d.files for k in FOOT_KEYS)
    foot_pos = d["foot_pos"] if has_feet else None
    contact_n = d["foot_contact_n"] if has_feet else None
    max_n = float(d["foot_contact_max_n"]) if has_feet and "foot_contact_max_n" in d.files else 1.0

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
        row = {"env": e, "cadence": cadence, "speed": speed, "stride": stride, "jerk": jerk}
        if has_feet:
            row.update(foot_swing_stats(foot_pos, contact_n, e, max_n))
        per_env.append(row)
    return {"per_env": per_env, "n_env": actions.shape[1], "n_steps": actions.shape[0],
            "has_feet": has_feet}


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
    if res["has_feet"]:
        print(f"  {'env':>3} {'cadence':>9} {'speed':>7} {'v/cad':>8} {'jerk':>8} "
              f"{'swing%':>7} {'step_hz':>8} {'sw_clear':>9}   verdict")
    else:
        print(f"  {'env':>3} {'cadence':>9} {'speed':>7} {'v/cad':>8} {'jerk':>8}   verdict")
        print("  [note] trace predates foot recording: cadence is the knee-ACTION "
              "FFT and v/cad is speed/cadence, so neither measures step length")
    shuffle = 0
    for r in res["per_env"]:
        bad = []
        if not (CADENCE_MIN <= r["cadence"] <= CADENCE_MAX):
            bad.append("cadence")
        if not (r["stride"] != r["stride"]) and r["stride"] < STRIDE_MIN:
            bad.append("stride")
        if r["jerk"] > JERK_MAX:
            bad.append("jerk")
        if res["has_feet"] and r.get("swing_frac", 0.0) <= 0.0:
            # The decisive one: no foot ever left the floor, so nothing here can
            # be a walk regardless of what the action-derived columns say.
            bad.append("no-swing")
        if bad:
            shuffle += 1
        verdict = "SHUFFLE: " + ",".join(bad) if bad else "walk-like"
        if res["has_feet"]:
            print(
                f"  {r['env']:>3} {r['cadence']:>9.2f} {r['speed']:>7.3f} "
                f"{r['stride']:>8.3f} {r['jerk']:>8.4f} {100*r['swing_frac']:>6.1f}% "
                f"{r['step_hz']:>8.2f} {r['swing_clear']:>+9.3f}   {verdict}"
            )
        else:
            print(
                f"  {r['env']:>3} {r['cadence']:>9.2f} {r['speed']:>7.3f} "
                f"{r['stride']:>8.3f} {r['jerk']:>8.4f}   {verdict}"
            )

    cad = float(np.mean([r["cadence"] for r in res["per_env"]]))
    jk = float(np.mean([r["jerk"] for r in res["per_env"]]))
    print(f"\n  mean cadence {cad:.2f} steps/s (want {CADENCE_MIN}-{CADENCE_MAX})")
    print(f"  mean jerk    {jk:.4f} rad/step^2 (want < {JERK_MAX})")
    if res["has_feet"]:
        sw = float(np.mean([r["swing_frac"] for r in res["per_env"]]))
        sh = float(np.mean([r["step_hz"] for r in res["per_env"]]))
        sc = float(np.nanmean([r["swing_clear"] for r in res["per_env"]]))
        print(f"  MEASURED swing  {100*sw:.1f}% of steps have a foot off the floor "
              f"(0% means the feet never leave)")
        print(f"  MEASURED steps  {sh:.2f} steps/s from contact rising edges")
        print(f"  MEASURED clear  {sc:+.3f} m vs the {SWING_FOOT_HEIGHT} m target "
              f"(negative = swinging feet still too low)")
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
