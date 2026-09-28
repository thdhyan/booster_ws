"""k1m -- turn a video or a text prompt into Booster K1 joint angles.

    k1m video dance.mp4                  video  -> CSV + replay
    k1m text "a person dances"           text   -> CSV + replay
    k1m csv some_motion.csv              existing retarget, no GPU
    k1m npz smplx_amass.npz              retarget a known SMPL-X file
    k1m doctor                           which remote stages are usable
    k1m joints out/foo_k1.csv            print the joint table

The heavy stages (GVHMR, Kimodo, GMR) run on a DGX Spark over ssh because the
laptop is a client only; the gate and the MuJoCo replay run locally.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import local_stages, pipeline, remote_stages, schema


def _print_result(r: pipeline.Result) -> int:
    print(f"\n{'=' * 70}")
    print(f"joint angles : {r.motion_csv}")
    if r.gate_json:
        print(f"gate report  : {r.gate_json}")
    if r.replay_mp4:
        print(f"replay video : {r.replay_mp4}")
    print(f"frames       : {r.n_frames} @ {r.fps:g} Hz "
          f"({r.n_frames / r.fps:.2f}s)")
    if r.gate_summary:
        print(f"\n{r.gate_summary}")
    for w in r.warnings:
        print(f"  note: {w}")
    if r.gate_json and not r.gate_passed:
        print("\nNOT SAFE FOR HARDWARE. This is a kinematic retarget with no "
              "foot locking,\nno dynamics and no actuator model. Getting it "
              "onto a real K1 needs the RL\ntracking-policy stage plus "
              "foot-plant post-processing.")
    print("=" * 70)
    return 0 if (not r.gate_json or r.gate_passed) else 1


def cmd_video(a) -> int:
    r = pipeline.from_video(a.video, outdir=a.out, stem=a.name or "",
                            fps=a.fps, do_gate=not a.no_gate,
                            do_render=not a.no_render, render_gl=a.gl,
                            max_seconds=a.max_seconds)
    return _print_result(r)


def cmd_text(a) -> int:
    r = pipeline.from_text(a.prompt, outdir=a.out, stem=a.name or "text",
                           seconds=a.seconds, seed=a.seed, fps=a.fps,
                           do_gate=not a.no_gate, do_render=not a.no_render,
                           render_gl=a.gl)
    return _print_result(r)


def cmd_npz(a) -> int:
    r = pipeline.from_smplx(a.npz, outdir=a.out, stem=a.name or "",
                            fps=a.fps, do_gate=not a.no_gate,
                            do_render=not a.no_render, render_gl=a.gl)
    return _print_result(r)


def cmd_csv(a) -> int:
    r = pipeline.from_csv(a.csv, outdir=a.out, fps=a.fps,
                          do_gate=not a.no_gate, do_render=not a.no_render,
                          render_gl=a.gl)
    return _print_result(r)


def cmd_doctor(a) -> int:
    print("K1 joint order (CSV contract):")
    for i in range(0, schema.N_JOINTS, 2):
        row = "  ".join(f"{j:2d} {schema.K1_JOINT_NAMES[j]:<24}"
                        for j in (i, i + 1) if j < schema.N_JOINTS)
        print(f"  {row}")
    print(f"\nCSV: {schema.N_COLS} cols = 3 pos + 4 quat(xyzw) + "
          f"{schema.N_JOINTS} joints")
    try:
        schema.verify_joint_order()
        print("joint order vs K1 MJCF: exact match")
    except Exception as e:                            # noqa: BLE001
        print(f"joint order check FAILED: {e}")
    try:
        lim = schema.joint_limits_from_urdf()
        print(f"URDF limits: {len(lim)} joints parsed")
    except Exception as e:                            # noqa: BLE001
        print(f"URDF limit parse failed: {e}")

    print("\nremote stages (heavy work runs here, laptop is client only):")
    for line in remote_stages.preflight():
        print(f"  {line}")
    return 0


def cmd_joints(a) -> int:
    m, fps = schema.read_csv(a.csv)
    if a.frame >= len(m):
        print(f"frame {a.frame} out of range (0..{len(m) - 1})", file=sys.stderr)
        return 1
    row = m[a.frame]
    print(f"# {a.csv}  frame {a.frame}/{len(m)}  @{fps:g} Hz")
    print(f"# root pos [m]  {row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}")
    print(f"# root quat xyzw {row[3]:+.5f} {row[4]:+.5f} {row[5]:+.5f} {row[6]:+.5f}")
    print(f"# {'joint':<26} {'q [rad]':>10} {'deg':>9}   urdf range [rad]")
    for i, name in enumerate(schema.K1_JOINT_NAMES):
        q = row[7 + i]
        rng = ""
        try:
            lo, hi = schema.joint_limits_from_urdf()[name]
            rng = f"[{lo:+.3f}, {hi:+.3f}]"
        except Exception:                             # noqa: BLE001
            pass
        print(f"  {name:<26} {q:>+10.4f} {np_deg(q):>+9.2f}   {rng}")
    return 0


def np_deg(r: float) -> float:
    return r * 180.0 / 3.141592653589793


def _add_common(p, suppress: bool = False) -> None:
    """Shared options, added to the top-level parser and to every subcommand.

    Suppressed defaults let the flags work in either position
    (`k1m --fps 50 video x.mp4` and `k1m video x.mp4 --fps 50`) without the
    subparser's default clobbering a value given before the subcommand.
    """
    d = argparse.SUPPRESS if suppress else None
    p.add_argument("--out", default="k1m_out" if not suppress else d,
                   help="output directory")
    p.add_argument("--fps", type=float, default=30.0 if not suppress else d,
                   help="motion rate")
    p.add_argument("--gl", default="egl" if not suppress else d,
                   choices=["glfw", "egl", "osmesa"])
    p.add_argument("--no-gate", action="store_true",
                   default=False if not suppress else d,
                   help="skip the feasibility gate (NOT hardware safe)")
    p.add_argument("--no-render", action="store_true",
                   default=False if not suppress else d,
                   help="skip the replay video")
    p.add_argument("--name", default="" if not suppress else d,
                   help="output basename")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="k1m", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    def mk(name: str, help_: str):
        sp = sub.add_parser(name, help=help_)
        _add_common(sp, suppress=True)
        return sp

    v = mk("video", "video -> K1 motion")
    v.add_argument("video")
    v.add_argument("--max-seconds", type=float, default=0.0)
    v.set_defaults(fn=cmd_video)

    t = mk("text", "text prompt -> K1 motion")
    t.add_argument("prompt")
    t.add_argument("--seconds", type=float, default=8.0)
    t.add_argument("--seed", type=int, default=7)
    t.set_defaults(fn=cmd_text)

    n = mk("npz", "SMPL-X AMASS npz -> K1 motion")
    n.add_argument("npz")
    n.set_defaults(fn=cmd_npz)

    c = mk("csv", "existing K1 motion CSV -> gate + replay")
    c.add_argument("csv")
    c.set_defaults(fn=cmd_csv)

    d = mk("doctor", "check the environment")
    d.set_defaults(fn=cmd_doctor)

    j = mk("joints", "print one frame's joint angles")
    j.add_argument("csv")
    j.add_argument("--frame", type=int, default=0)
    j.set_defaults(fn=cmd_joints)

    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
