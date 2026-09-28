# k1m — video or text in, Booster K1 joint angles out

One package that wraps the whole retargeting chain: human motion → SMPL-X →
Booster K1 joint angles, gated for physical feasibility and replayed in MuJoCo.

```
video ─► GVHMR ─► SMPL-X ─┐
                          ├─► GMR ─► K1 CSV ─► feasibility gate ─► MuJoCo replay
text  ─► Kimodo ──────────┘
```

## Install

```bash
cd ~/Projects/booster_ws
uv venv --python 3.12 ~/.venvs/k1m
uv pip install --python ~/.venvs/k1m/bin/python \
    mujoco 'numpy<2' scipy imageio imageio-ffmpeg pillow gradio
~/.venvs/k1m/bin/pip install -e .     # or just run from the repo root
```

## Use

```bash
k1m doctor                              # which stages are usable
k1m video dance.mp4                     # video  → CSV + replay
k1m text "a person dances the Macarena"  # text   → CSV + replay
k1m csv existing_k1.csv                 # already retargeted, no GPU
k1m npz smplx_amass.npz                 # a known SMPL-X file
k1m joints out/dance_k1.csv --frame 125 # one frame's angles vs URDF limits
```

Web UI:

```bash
~/.venvs/k1m/bin/python app.py          # http://127.0.0.1:7860
```

Common flags work before or after the subcommand, and both mean the same thing:

```bash
k1m --fps 50 --out runs/ video dance.mp4
k1m video dance.mp4 --fps 50 --out runs/
```

`k1m <cmd>` exits non-zero when the gate fails, so it drops into CI or a
pipeline as a real check rather than a report you have to read.

## Where the work runs

The laptop is a client only. The models run on a DGX Spark over ssh:

| stage | host | venv | what it does |
|---|---|---|---|
| `gvhrm` | `aim_spark02` | `~/Projects/.venv-gvhrm` | video → SMPL-X, world-frame |
| `kimodo` | `aim_spark03` | `~/Projects/kimodo_ws/kimodo/.venv` | text → SMPL-X |
| `gmr` | `aim_spark03` | `~/Projects/.venv-gmr` | SMPL-X → K1 CSV (mink + MuJoCo IK) |
| gate | local | — | 10 physics checks vs the K1 URDF |
| replay | local | — | 3-panel MuJoCo PD replay |

`k1m doctor` checks each venv *and* that its modules import, so a half-finished
install reports as broken instead of as ready.

## The gate is the safety signal, not the render

Every path ends at `scripts/motion_feasibility_gate.py`, which checks joint
limits, velocity, acceleration, torque, foot penetration, foot float, foot slip,
root height, double support and CoM margin against the real K1 URDF.

This matters because the retargeters are **kinematic IK solvers with no foot
locking, no dynamics and no actuator model**. A clean-looking render is not
evidence a robot can track the motion. Our Macarena render, for example, is
visually fine and simultaneously:

- asks `AAHead_yaw` for **9.58×** its effort limit
- reaches **2.45×** the joint velocity limit on `Right_Shoulder_Roll`
- puts a sole **91 mm** through the floor
- slides a planted foot at **0.695 m/s**

and it passes `joint_limits` while pinning **6 joints exactly at their limits** —
saturation, not compliance. The web UI flags those joints explicitly.

Getting this onto a real K1 needs the RL tracking-policy stage plus foot-plant
post-processing, not a better retarget. See `docs/robocup_gmr_research.md` and
`docs/video_to_motion_plan.md`.

## Status

| path | state |
|---|---|
| `k1m csv` | works, no GPU — gate + replay verified |
| `k1m npz` | works — GMR on `aim_spark03`, numbers match the wt-smplx run exactly |
| `k1m text` | wired; Kimodo's real CLI is `python -m kimodo.scripts.generate "<prompt>" --model Kimodo-SMPLX-RP-v1 --output STEM` |
| `k1m video` | **blocked** — GVHMR needs `pytorch3d`, which has no aarch64 wheel and is building from source |

GVHMR on a GB10 needed more than a plain install: its `requirements.txt` pins
`torch==2.3.0+cu121` (no aarch64 wheel), `chumpy` breaks on numpy >= 1.24
(`np.int` removal), and `detectron2` refuses to build unless `CUDA_HOME`'s
toolkit matches `torch.version.cuda`. Current state on `aim_spark02`: torch
2.14.0+cu130, 17/18 modules import, detectron2 0.6 built. `pytorch3d` is the
last one.

Note GVHMR does not export AMASS — `tools/demo/demo.py` writes a `pred` torch
dict — so `k1m/gvhmr_amass_export.py` reshapes it into the AMASS layout GMR
consumes. Its `body_pose` is `(F, 63)`, i.e. 21 joints, which already matches
AMASS `pose_body`.

## Layout

```
k1m/
  schema.py        the K1 CSV contract: 3 pos + 4 quat(xyzw) + 22 joints
  remote.py        ssh / scp helpers
  remote_stages.py gvhrm, kimodo, gmr  (on a spark)
  local_stages.py  gate + replay       (local)
  pipeline.py      orchestration, one function per input type
  cli.py           k1m <video|text|csv|npz|doctor|joints>
app.py             Gradio front end
```

The gate and recorder are invoked as subprocesses rather than reimplemented.
Both have accumulated fixes — MJCF geom groups, quaternion order, the K1 sole
frame offset — that would be easy to lose by forking them into a library.

## Two conventions that are easy to get wrong

**Root quaternion is xyzw**, not wxyz. Booster wants xyzw. GMR's
`smplx_to_robot.py:162` already reorders its MuJoCo `wxyz` qpos, but other GMR
entry points (`GMRResult`) publish wxyz, so it is path-dependent.

**Joint order is fixed by the robot.** GMR's `K1_serial.xml` declares its 22
hinges in exactly `K1_JOINT_NAMES` order, so no reordering is needed — but a
producer that *did* need it would yield motion that still looks plausible in a
render. `schema.verify_joint_order()` re-checks it against the MJCF at runtime
and fails loudly, and `k1m doctor` runs it.

## fps

The motion CSV has no header and no rate field, so `k1m` writes a
`<name>.fps.txt` sidecar and reads it back. Get this wrong and every velocity
and acceleration number is wrong by a constant factor — squared, for
acceleration — so the gate also takes `--fps` explicitly.
