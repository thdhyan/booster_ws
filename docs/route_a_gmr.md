# Route A — Kimodo (SMPL-X) -> GMR -> Booster K1

Text-to-motion dance retargeted onto the K1 through GMR. This is the route that
was blocked; both blockers are now fixed and the chain runs end to end.

```
text prompt
  -> Kimodo-SMPLX              (aim_spark03)  -> 8 s @ 30 Hz, 77-joint SOMA BVH
  -> SMPL-X AMASS .npz                        -> pose_body (240, 63) = 21 joints
  -> GMR smplx_to_robot.py                    -> 239 frames x 22 DoF, 30 Hz
  -> scripts/gmr_pkl_to_k1_csv.py             -> Booster motion CSV (7 + 22)
  -> scripts/motion_feasibility_gate.py       -> verdict
  -> scripts/record_k1_motion.py              -> docs/videos/k1_macarena_gmr.mp4
```

## The two blockers, and what actually caused them

### 1. `einsum(): subscript l has size 20 ... previously seen size 26`

This was **not** a reduced-joint-set problem, which was the initial hypothesis.
Kimodo's `pose_body` is `(240, 63)` = 21 joints, which is standard full SMPL-X.

The real mismatch was in the shape space. The reported 20-vs-26 were torch's
broadcast sizes, not the literal dimensions; the actual conflict was **16 vs 10**:

| | value |
|---|---|
| `betas` in Kimodo's AMASS | `(16,)` — full SMPL-X shape space |
| `smplx.create(...)` default | `num_betas=10`, so `shapedirs` is `(10475, 3, 10)` |

which blows up at
`blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])`.

GMR hardcoded the default and never sized it from the data. Fix: derive it.

```python
_nb = int(np.asarray(smplx_data["betas"]).reshape(-1).shape[0])
body_model = smplx.create(..., num_betas=_nb, use_pca=False)
```

`patches/numbetas.patch` also carries the earlier `expression` batching fix
(unbatched `(10,)` cannot broadcast against `(num_frames, 3)` pose tensors).

### 2. `GLFWError: X11: The DISPLAY environment variable is missing`

`RobotMotionViewer.__init__` called `mjv.launch_passive` unconditionally, so GMR
could not run headless at all — on a server, in CI, or over SSH. `mj_step` on
`MjModel`/`MjData` already happened before the window was opened, so the
kinematics driving the retargeting do not depend on the window; only the
on-screen viewer did. Fixed by adding a `headless` flag that skips
`launch_passive` and guards the five downstream uses (`opt.flags`, `cam.*`,
`user_scn`, `sync`, `close`), with the camera passed as `None` to the offscreen
renderer so `--record_video` still works. `patches/headless.patch`.

## Note on the converter

`scripts/gmr_pkl_to_k1_csv.py` is a thin `hstack`, which is only safe because two
things were **verified rather than assumed**:

1. `smplx_to_robot.py:162` writes
   `root_rot = np.array([qpos[3:7][[1,2,3,0]] ...])`, reordering MuJoCo's
   scalar-first `wxyz` into `xyzw`. Booster wants `xyzw`, so the saved `root_rot`
   is already correct — the GMRResult `wxyz` note does not apply to this path.
2. GMR's `assets/booster_k1/K1_serial.xml` declares its 22 hinge joints in
   exactly `K1_JOINT_NAMES` order, so `dof_pos` is already in CSV order.

The script re-checks (2) against the XML at runtime and exits non-zero on a
mismatch, because a silent joint reordering would still render as plausible
motion and would be nearly impossible to notice by eye.

## Result — `assets/motions/macarena_k1.csv`

239 frames @ 30 Hz = 7.97 s, 29 columns, root quats unit-norm to 1.1e-15.

Feasibility gate: **FAIL, 5 blocking issues.**

| check | verdict |
|---|---|
| joint_limits | ok — all 22 within URDF range |
| root_height | ok — 0.459–0.507 m (nominal 0.57) |
| double_support | ok — both-feet 100% (standing dance) |
| com_margin | ok — always double support |
| foot_float | ok — never airborne |
| joint_velocity | **FAIL** — `Right_Shoulder_Roll` 44.09 rad/s = 2.45x limit |
| joint_accel | **FAIL** — `Right_Elbow_Pitch` 2387.7 rad/s² = 29.85x limit |
| torque_estimate | **FAIL** — `AAHead_yaw` ~57.5 Nm = 9.58x the 6 Nm limit |
| foot_penetration | **FAIL** — right sole 91.0 mm below floor at frame 125 |
| foot_slip | **FAIL** — left foot slides 0.695 m/s while planted |

These are the expected consequences of GMR being a pure kinematic IK solver: it
has no foot locking, no dynamics, and no actuator model, so a fast dance comes
out fast, and the feet skate and sink. The head actuation overshoot is
partly an artefact of `AAHead_yaw` carrying a big fast excursion that the 6 Nm
limit cannot hold.

## The replay video is dynamic, not kinematic

`scripts/record_k1_motion.py` does not simply pose the model. Per rendered frame
it runs `args.sim_fps // args.fps` (= 6) `mj_step`s at `dt = 1/200 s` while a PD
controller pulls the joints toward the reference — so the 30 Hz frames are
correctly timed at 1/30 s each, and the header numbers are real measurements:

- `q_err` = `max|qpos - q_des|` — genuine PD tracking error under dynamics
  (0.303 rad at t=2.0 s, 0.387 rad at t=4.2 s)
- `root_z` = the *simulated* root, which exceeds the commanded CSV maximum of
  0.507 m because the PD pushes the robot off the reference

A caveat on the earlier `q_err` comparison: the SOMA/T1 Macarena was a
different robot (T1, 29 DoF), a different source (SOMA BVH, not SMPL-X), and was
not measured with identical recorder settings, so 0.387 vs 0.872 rad is not a
controlled comparison. It is suggestive, not evidence.

## Reproduce

```bash
# on aim_spark03
cd ~/Projects/GMR
source ../.venv-gmr/bin/activate
export PYTHONPATH=.
python scripts/smplx_to_robot.py --smplx_file macarena_amass.npz \
    --robot booster_k1 --headless --save_path out/mac_k1.pkl
python gmr_pkl_to_k1_csv.py --in out/mac_k1.npz --out out/macarena_k1.csv \
    --xml assets/booster_k1/K1_serial.xml
```

`--robot booster_k1` does work, and `ik_configs/smplx_to_k1.js` does exist —
only `scripts/bvh_to_robot.py` omits `booster_k1` from its `--robot` choices.

## Still open

The gate is the authority on feasibility and it says this is not hardware-ready.
The path from here is not a retarget fix but the RL tracking-policy stage, plus
foot-plant post-processing — which is what the SOMA route's converter already
does via `--ground-feet`.
