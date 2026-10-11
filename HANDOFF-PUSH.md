# HANDOFF — PUSH (box-pushing) workstream on `zz-bw`

**For:** a fresh agent or a new session picking this up cold.
**Scope:** the K1 box-push task only. The velocity/gait/soccer workstreams are tracked in
`HANDOFF.md`, `TRAINING.md`, `TASKS.md` and `STATE.md` — those are stale (Sept–Oct 6) and
are **not** updated by this workstream. Do not edit them from here.
**Repo:** `/home/thakk100/Projects/booster_ws`, branch `main`, GitHub `thdhyan/booster_ws`.
**Host:** `zz-bw` (`cs-zhang-net-01`) — 2× RTX PRO 6000 Blackwell, 97 GB each.
**Last commit this doc describes:** `6834137` (2026-10-10).

---

## 0. TL;DR — where things actually are

| Item | State |
|---|---|
| Squat **teacher** | ✅ trained, `model_8100.pt`, success 0.83 |
| Squat **student** (distill) | ✅ **finished all 12000 iterations** → `model_11999.pt` |
| Squat teacher TorchScript (push base) | ✅ `k1_squat_base.pt`, 1.17 MB, verified 238→12 |
| Push env — **builds** | ✅ verified in-sim 2026-10-10 (`6834137`), 4096 envs, 3.6 s/iter |
| Push env video (untrained actor) | ✅ `logs/push_zero/push_env_zero.mp4`, 500/500 frames decodable; base stands (trunk z 0.45–0.57) |
| Reach **training** | ⏹ stopped 2026-10-10 at user request (~100 it). Push-SG already contains the approach terms, so a separate Reach stage is redundant |
| Push-SG **training** | ▶️ running since 2026-10-10 15:54, tmux `k1_push` (+ watchdog `k1_watch`), log `$S/push_sg.log`, run `p6_push/2026-10-10_15-54-48_p6_push`; box 0.67–0.76 m, 3–10 kg; reward 45.9 at it 577/6000. it300 play video on Drive (§12) |
| Carry task (P7) | ▶️ **training** since 2026-10-10 16:36, tmux `k1_carry`, log `$S/carry.log`, run `p7_carry/2026-10-10_16-36-26_p7_carry`, 4096 envs, ~4.2 s/it (~7 h for 6000); it 203: reward 122, 1.7 goal successes/episode, 0 weld breaks. Smoke + debug video passed after the box-scale fix `aad6079` — see §11/§12 |
| Push stage / push student | ❌ not started (needs Reach first; no push student task is registered) |
| Tests | 329 passed, 1 pre-existing failure (`test_deployability`) |

**The single most important thing to know:** no push policy has ever been trained. The
environment has been blocked on one bug for most of this session. Everything below the
"current blocker" section is preparation that is done and committed.

---

## 1. What the task actually is

The K1 humanoid must walk up to a box and **push it with its arms** to a goal position,
using its legs only for locomotion and balance. It is deliberately **not** a base-push /
lean-into-the-box task — the user chose arm contact explicitly.

Two stages, trained in order:

1. **Reach** — get both wrists onto the box. No box motion is rewarded yet.
2. **Push** — move the box to the goal.

### Scene

| Element | Value | Why |
|---|---|---|
| `replicate_physics` | `False` | per-env USD domain randomisation requires it |
| env spacing | **8.0 m** | boxes must not interpenetrate across cells |
| ground | `/World/ground`, world plane | backstop only |
| `ground_patch` | per-env, `AssetBaseCfg` | **each cell gets its own friction** |
| friction range | **0.7 – 1.2** | see the warning below — do not lower it |
| box | `RigidObjectCfg` cube, respawned in an annulus r ∈ [0.9, 1.4] m | |
| `height_scanner` | `RayCasterCfg` on the trunk | base height |

### Action — 10-dim

```
[ vx, vy, wz, H* ]  frozen-base velocity command   (4)
[ wrist_left dx,dy,dz ]                            (3)
[ wrist_right dx,dy,dz ]                           (3)
```

The first 4 are **not** joint commands. They are fed to a **frozen** TorchScript base
policy (`FrozenBaseVelocityAction`), which outputs the 12 joint targets. `H*` is the
commanded trunk height and is what makes the squat base's observation 238-dim rather than
237. Both wrists are driven by differential IK, max 5 cm per step.

**The base policy is frozen.** It is never trained by the push run. That is why a squat
teacher is an acceptable base — it does not need to be a locomotion expert, only stable.

### Observation

One privileged group, `teacher` (`concatenate_terms=True`, corruption on during training).
Both actor and critic read it: `obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}`.

### Termination / success

Success is **not** a single frame. Mean error over the box's 8 corners must stay under
`SUCCESS_ERR_M = 0.08` for `SUCCESS_HOLD_S = 1.0` s. The `parked_bonus` reward mirrors
this same `goal_reached` signal — it is deliberately *not* the reference repo's
`box_inside_and_parked`, which does not exist here.

### Reward sets — two registered pairs

| Task | Reward set |
|---|---|
| `Isaac-Push-Reach-K1-v0` | our own terms |
| `Isaac-Push-K1-v0` | our own terms |
| `Isaac-Push-Reach-SG-K1-v0` | NVIDIA-AGILE-parity SG terms |
| `Isaac-Push-SG-K1-v0` | NVIDIA-AGILE-parity SG terms |

The SG pair is a **separate task**, not an overwrite, so the two reward sets can be A/B'd
against the same frozen base. SG terms live in `push_rewards_sg.py` (7 terms, ported,
arms-adapted, self-contained). Weights are in `push_env_cfg.py`; note the sign convention —
functions return **negative** distance, so **positive** weight = penalty. An earlier
version had negative weights, which double-negated into a *reward* for error.

---

## 2. ⚠️ Do not lower the ground friction floor

The floor is **0.7**, not the reference repo's 0.3. This is measured, not stylistic:
at 0.5 the **frozen base falls over** (`DIAG_FAILED_BASE_STILL_FALLS`, `done_rate=0.0425`).
The base cannot adapt to a slick floor because it is frozen. Lowering it will look like a
small config tweak and will silently destroy every run.

---

## 3. How to run things on `zz-bw`

### The launcher

Everything runs through `~/run_k1_train.sh`, which enters a SingularityCE container.
**Never run Isaac directly** — the container provides the GPU runtime and the bind mounts.

```bash
K1_TRAIN_SCRIPT=<script> K1_GPU=<n> ~/run_k1_train.sh <args...>
```

| Variable | Meaning |
|---|---|
| `K1_TRAIN_SCRIPT` | script name, **resolved from the container workdir root** |
| `K1_GPU` | GPU index; defaults to 1 |
| `PUSH_BASE_MODE` | `squat` selects the frozen squat base |

`K1_TRAIN_SCRIPT` is resolved from the workdir root. **Do not symlink `train.py`** —
through the symlink `sys.path[0]` becomes the clone root and the run dies with
`No module named 'cli_args'`. For training use `K1_TRAIN_SCRIPT=run_unbuffered.py` (an
untracked wrapper in the clone root that execs the real `scripts/train.py`). Other scripts
can be given by full path, e.g. `K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/play_record.py`.

`--containall` strips host env: `PUSH_BASE_MODE` is **not** passed into the container
(squat is the default anyway).

### Paths

| Host | Inside container |
|---|---|
| `/export/scratch/thakk100/k1` | — (scratch root) |
| `…/k1/tmp/booster_ws` | `/tmp/booster_ws` ← **the clone** |
| `…/k1/logs` | `/workspace/mounts/logs` |
| `…/k1/models` | `/workspace/mounts/models` |

**Trap:** `$C/logs` on the host is a symlink to `/workspace/mounts/logs` that only resolves
*inside* the container. Every host-side check through it sees a dangling link. Use the real
host path `…/k1/logs` when checking from the host.

### Sync before every run

```bash
cd /export/scratch/thakk100/k1/tmp/booster_ws
git fetch -q origin main && git reset --hard origin/main -q
git submodule update --init --recursive
```

`booster_train` is a **submodule** with real content of its own (ankle `armature_ratio`
fixes live there). A plain clone without `--recursive` will silently miss it.

### Stage the frozen base

The push env reads `models/k1_push_base.pt` (override with `PUSH_BASE_POLICY`). It is the
**squat teacher TorchScript, 238→12**. Copy it into the clone before any push run:

```bash
cp -f /export/scratch/thakk100/k1/logs/export/k1_squat_base.pt \
      /export/scratch/thakk100/k1/tmp/booster_ws/models/k1_push_base.pt
```

---

## 4. Running the actual training

### 4a. Reach stage (do this first)

```bash
S=/export/scratch/thakk100/k1; C=$S/tmp/booster_ws; M=/workspace/mounts/logs
cd $C && git fetch -q origin main && git reset --hard origin/main -q
ln -sf isaac_tasks/k1_velocity/scripts/train.py ./train.py
mkdir -p models && cp -f $S/logs/export/k1_squat_base.pt models/k1_push_base.pt

tmux new -d -s k1_reach
K1_TRAIN_SCRIPT=train.py K1_GPU=1 PUSH_BASE_MODE=squat ~/run_k1_train.sh \
  --task Isaac-Push-Reach-SG-K1-v0 \
  --num_envs 4096 --max_iterations 6000 2>&1 | tee $S/push_reach.log
```

**Start with a smoke test** — 3 iterations — before committing hours of GPU:

```bash
... ~/run_k1_train.sh --task Isaac-Push-Reach-SG-K1-v0 --num_envs 4096 --max_iterations 3
grep -aE "Learning iteration|Error|Traceback" $S/push_reach.log | tail -5
```

### 4b. Push stage — only after Reach works

Same command with `--task Isaac-Push-SG-K1-v0`.

### 4c. Squat distillation (already done — reference only)

```bash
K1_TRAIN_SCRIPT=train.py K1_GPU=1 ~/run_k1_train.sh \
  --task Isaac-Velocity-Squat-K1-Distill-v0 --num_envs 4096 --max_iterations 12000
```

The teacher must be **staged as a pseudo-run inside the student's own experiment dir**
(`p2_move_student/teacher_squat/model_squat.pt`) because `get_checkpoint_path` cannot
follow `..`.

### 4d. Plateau watchdog

Runs separately from `train.py` on purpose, so it can stop an in-flight run without
risking a corrupted training step:

```bash
K1_TRAIN_SCRIPT=plateau_stop.py ~/run_k1_train.sh \
  --log $S/push_reach.log --window 150 --min-delta 1.0 --patience 3
```

---

## 5. Recording video

**Reporting cadence (user requirement, 2026-10-10):**

- After every smoke test or basic test, post a short note (what ran, the result, a frame or
  video link) to Google Slides and Drive.
- While a training run is going, record a debug video from its latest checkpoint **every
  3 hours** (play run on GPU1), upload it to Drive and add a slide.

```bash
K1_TRAIN_SCRIPT=play_record.py K1_GPU=1 PUSH_BASE_MODE=squat ~/run_k1_train.sh \
  --task Isaac-Push-SG-K1-Play-v0 --checkpoint <ckpt> \
  --num_envs 8 --steps 700 \
  --video_out $M/push.mp4 --trace_out $M/push.npz \
  --panel_video --label "push policy"
```

Play tasks (small env count, no corruption, no curriculum — everything else identical to
the training cfg):

| Task | Purpose |
|---|---|
| `Isaac-Push-SG-K1-Play-v0` | render the push env |
| `Isaac-Push-Reach-SG-K1-Play-v0` | render the reach env |
| `Isaac-Velocity-Squat-K1-Play-v0` | render the squat **teacher** |
| `Isaac-Velocity-Squat-K1-Distill-Play-v0` | render the squat **student** |

`--checkpoint` for `play_record.py` may be absolute. (`train.py --checkpoint` may **not** —
it must be clone-relative.)

### Video quality: 1080p per panel, denoised (user requirement, 2026-10-10)

The user wants **clear videos: every panel 1080p, no renderer grain**. Since `6abff32`
`play_record.py --panel_video` defaults to:

| Flag | Default | Effect |
|---|---|---|
| `--panel_size W H` | `1920 1080` | size of **each** panel; the 2×2 tile is 3840×2160 + header |
| `--panel_spp` | `8` | `/rtx/rtpt/spp` (real-time path tracer samples per pixel; the kit default is 1, which is the grain) |
| `--panel_subframes` | `4` | Kit updates per captured frame with physics paused, so the temporal denoiser converges on each pose |

`set_clean_render()` also turns on `/rtx/newDenoiser/enabled` and the firefly filter.
Rendering stays clip-only (cameras exist only with `--panel_video`). It is several
times slower than the old 480×270 panels; use fewer `--steps` for quick checks. Don't go
back to small or noisy panels for anything shown to the user.

**Goal markers now render under `--headless` too.** `K1DebugMarkers` draws plain USD
curves (no `VisualizationMarkers`, which need a visualizer): a green wireframe of the
push goal pose (`push_state.goal_pos/goal_quat/half_extents`, the 8 corners `goal_reached`
scores) plus a small centre sphere, or for carry a ring at `success_radius` around the
goal. The header shows `goal (x,y) box->goal=…m` (push) or `goal … dist=…m` (carry).

### ⚠️ Always check `PANEL_VERIFY`, never file existence

A recording can exist, have the right size, and report the correct duration while being
almost entirely undecodable. The muxer writes the `moov` index whether or not the `mdat`
survives. **Gate on the marker:**

```
[PANEL_VERIFY] pre-move  frames_added=700 decodable=700 bytes=...
[PANEL_VERIFY] post-move frames_added=700 decodable=700 OK bytes=... /path.mp4
```

`OK` only if `decodable == frames_added`. `UNCHECKED` means the check could not run — that
is **not** the same as corrupt, and not the same as fine. Verify locally before reporting
a video to the user:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 video.mp4
```

Note: ffmpeg reports corruption as `Invalid NAL unit size`, **not** as lines containing
the word "error".

---

## 6. Build blockers (resolved 2026-10-10) — and the wrong diagnoses behind them

Fixed after the import-order fix, all verified in-sim:

* `expand_env_regex_ns` does not exist in the container's IsaacLab (`fb24c08`): expand
  with `GROUND_PATCH_PRIM.format(ENV_REGEX_NS=env.scene.env_regex_ns)`.
* `randomize_ground_friction` read the default material binding (empty → an *invalid*
  Material, not `None`) and wrote friction as shader inputs PhysX never reads. Now one
  `UsdPhysics.MaterialAPI` material per patch, bound on the `"physics"` purpose.
* `RayCaster currently only supports one mesh prim. Received: 2`: scanner casts onto
  `/World/ground` only; the frozen base's height scan adds `GROUND_PATCH_THICKNESS` to
  its offset (`push_mdp.FrozenBaseVelocityAction`).
* Early `height_below_command` terminations (~0.9 in the first iterations) come from
  noisy H* exploration (termination compares against the instantaneous command); with
  near-zero actions the base stands. Not a bug, but watch it.
* The old watchdog keyed on `Reward/total`, which the log never prints; use
  `--key "Mean reward"`.
* `make_zero_actor.py` raises `'Logger' object has no attribute 'writer'` *after*
  `torch.save` succeeds — the checkpoint is usable.

### Earlier: the import-order blocker

The push env failed to build with:

```
ValueError: Unknown asset config type for height_scanner: RayCasterCfg(...)
```

That message is **unreliable by construction**: `height_scanner` *is* a `RayCasterCfg`,
`RayCasterCfg` *is* a `SensorBaseCfg` subclass, and `SensorBaseCfg` is checked at
`interactive_scene.py:947`, well before the `raise` at 994.

**Actual cause:** `make_zero_actor.py` imported `isaaclab_tasks.utils.hydra`,
`isaaclab_rl.rsl_rl`, `rsl_rl` and the task packages **before** calling
`AppLauncher(args_cli)`. `SimulationApp` must exist first. Importing Isaac early splits
module identity, so `interactive_scene` tests the scene's cfgs against a `SensorBaseCfg`
object loaded after app start while the cfgs were built from the pre-app graph — nominally
the same class, distinct objects, `isinstance` returns `False`, dispatch falls through.

Fixed in `8dae594`. **Enforced** by `tests/test_import_order.py`, a static AST guard
across every script in `scripts/`: no `isaaclab*` / `rsl_rl` / `booster_train` import may
appear above the `AppLauncher` call, except `isaaclab.app` itself.

Four wrong diagnoses preceded it, recorded so they aren't repeated:

1. **Class identity** — plausible, wrong; `isinstance` was genuinely `True`.
2. **Resolver branch order** — wrong; the order was already correct.
3. **GPU contention / two Omniverse apps** — wrong; this *is* a real failure mode and did
   break a different run, which is exactly what made it tempting.
4. **Stale clone** — wrong; a launcher of mine ran `git checkout -- .` with no preceding
   `fetch`, which reverted my fixes and made the box run old code.

The thing that actually located it: a probe that instantiated `AppLauncher` **first** and
therefore imported second passed the identical build. **When a failure produces no
Python traceback, or an error that is unreachable by the reasoning it invites, diff
against a known-good sibling before theorising.**

### Still-unverified

`8cfd278` (ground-patch prim resolution) and `8dae594` (import order) are committed and
unit-tested but **have not been confirmed in-sim**. The first thing to do is the smoke
test in §4a. Expect `randomize_ground_friction` to print:

```
[push] ground patch prims: <N> (matched '/World/envs/env_.*/ground_patch')
[push] ground friction: <N> cells, mu 0.70-1.20 (floor 0.7 keeps the frozen base standing)
```

If the patch prims are not found, the function now raises naming the path it searched
rather than surfacing later as `Accessed schema on invalid prim` from unrelated code.

---

## 7. Operational traps (each one cost real time)

| Trap | Detail |
|---|---|
| `python.sh` / `run_k1_train.sh` return rc=0 on crash | **Gate on log markers**, never on rc |
| `pkill -f "<pat>"` over ssh | Matches its own remote command line and kills the launcher. Use a bracketed pattern: `[a]b.sh` |
| Two Omniverse apps at once | Kit dies at startup with only a crash dump, **no Python traceback**. Run sequentially |
| Two filesystems | `/tmp/opencode` and `~/run_k1_train.sh` are wiped on server restart; re-create launcher scripts |
| Laptop disk | Was at **100% (182 MB free)**, which silently killed the agent shell. Now 9.0 GB after clearing `~/.gz/sim/log` (8.5 GB of Gazebo logs). Check `df -h /` before big pulls |
| Videos in git | `git add -A isaac_tasks/` once swept 800 MB of mp4s into a commit; GitHub replied `pre-receive hook declined`. `videos/`, `*.mp4`, `*.npz` are now gitignored |
| `git push` from laptop | Occasionally stalls silently. Auth is fine (`ssh -T git@github.com` → `Hi thdhyan!`). Retry |
| `git checkout -- .` | Without a preceding `git fetch`, this reverts your own scp'd fixes |
| Test runner | `env -u PYTHONPATH -u ROS_DISTRO ~/Projects/IsaacLab-ea/.venv/bin/python -m pytest`. Bare `python3` hits a broken 750 MB user-site torch and **hangs** |
| `zz-bw` is shared | 2–25 users. Load has ranged 0.18 → 55. Iteration time for identical code swung 3.9 → 44 s/iter. Check `nvidia-smi` before assuming a code problem |

---

## 8. Fenced observation contracts

These were each wrong at some point. They are load-bearing — a mismatch is a shape error
at the first policy call, not a config nicety.

| Policy | Obs dim |
|---|---|
| squat teacher | **238** (237 + `H*`) |
| frozen push base (squat TorchScript) | **238** |
| velocity teacher | 237 |
| student | **500** (50 × 10 history) |

The legacy squat export was 236. `_process_squat` now appends the gait clock driven by
the action's velocity slice.

---

## 9. Suggested order of work

1. Run the **Reach smoke test** (§4a, 3 iterations). Confirm the scene builds and the
   `ground patch prims` / `ground friction` lines appear. **This is the gate for
   everything.**
2. Record `Isaac-Push-SG-K1-Play-v0` with `make_zero_actor.py` (untrained actor) and pull
   the video. Proves the environment without waiting for training — the robot will
   shuffle and fall over, which is expected and is the point.
3. Launch **Reach** training, 4096 envs, `max_iterations 6000`, watchdog attached.
4. Record the first Reach video as soon as `save_interval` produces a checkpoint.
5. Then **Push** stage, same task with `Isaac-Push-SG-K1-v0`.
6. Keep the squat student recording as the deployable-policy demonstration — it finished
   all 12000 iterations and is otherwise unused.

### Open question for the user

> "boxes without position" in the original request was never parsed and still needs
> clarification. Most likely it means: drop the box pose from the policy observation.
> That is a real change — `push_mdp.py` and the obs group in `push_env_cfg.py` — and it
> should not be guessed at.

---

## 10. Key files

| Path | What |
|---|---|
| `tasks/push/push_env_cfg.py` | scene, events, obs, actions, rewards, terminations, curriculum |
| `tasks/push/push_sg_env_cfg.py` | SG reward/env subclasses + the play variants |
| `tasks/push/push_rewards_sg.py` | 7 ported SG reward terms (arms-adapted) |
| `tasks/push/push_mdp.py` | `_process_squat` (238-dim), `randomize_ground_friction`, `reset_box` |
| `tasks/push/agents/rsl_rl_ppo_cfg.py` | PPO runner cfg |
| `scripts/train.py` | training entrypoint |
| `scripts/play_record.py` | render + HUD + panels + TorchScript export |
| `scripts/make_zero_actor.py` | untrained actor for env verification |
| `scripts/plateau_stop.py` | plateau watchdog |
| `scripts/panel_video.py` | tiled panel recorder + `PANEL_VERIFY` |
| `scripts/probe_scene_resolver.py` | scene diagnostic probe (AppLauncher first!) |
| `tests/test_import_order.py` | the guard that would have caught the blocker |
| `tests/test_push_scene_layout.py` | ground-patch path guard |

Reference repo for the SG terms: `~/Projects/ebasa/Push-Things`, branch `dhyan/obs_groups`.
It has `gated_*` rewards, **not** the 7 SG terms, and no K1 config — the port is
hand-written and self-contained, with no dependency on that repo.

---

## 11. Carry task (P7) — `Isaac-Carry-K1-v0` / `Isaac-Carry-K1-Play-v0`

User request (2026-10-10): box fixed between both forearm stubs, changeable width,
decoupled WBC trained with different box masses, walking to a goal; goal shown as a
faint red/green alpha cube.

| Piece | Where | Notes |
|---|---|---|
| Hold pose + weld geometry | `tasks/carry/carry_mdp.py` | IK table per width solves the **stub tip** (0.20 m along hand-link ±y), not the hand origin (that sits on the elbow axis). Mirror = `(1,-1,1,-1)`. Tested vs URDF: `tests/test_carry_geometry.py` |
| Box + welds | `spawn_carry_box` (prestartup) | per-env width U(0.25,0.40) m, mass U(1,8) kg, 0.18×w×0.18 m, no collider; two `UsdPhysics.FixedJoint`s box↔hand links, `excludeFromArticulation` |
| Goal command | `tasks/carry/carry_command.py` | registered as `base_velocity` (subclass of UniformVelocityCommand) so gait rewards keep working; derives (vx,vy,wz) toward the goal; goal 1–4 m any direction; success = within 0.25 m for 1 s → +20 and a new goal; cube 0.5 m, opacity 0.3, red → green while inside |
| Env | `tasks/carry/carry_env_cfg.py` | velocity env + flat plane, no height scan, env_spacing 10 m, arm `arm_pos` action (scale 0.25 around the hold pose), arm gains 40/2 Nm/rad (R14 default ≈4 can't hold a box) |
| Decoupled actor | `tasks/carry/decoupled_policy.py` | `SplitMLP`: legs MLP→12, arms MLP→8, both from the full obs, trained jointly by one PPO; JIT export unchanged. `tests/test_carry_decoupled_policy.py` |
| Runner | `tasks/carry/agents/rsl_rl_ppo_cfg.py` | experiment `p7_carry`, teacher obs group (incl. privileged box width/mass) |

Launch (running since 2026-10-10 16:36, run `2026-10-10_16-36-26_p7_carry`):

```bash
K1_TRAIN_SCRIPT=run_unbuffered.py K1_GPU=1 ~/run_k1_train.sh --task Isaac-Carry-K1-v0 \
  --num_envs 4096 --max_iterations 6000 --headless 2>&1 | tee $S/carry.log
```

Smoke test (64 envs, 3 it) passed: box DR 0.25–0.40 m / 1.0–7.9 kg, `DecoupledMLPModel`
built, `weld_error` ≈ −3e-4, `weld_broken` 0. First full-run iterations: same weld numbers.

**Bug fixed (`aad6079`):** the first debug video showed every box at the 1 m prototype
size, swallowing the robot. The spawner's `<box>/geometry` prim is typeless (not
Xformable), so the scale op authored there was ignored. `spawn_carry_box` now scales
`<box>/geometry/mesh` (the Cube, which already has `xformOp:scale`). Re-recorded
video shows the 0.18×w×0.18 box between the forearms.

Open risks:
- weld stability with 8 kg on 14 Nm arms (watch `weld_error` and the `weld_broken`
  termination rate);
- **goal cube: visible only without `--headless`.** IsaacLab 3 dispatches
  `VisualizationMarkers` callbacks from `SimulationContext.update_visualizers`, and only
  when a marker-capable visualizer is attached; `--headless` attaches none, so the
  cubes never get positions. Record carry play videos **without `--headless`** (the
  container has no display, and AppLauncher still renders off-screen). Confirmed in
  `carry_model_200.mp4`: red cube, turning green while the trunk is inside.
- the cube renders **opaque**, not faint: `opacity=0.3` on the PreviewSurface is
  ignored by this render path. Setting `/rtx/translucency/enabled` did nothing
  (`2911819`, reverted `d8f85c9`). Untried: OmniPBR/OmniGlass MDL material via
  `MdlFileCfg`, or a wireframe/edge marker instead of a solid cube;
- the play HUD header says `cmd=none (stand task)` for carry (the panel recorder's
  task heuristic). Cosmetic; the trailing `command (vx,vy,wz)` field is correct.

---

### 11b. Tiered rewards (`e658931`), user request 2026-10-10: "robot keeps falling"

The first carry run fell in ~15% of episodes (`base_orientation`) because every reward
needed walking. The user chose **gated tiers** and **kill + warm-start resume**:

| Tier | Terms | Rule |
|---|---|---|
| 1, always paid | `stand_hold` (w 1), `stand_still` (w 2) | `hold_quality` = exp(−‖box − nominal‖²/0.08²) · exp(−‖g_xy‖²/0.3²), box in the root frame vs `carry_state.box_offset`; `stand_still` also needs ‖cmd‖<0.05 and pays exp(−(v²+ω²)/0.2²) |
| 2, gated | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `goal_coarse`, `goal_fine`, `goal_reached` | wrapped in `carry_mdp.gated` (term × hold_quality) in `K1CarryEnvCfg.__post_init__`; the inner function is stored as a `"module:function"` string so Hydra can round-trip the cfg |
| 3, curriculum | `goal_radius` curriculum term | max goal radius starts at 1 m and grows by 0.25 m up to 4 m each time a window of `num_envs` resets has < 8% falls (`base_orientation` + `root_height`). Logged as `Curriculum/goal_radius/{max_radius,fall_rate}` |

Also: 20% of goals are **stand goals** (radius 0, 4 s dwell), and `goal_reached` is
scaled by `clamp(goal_r / 1 m, 0, 1)`, so standing can't farm the success bonus.
Min radius is now 0.5 m. Pure maths is in `carry_tiers.py` (no Isaac imports), tested by
`tests/test_carry_tiers.py`. Observation dims are unchanged, so old checkpoints load.

## 12. Videos delivered (2026-10-10)

Recorded on zz-bw with `play_record.py --panel_video --label` (4 panels: overview,
top_down, follow, side, plus status header). The container's `PANEL_VERIFY` prints
`UNCHECKED` (its frame counter hits `IndexError`), so the files were checked locally
with `ffprobe -count_frames` and frames were inspected.

| Clip | Checkpoint | Frames | Drive file id |
|---|---|---|---|
| `p6_push_sg_it300_2026-10-10.mp4` | `p6_push/2026-10-10_15-54-48_p6_push/model_300.pt`, `PUSH_BASE_MODE=squat` | 700 | `1epQPCiKQQcpZ9--sA7kuUZlkx_RUcQtn` (poster `1Fa-HVpfHYbdwI2M79DFHP1ttNs4CSMYZ`) |
| `p7_carry_smoke_untrained_2026-10-10.mp4` | `p7_carry/2026-10-10_16-10-43_p7_carry/model_2.pt` (smoke, untrained; recorded `--headless`, so no goal cube) | 500 | `1znMNmwKo_cROberelAz-wVQOzjWgMDKK` |
| `p7_carry_model200_2026-10-10.mp4` | `p7_carry/2026-10-10_16-36-26_p7_carry/model_200.pt` (no `--headless`; goal cube visible) | 750 | `11QoC98nUhh0_ZQ21IBpMO3chouKaH5TZ` |

- Drive folder: [Booster videos](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy).
- Slides: appended as slides 34–36 (`s_p6sg_1010`, `s_p7carry_1010`, `s_p7c200_1010`, layout `p12`,
  embedded Drive video plus a title text box) to the
  [deck](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit).
- Local copies: `k1m_out/push_videos/` (gitignored output dir).
- **Upload route:** Composio's `GOOGLEDRIVE_UPLOAD_FILE` caps at 5 MB. Use the laptop's
  rclone remote instead: `rclone copyto <mp4> gdrive-thakk100:<name> --drive-root-folder-id 1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy`
  (`gdrive-dhyan` is not configured on the laptop). Slides go through Composio
  `GOOGLESLIDES_PRESENTATIONS_BATCH_UPDATE` (`createSlide` + `createShape` TEXT_BOX +
  `createVideo` source DRIVE, guarded by `requiredRevisionId`). Layout `p12` has no
  TITLE placeholder.

Next videos: Push-SG at a later checkpoint (e.g. model_1500) and carry at ~model_1000,
uploaded the same way.

## 13. Multi-robot push (M1 play, M2 training) — 2026-10-10

- **M1** `Isaac-Push-Multi-K1-Play-v0` (`push_multi_env_cfg.py`): 2 K1s at y=±0.5 m, one shared
  box (0.85–0.95 m, 3–10 kg), forward goal cone. Per robot: `base_velocity_k`, `wrist_{left,right}_k`,
  `wrist_target_k`, `height_scanner_k`, 109-dim `teacher_k` (single-robot layout).
  `play_record.py` stacks the groups and runs the single-robot policy once.
  Smoke (p6 `2026-10-10_18-59-36/model_200`): pipeline OK, 400 frames 3840×2300, but robots drift off
  the box (wrist-box gap 0.8→1.3 m, cmd saturated) — the checkpoint is too early.
  Drive `p6_push_multi_m1_smoke_2026-10-10.mp4` (`1F5QllGXsA8tEJM537zlnmu2erAhvGJoU`), slide 38.
- **M2** `Isaac-Push-Multi-K1-v0` + `AgentsInBatchWrapper` (`agents_in_batch.py`, applied by train.py
  when `env_cfg.agents_in_batch > 0`): N envs → 2N rows, one shared actor/critic (IPPO with shared
  weights), reward per robot = team terms + own `_k` terms, dones broadcast. Box 6–16 kg, obs noise on.
  Runner `p6_push_multi`. Unit test with a fake env passed; 3-iteration train smoke (64 envs, GPU0)
  passed. `--checkpoint` from another experiment now loads directly (commit 30b4672).
- **Full M2 run not started**: dl disk 100 % + 4 GPUs busy (other user), lambda NVML mismatch
  (reboot). Proposed: zz-bw GPU0 next to ball-track:
  `K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/train.py K1_GPU=0 ~/run_k1_train.sh --task Isaac-Push-Multi-K1-v0 --num_envs 2048 --max_iterations 3000 --headless --checkpoint logs/rsl_rl/p6_push/<best run>/model_<N>.pt --warm_start`
  Better: warm-start from a single-robot push checkpoint that already reaches goals.
- Status slide 37 + Drive `status_2026-10-10_2000.md`.

### 13b. MAPPO, separate policies (skrl fork) — 2026-10-10 ~20:40

- User asked for skrl MAPPO with **one policy per robot**, using the fork `github.com/thdhyan/skrl`
  (skrl 2.1 API; its MAPPO equals upstream develop, the fork only adds POMDP/OmniReset wrappers).
- `marl_adapter.py`: `MultiRobotMARLAdapter` exposes the manager-based multi-robot push env as a
  DirectMARLEnv (`possible_agents`, per-agent spaces, `state()` = concat of every `teacher_k`, dict
  step/reset) for skrl's `IsaacLabMultiAgentWrapper`. Rewards via `RewardSplit` (shared with M2).
- `scripts/train_mappo.py`: per robot a Gaussian policy (109→512-256-128→10, ELU) + a centralized
  value net (218→…→1). `--checkpoint <rsl p6 .pt>` warm-starts every agent: actor weights,
  log_std = log(std_param), obs scaler ← rsl normalizer; critic weights on the agent's own 109-column
  block (zeros on the other robot's), shared scaler = stats repeated. Scalers use clip 1e3 (rsl does
  not clip; skrl's default is 5σ). Logs `logs/skrl/p6_push_mappo/<stamp>/checkpoints/agent_<t>.pt`.
- zz-bw: the fork is cloned at `$S/tmp/skrl` (= `/tmp/skrl` in the container, pinned `1388269b`);
  train_mappo puts it first on `sys.path` (`K1_SKRL_PATH` overrides). The container cannot keep
  user-site installs (HOME override blocked), so do not `pip install --user` skrl.
- Fix needed for skrl: it steps the env under `no_grad`, not `inference_mode`, so the frozen base's
  `_last` buffer was an inference tensor → in-place reset failed. Now cloned (`56e9a23`).
- Smoke (64 envs, 3 updates, GPU0): passed, `2026-10-10_20-32-18_p6_push_mappo/checkpoints/agent_72.pt`.
  ~1.1 s per env step at 64 envs while GPU0 also trains ball-track.
- Play: `play_record.py --mappo --checkpoint .../agent_<t>.pt --task Isaac-Push-Multi-K1-Train-Play-v0`
  (plain-torch loader, no skrl needed). Smoke video queued after the batch: `$S/record_mappo_smoke.sh`
  → `videos/batch_1010_2015/mappo_smoke_it3.mp4`.
- Full MAPPO run: **not started**, same host question as M2 (needs user OK for shared GPU0):
  `K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/train_mappo.py K1_GPU=0 ~/run_k1_train.sh --task Isaac-Push-Multi-K1-v0 --num_envs 2048 --max_iterations 3000 --headless --checkpoint logs/rsl_rl/p6_push/<run>/model_<N>.pt`

### 13c. Video batch fix — 2026-10-10 ~20:25

- First 20:06 batch wrote no mp4s: the panel header wrapped to a varying number of lines, so the
  frame height changed and imageio raised "All images in a movie should have same size" (rc still
  printed 0 because the script echoes `$?` of the launcher wrapper). Header line count is now fixed on
  the first frame (`4a05dc3`); batch restarted 20:27 (`$S/record_batch_1010.sh`, tmux `k1_rec_batch`).
- Marker fix (`7416e48`): velocity target/estimate spheres hidden in multi-robot videos. Remaining
  markers: cyan = robot root, orange = wrist targets, green wireframe = goal box.

### 13d. 3-hour report 21:45 + MAPPO full run

- MAPPO full run (user OK 20:38): tmux `k1_mappo`, GPU0, 2048 envs, 3000 it, warm p6 it400,
  `logs/skrl/p6_push_mappo/2026-10-10_20-39-31_p6_push_mappo`, wandb run `8d0jdhbz`; ~1.6 s/step → ~31 h.
- 20:27 batch: only `p6_push_it300` finished (71 min/clip on loaded GPU1); stopped and replaced by
  `$S/record_batch_3h_2145.sh` (tmux `k1_rec_3h`, latest ckpts, 400 steps each):
  kick it700 → track it700 → carry it2000 → squat it21800 → push it500 → MAPPO it100 (`--mappo`).
  Output `logs/videos/batch_3h_2145/`.
- Drive: `status_2026-10-10_2145.md`, `p6_push_it300_2026-10-10.mp4`; slides 39 (status) + 40 (push video).
