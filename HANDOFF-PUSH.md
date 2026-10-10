# HANDOFF — PUSH (box-pushing) workstream on `zz-bw`

**For:** a fresh agent or a new session picking this up cold.
**Scope:** the K1 box-push task only. The velocity/gait/soccer workstreams are tracked in
`HANDOFF.md`, `TRAINING.md`, `TASKS.md` and `STATE.md` — those are stale (Sept–Oct 6) and
are **not** updated by this workstream. Do not edit them from here.
**Repo:** `/home/thakk100/Projects/booster_ws`, branch `main`, GitHub `thdhyan/booster_ws`.
**Host:** `zz-bw` (`cs-zhang-net-01`) — 2× RTX PRO 6000 Blackwell, 97 GB each.
**Last commit this doc describes:** `8dae594`.

---

## 0. TL;DR — where things actually are

| Item | State |
|---|---|
| Squat **teacher** | ✅ trained, `model_8100.pt`, success 0.83 |
| Squat **student** (distill) | ✅ **finished all 12000 iterations** → `model_11999.pt` |
| Squat teacher TorchScript (push base) | ✅ `k1_squat_base.pt`, 1.17 MB, verified 238→12 |
| Push env — **builds** | ⚠️ fixed on `main` (`8cfd278`), **not yet verified in-sim** |
| Push **training** | ❌ never started. No push policy exists. |
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

`K1_TRAIN_SCRIPT` must be reachable from the workdir root, so symlink it first:

```bash
cd /export/scratch/thakk100/k1/tmp/booster_ws
ln -sf isaac_tasks/k1_velocity/scripts/train.py ./train.py
```

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

## 6. Current blocker — and the four wrong diagnoses behind it

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
