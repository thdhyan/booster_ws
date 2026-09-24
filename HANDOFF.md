# Booster K1 Workspace — Handoff

## 📌 SESSION HANDOFF — 2026-09-24 (read this section first; older history below)

> **Branches:** `dev/soccer-p3p4` @ `f249c95` (active, pushed — P3/P4/push work) ·
> `dev/phase-6-soccer-hrl` @ `f3000d6` (pushed — recordings/docs; videos+models live here on GitHub) ·
> the original `dev/box-push-wrist-ik` branch was **renamed** to `dev/soccer-p3p4` (box-push env is authored on it, same branch).
> **GitHub:** https://github.com/thdhyan/booster_ws · **Plan:** `PLAN_PHASE6_SOCCER_HRL.md` · **Run log:** `TRAINING.md`
> **Drive:** [policy videos folder](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy) · [Slide deck (4 videos embedded)](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit) ·
> training videos auto-mirror from the sparks (below).

### What is RUNNING right now (all remote, in tmux, survive disconnects)

| Box | tmux | What | Log |
|---|---|---|---|
| spark02 | `k1_spark_hp` | **Run-11** partial-control + arm-delta curriculum (smoke-gated → 512×3000) — relaunched 13:17 UTC after the server restart | `scripts/hp.full.log` (Run-10 archived as `scripts/hp.run10.log`, `HP_FULL_RC=0` @ iter 2999) |
| spark02 | `k1_gdrive_sync` | rclone loop → Drive `gdrive-thakk100:Booster/logs` (`*.mp4`, every 10 min) ✅ live | `scripts/gdrive_sync.log` |
| spark02 | `k1_p3smoke` | P3 head-track smoke (16×3, cameras+YOLO), 1-D-reward build — relaunched 13:17 UTC | `scripts/p3smoke.log` |
| — | — | Frozen-base export from Run-10 `model_2999` **DONE** (parity 2.87e-03): `models/k1_partialctrl_base.pt` on local + spark02 + spark04 | `scripts/export_base.log` |
| spark04 | `k1_spark_p1f` | P1f chain **resumed from teacher `model_1999.pt`** (1 iter left, then student 256×1500) | `scripts/p1f.f.log` |
| spark04 | `k1_spark_p2f` | P2f chain **resumed from teacher `model_1800.pt`** (1200 iters left, then student 512×3000) | `scripts/p2f.f.log` |
| spark04 | `k1_gdrive_sync` | Drive mirror (same as spark02) ✅ live | `scripts/gdrive_sync.log` |
| spark04 | `k1_pushsmoke` | **Box-push smoke** (16×3, video), event-signature fix — relaunched 13:17 UTC | `scripts/pushsmoke.log` |

spark01/03 are busy with other tenants (GPU 83/95 %). Do not touch.

### ⚠️ Server restart incident (2026-09-24 ~12:55 UTC) — RECOVERED

Both sparks rebooted mid-flight: **every training container and tmux session was
killed** (the `k1_gdrive_sync` loops came back / survived and were re-checked).
Salvage + recovery already done:

- **p1f teacher reached 1999/2000** → `model_1999.pt` survived. **p2f teacher at
  1854/3000** → `model_1800.pt`. Both chains relaunched **with resume** (new
  `train.py --checkpoint <path>` flag + `RESUME_CKPT` env in
  `scripts/spark_force_{host,container}.sh`) instead of restarting from zero.
- **Run-10 completed before the reboot** (`HP_FULL_RC=0`), `model_2999.pt`
  pulled to `logs/rsl_rl/k1_partialctrl_base/2026-09-23_22-20-44_k1_partialctrl_base/`,
  log archived as `hp.run10.log`, **frozen base re-exported from it** (parity
  2.87e-03) and copied to local + spark04.
- **Lesson:** remote jobs are NOT reboot-safe in tmux alone — checkpoints every
  100 iters are the only durability. After any restart, check
  `ls -t logs/rsl_rl/*/*/model_*.pt` and relaunch with `--checkpoint`.

### First 5 commands when you return

```bash
# 1. both smokes + Run-10 status
ssh aim_spark04 'grep -E "PUSH_SMOKE_RC|\[push\]|frozen base|Error" ~/Projects/booster_ws/scripts/pushsmoke.log | tail'
ssh aim_spark02 'grep -E "reward .* returned|P3_SMOKE_RC" ~/Projects/booster_ws/scripts/p3smoke.log | tail -3; grep -E "Learning iteration|HP_FULL_RC" ~/Projects/booster_ws/scripts/hp.full.log | tail -2'
# 2. push-smoke artifacts (the video the user asked to see) — see "Pending" below
ssh aim_spark04 'ls -la ~/Projects/booster_ws/logs/rsl_rl/p6_push/*/videos/ 2>/dev/null'
# 3. force-run progress
ssh aim_spark04 'grep "Learning iteration" ~/Projects/booster_ws/scripts/p1f.f.log | tail -1; grep "Learning iteration" ~/Projects/booster_ws/scripts/p2f.f.log | tail -1'
```

### Pending — in priority order

1. **Box-push (P6): green smoke → video + screenshots + README.** The smoke
   (`Isaac-Push-K1-v0`, 16 envs, `--video`) is the user's explicit deliverable:
   on `PUSH_SMOKE_RC` success, (a) rsync
   `aim_spark04:.../logs/rsl_rl/p6_push/*/videos/*.mp4` → local
   `isaac_tasks/k1_velocity/videos/push_smoke.mp4`; (b) extract 2–3 PNG frames
   (in-container `ffmpeg -i ... -vf "select=eq(n\,K)" -vframes 1`; ffmpeg is in
   the image) into `videos/push_smoke_frame*.png`; (c) add a **P6 box-push
   section to `isaac_tasks/k1_velocity/README.md`** (obs/action/reward tables +
   video embed + screenshots); (d) commit + push; (e) upload the video to the
   Drive folder (permission anyone/reader) and add a slide to the deck
   (`GOOGLESLIDES_PRESENTATIONS_BATCH_UPDATE` with `createVideo`, **source
   enum is `DRIVE`**, not `DRIVE_FILE`; Composio session id `trip`).
   If the smoke still fails: the remaining suspects are the two IK action
   terms and the frozen-base action term (everything else was verified: cfg
   validation passed, obs/actions/rewards/events all instantiate).
2. **P3 smoke green → launch full runs** on spark02:
   `bash scripts/spark_soccer_host.sh p3` (tmux `k1_spark_p3`, log
   `scripts/p3.soccer.log`; smoke-gated internally, then 512×2000) and
   `bash scripts/spark_soccer_host.sh p4t` (P4 teacher, 512×3000). Only one
   Isaac stack at a time on spark02 beyond Run-11 (RAM ~121 GB total).
3. ~~**Run-10 → Run-11 handoff**~~ — **DONE 2026-09-24 03:35 UTC**: Run-10
   finished `HP_FULL_RC=0` (iter 2999/3000); `hp.full.log` archived as
   `hp.run10.log`; `model_2999.pt` pulled to
   `logs/rsl_rl/k1_partialctrl_base/2026-09-23_22-20-44_k1_partialctrl_base/`;
   Run-11 launched (tmux `k1_spark_hp`). Remaining from that step:
   (a) `scripts/export_base_policy.sh` is re-exporting the frozen base from
   the final ckpt — pull `models/k1_partialctrl_base.pt` (local→spark04) when
   `BASE_EXPORT_DONE` appears; (b) partial-control play video on spark04:
   `ssh aim_spark04 'cd ~/Projects/booster_ws && SKIP_CORE=1 bash scripts/record_policies_host.sh'`.
   Original commands kept for reference:
   ```bash
   ssh aim_spark02 'cd ~/Projects/booster_ws && cp scripts/hp.full.log scripts/hp.run10.log && tmux kill-session -t k1_spark_hp 2>/dev/null'
   rsync -az aim_spark02:Projects/booster_ws/logs/rsl_rl/k1_partialctrl_base/2026-09-23_22-20-44_k1_partialctrl_base/model_2999.pt logs/rsl_rl/k1_partialctrl_base/2026-09-23_22-20-44_k1_partialctrl_base/
   # ship code local->spark02, re-export the frozen base from the FINAL ckpt, then:
   ssh aim_spark02 'cd ~/Projects/booster_ws && bash scripts/spark_partial_host.sh'   # Run-11, tmux k1_spark_hp
   # partial-control play video (SKIP_CORE=1) on spark04:
   ssh aim_spark04 'cd ~/Projects/booster_ws && SKIP_CORE=1 bash scripts/record_policies_host.sh'
   ```
   Re-export the frozen base for P6 after Run-10/11 final:
   `scripts/export_base_policy.sh` (container-side) → `models/k1_partialctrl_base.pt`.
4. Force chains (p1f/p2f) finish on their own (teacher→student chained);
   record + upload their videos when done (same recipe as #1).
5. Commit anything uncommitted; branches are pushed at `f249c95`.

### Architecture decisions LOCKED this session (user-confirmed)

- **P3 head tracking** (`Isaac-HeadTrack-K1-v0`): head-only (legs PD-held),
  deployable obs = **camera detections only** (YOLOv8n `sports ball` bbox on a
  head cam mounted on `Head_2`, 320×240 @10 Hz → `(visible, du, dv)`), no GT;
  primary reward = centredness of the detection; **ball-speed curriculum
  0→0.8 m/s (iters 200→1200)**; **CCW search**: no detection 0.5 s → in-place
  counter-clockwise command `(0,0,+0.6)` to the locomotion policy
  (`head_mdp.ccw_search_command` + compose `update_search`).
- **P4 teacher** (`Isaac-Kick-Ball-K1-Teacher-v0`): now also drives the head
  (14-dim action) + `ball_in_frame`/`head_ball_aim` shaping; student path
  unchanged (needs P3 + vision estimator later).
- **P6 box-push** (`Isaac-Push-Reach-K1-v0` → `Isaac-Push-K1-v0`):
  - **9-dim action = velocity override (3) + left/right wrist EE deltas (3+3)**;
    the velocity slice feeds a **frozen Run-11 partial-control TorchScript**
    (`models/k1_partialctrl_base.pt`, obs assembled in the exact 68-dim partial
    order inside `FrozenBaseVelocityAction`); wrists via two stock
    `DifferentialInverseKinematicsActionCfg` terms (`body_name=left/right_hand_link`,
    `scale=0.05`, dls, position mode, relative). Contact-only (no welds).
  - Box = 1 m prototype cube; **USD-time per-env DR** (mode `"usd"`,
    `replicate_physics=False`): edge 0.7–1.5 m, mass 3–25 kg (Reach: 12–25),
    friction 0.3–1.2, fixed per env; semi-transparent green (pxr
    `UsdPreviewSurface` opacity 0.35, startup event).
  - Goals: **corner corners + corner centroid ("cumulative sum")** tracking
    against a cumulative goal-offset integrator along the push heading
    (curriculum 0.3→1.5 m). Fully observable **teacher** obs group (mass,
    size, 8 corners, goal pose + corners, offset, wrist targets, proprio).
  - Wrist-target command = 2 contact points on the box near face (base frame,
    clamped to reach); wrist action = deltas around current EE pose.
- **Deployability test**: push uses the `TeacherCfg` group name (privileged
  convention) so the static gate still enforces the no-GT rule everywhere
  else. Gate currently **HOLDS — 14 tasks** incl. head + kick + push.

### Gotchas that cost hours (do not rediscover)

- **Reward terms must return 1-D `(N,)`** — the reward buffer is 1-D, so a
  `(N,1)` term broadcasts to `(N,N)` and throws
  `output with shape [N] doesn't match the broadcast shape [N, N]` inside
  `reward_manager.compute` (the traceback does NOT name the term — guard your
  custom terms or bisect).
- Event/curriculum/obs term signatures are validated against
  `(env, env_ids, …)` (min_argc=2 for events/curricula): a missing `env_ids`
  parameter or non-defaulted names fails env construction with
  "expects mandatory/optional parameters".
- `python.sh` returns **rc=0 on crashes** — always read the log, gate on
  artifacts (mp4 size, trace, export), never on rc alone.
- **Curriculum terms** must be `def f(env, env_ids, <all-defaults>)` — the
  manager validates the signature skipping TWO params and rejects unexpected
  names. Iterations come from
  `(sim.get_physics_step_count() // decimation) / steps_per_iter`, not
  `common_step_counter`.
- Tensor indexing: `joint_pos[:, ids][:, 0]`; `joint_pos[:, ids, 0]` silently
  yields `(N, 2)` (bit P3 and kick once already).
- `randomize_rigid_body_scale` is **USD-cooked**: only before sim start
  (`mode="usd"`, per-env fixed) and requires `replicate_physics=False`.
- ultralytics `model.predict` on a **tensor** requires stride-32 dims
  (240×320 rejected); pass a **list of HWC uint8 frames** instead.
- configclass: `class_type` must be set **at decoration time** (assigning it
  after `@configclass` leaves `None` → "Missing values detected").
- `DifferentialInverseKinematicsActionCfg` field is **`body_name`**, not `body`.
- Smoke scripts must export `WANDB_API_KEY` from `logs/.wandb_key` (or die at
  runner init). `scripts/spark_soccer_container.sh` does this for full runs.
- Drive remote on the sparks is **`gdrive-thakk100`** (dl uses a different
  alias); rclone linux-arm64 binary + config are installed on spark02/04 and
  synced in tmux `k1_gdrive_sync`.
- Sparks: container-created files are **root-owned** (videos, `__pycache__`) —
  rsync with `--no-owner --no-group --no-perms --exclude='__pycache__'`.
  Occasional `ssh: Could not resolve hostname aim_sparkNN` blips → retry.
- Never `git add src/k1_description/assets` (pre-existing dirty submodule).
- The box-push env lives in `isaac_tasks/k1_velocity/source/k1_velocity/tasks/push/`
  (not a separate package); all 4 registration sites import it
  (`tasks/__init__.py`, `register_tasks.py`, `scripts/train.py`,
  `scripts/play_record.py`).

### Completed earlier the same day (for context)

- Recorded + verified 4 policy videos (P1 t/s, P2 t/s) with HUD, JIT parity
  2.2–3.7e-03, TorchScript exports in `models/` (LFS) — committed on
  `dev/phase-6-soccer-hrl`, embedded in READMEs, mirrored to Drive + Slides.
- P1f/P2f force-variant tasks (teacher sees the 6-dim shove wrench, students
  blind) — smoke-gated chains launched on spark04.
- Run-10 partial-control base (arm-pose curriculum) + Run-11 arm-delta
  curriculum implemented (`arm_delta_change` 0→1 over iters 600→2500, play cfg
  pins 0).
- k1_soccer_compose.py: 12-dim detection head-obs + CCW search state machine.

---

> **Older handoff (2026-09-22)** — kept below for history.
> **Date:** 2026-09-22 · **Branch:** `dev/phase-6-soccer-hrl` @ `16a6803` (`main` == `origin/main` @ `88f73fc`, tag `v0.7-phase0-complete` pushed)
> **GitHub:** https://github.com/thdhyan/booster_ws
> **Read first:** `STATE.md` (snapshot) → `PLAN_PHASE6_SOCCER_HRL.md` (the active plan) → `TASKS.md` T6.* (execution list)
> Supersedes the older two-agent handoff (`HANDOFF-AGENTS.md` — contact-sensor + kick-task jobs **landed**: `faefa3e`, `89b13a9`).

---

## Current State

### ✅ T6.2 — IsaacLab 3.0-EA task migration (2026-09-22 — **Gate G1 PASSED**, `16a6803`)
- **Canonical launch (supersedes `start_training.sh` + old `scripts/train.py` wrappers — 3.0 removed backend-local scripts):**
  ```bash
  source scripts/phase6_env.sh
  scripts/train_guard.sh --name <run> -- isaaclab train --rl_library rsl_rl \
      --task Isaac-Velocity-Rough-K1-v0 \
      --external_callback k1_velocity.register_tasks.register_tasks \
      --num_envs 256 --max_iterations 5000 --seed 42 --viz none
  ```
  Third-party tasks register via `--external_callback` (→ `k1_velocity/register_tasks.py`); packages ride `PYTHONPATH`. Distillation runs are native: same command, `--task Isaac-Velocity-Distill-K1-v0 --checkpoint <teacher model.pt>`.
- **G1 evidence:** 16 envs × 2 iters `rc=0` · wandb online `booster_k1_locomotion/x1ptwkae` (that cfg predated the project switch; runs now log to **`booster_k1_soccer_hrl`** per PLAN §3.6) · `logs/rsl_rl/k1_velocity_rough/<ts>/model_0,1.pt` · `feet_air_time`=3e-05 (>0) + `feet_slide`=-2.2e-04 → **contact fires directly from `K1_22dof.urdf` on the 6.1 importer — `flatten_k1_usd.py` is off the training path** (kept for reference).
- **Key 3.0 deltas applied:** `isaaclab_tasks.core.velocity` imports · `sim.use_newton_actuators=False` (custom `BoosterDelayedPDActuator` needs the Isaac Lab execution path) · fork **`booster_train` `2879b1a`** (`_parse_joint_parameter` → `resolve_joint_parameter`) · `SceneEntityCfg.body_names` = bare link names (`left_foot_link`, no `Robot/` prefix; prim_paths keep it) · wandb project `booster_k1_soccer_hrl` · guard fix (invalid `ManagedOOMPreference=nodest` killed every scope on systemd 255) · `phase6_env.sh` puts venv bin on PATH (`isaaclab` CLI).
- **Kick (T6.2.3, ~half done):** cfg imports+instantiates ✅; mdp restored to authentic `89b13a9` **244-line** version (the `88f73fc` LFS restore had silently landed a stale 210-line cached object missing `reset_ball_omnireset` — see STATE.md) + `write_root_pose/velocity_to_sim_index` + `ProxyArray.torch` in `_set_ball_pos`. **Remaining:** obs-fn ProxyArray/quat audit (`ball_pos_in_robot_frame` etc.) + kick smoke.

### ✅ Training environment rebuilt (2026-09-22, T6.1 — Gate G0 PASSED)
- **Venv:** `~/Projects/IsaacLab-ea/.venv` (uv-managed, py3.12) — Isaac Sim **6.1.0** + IsaacLab **v3.0.0-EA** (worktree `~/Projects/IsaacLab-ea`; shared `~/Projects/IsaacLab` untouched on `perf-2026-07-06`), torch 2.11.0+cu128 (CUDA OK), rsl_rl, **ultralytics 8.4.158**, wandb, imageio+ffmpeg.
- **G0**: headless Kit 110.3 launch on the RTX 4060 ✅. 9 K1 tasks register (velocity rough/distill/contact/play + kick teacher/distill).
- **Use:** `source scripts/phase6_env.sh` → `phase6-python <script>` (packages ride PYTHONPATH; not pip-installed: root-owned egg-info removed, and EA dropped the `omni.isaac.lab.tasks` entry-point group so metadata isn't consumed). **Reproduce/repair:** `scripts/install_phase6_env.sh` — any future `uv sync` strips the extras, rerun it after.
- ⚠️ `~/Projects/IsaacLab/isaac6/.venv` belongs to a **parallel G1_sim session** (recreated it twice mid-run) — do not use or touch.
- `zz-bw` HPC still unreachable → **local-only training**; disk now **16 GB free** (guard floor 5 GB).

### ✅ Landed (still valid)
- Velocity task trained (old env): student reward 40.89 (`models/k1_velocity_student.pt`), teacher 25.08 (`velocity_teacher_4999.pt`).
- Kick task `Isaac-Kick-Ball-K1-v0` (OmniReset, goal detection) — `89b13a9`.
- USD flatten + contact rewards re-enabled — `faefa3e` (`scripts/flatten_k1_usd.py`, `K1_flat.usd` 4.2 MB). *Contact end-to-end never verified; IL 3.0 importer may make flatten obsolete — re-check.*
- Fleet sim ×3 backends (Gazebo/MuJoCo/Isaac), stereo ZED 2i rig, RoboCup MuJoCo capture — see `PHASE0-DONE.md`, old details in git history.

### 🟡 Drafts landed in git (T6.0, `6ae1d94`) — still not runnable
`isaac_tasks/k1_head_tracking/` + `isaac_tasks/k1_kicking/` (lone env-cfg files, NOT runnable: no `__init__.py`/agents/scripts, obs wired via velocity-command hacks) · `k1_soccer_compose.py` (3-policy state machine, needs trained head/kick policies) · `PLAN_MULTILAYER.md`.

### ✅ Phase 6 plan + rewards LOCKED
Written this session, approved by user in chat:
- **`PLAN_PHASE6_SOCCER_HRL.md`** — full plan: git closure, env rebuild + IL 3.0-EA migration table, DR spec (incl. **ball position + ball color**), **auto-stability push randomization** (`random_body_push` on waist/pelvis/chest), camera+YOLO, **vision estimator (no-GT deployability)**, video cadence (200 iters → 30 s), wandb, PC guardrails, 9-run training matrix, gates G0–G6.
- **`TASKS.md` § PHASE 6** — T6.0–T6.6 checklist.
- **`STATE.md`** — repo/hardware/env snapshot.

---

## Phase 6 design (one-paragraph version)

Five policies, all MLP (512,256,128): **P1 stand** and **P2 walk** are *teacher–student* (teacher: rough terrain + height scan + foot contacts; student: blind 10-step history) with **random 20–80 N body-part pushes** so the robot self-balances when shoved at the waist. **P3 head-track** (2-dim head, YOLO-only obs) keeps the ball in the depth camera. **P4 chase&kick** is *teacher–student*: teacher sees true ball/goal, **student sees only the vision estimator** (YOLO bbox + depth → ball pos/vel; rectangular-goal tracker/rememberer → goal pos) — head driven by frozen P3 during rollouts. **P5 HRL manager** (5 Hz) picks among frozen P1–P4 + (vx,vy,wz) params; its student is estimator-only (200-dim history MLP). Rewards are user-approved and LOCKED in the plan — GT allowed only in rewards/teacher obs, never in deployable obs (static test T6.3.5 enforces). Trained sequentially, 256–512 envs, headless, wandb project `booster_k1_soccer_hrl` (entity `thakk100-dhyan-home`), video every 200 iterations.

---

## Machine rules (do not violate)

- **RTX 4060 8 GB / 16 GiB RAM / 16 GB disk free (post-install).** One training at a time; ≤512 envs; `train_guard.sh` watchdog + `systemd-run --scope MemoryMax=9G`; GPU >7 GB ⇒ kill. Never 4096 envs locally.
- Headless only (`--viz none` in IL 3.0).
- Videos: every 200 iterations, 30 s → `logs/videos/<run>/` + wandb.
- Never source system ROS2 and Isaac ROS2 together.
- wandb entity is **`thakk100-dhyan-home`** (never `thakk100`/`thdhyan`).

## Gotchas carried forward

- K1 joint A-prefix: `AAHead_yaw`, `ALeft_Shoulder_Pitch`, `ARight_Shoulder_Pitch`.
- URDF-imported USD nests link prims → contact sensors needed flat hierarchy (flatten script). IL 3.0 importer still nests by design; PR #6378 (merged on develop) may fix contact addressing — verify before re-flattening.
- rsl_rl ≥4: `actor=`/`critic=` model cfg, not `policy=`; deterministic distillation needs the `output_std` class-property guard (copy from `scripts/train_student.py`).
- IsaacLab 3.0 breaking changes: **quats XYZW**, `ProxyArray` (`.data.*.torch`), `write_*_to_sim_index/mask`, `isaaclab train` CLI, `VideoRecorderCfg`, `--viz none`, `enable_extension` for direct isaacsim imports — full table in plan §2.
- IsaacLab ships its own agent skills (`isaaclab-migrating-2x-to-3x` etc.) — use them during migration.
- LFS incident (fixed 2026-09-22): `002a647` (Sep 2) added `*.py filter=lfs` to `.gitattributes`; `dd3f6b2` (Sep 4) removed the rule but never de-LFS'd the 62 already-converted `.py` files, so a later checkout overwrote real sources with raw pointer blobs. Restored all 62 from local `.git/lfs/objects` (exact content, zero edit loss) + repaired a truncated `ball_to_goal_progress` RewTerm in `kick_env_cfg.py` (weight +1.0 per PHASE0-DONE.md). **If any file ever reads `version https://git-lfs.github.com/spec/v1`, it is corruption, not content** — recover from git history or `.git/lfs/objects/<oid[0:2]>/<oid[2:4]>/<oid>`, never edit around it.

## Verified commands (post-rebuild — placeholders until T6.1 done)

```bash
# smoke (local, any time):
<venv>/python isaac_tasks/.../train.py --task <ID> --num_envs 16 --max_iterations 2 --viz none

# real run (guarded):
./scripts/train_guard.sh "<venv>/python <train script> --task <ID> --num_envs 512 --max_iterations N"
# → wraps in systemd-run scope + watchdog; logs to logs/train_<id>.log
```

## Key files

| File | Purpose |
|---|---|
| `PLAN_PHASE6_SOCCER_HRL.md` / `TASKS.md` (T6.*) / `STATE.md` | plan / tasks / state — the contract for this phase |
| `isaac_tasks/k1_velocity/.../kick/` | kick task (migrate in T6.2.3) |
| `scripts/flatten_k1_usd.py`, `scripts/probe_k1_bodies.py` | USD flatten / body-name probe |
| `src/k1_sim_isaac/scripts/k1_soccer_compose.py` | runtime policy composition (extend in T6.6.2) |
| `src/k1_sim_isaac/scripts/{soccer_sim,stereo_cam_test}.py` | field/ball constants · stereo rig (depth source) |
| `models/*.pt` | exported policies (LFS) |

## Next steps

1. User **go-ahead on the plan** → T6.0 git closure → T6.1 env rebuild → T6.2 migration → T6.3 infra → T6.4 base policies → T6.5 HRL → T6.6 integration.
2. Every gate (G0–G6) blocks the next stage; user reviews wandb + videos every 200 iters.
