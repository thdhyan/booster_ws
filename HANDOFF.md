# Booster K1 Workspace — Handoff

## 📌 SESSION HANDOFF — 2026-09-24 (read this first; older history below)

> **Branches:** `dev/soccer-p3p4` (active — holds BOTH tracks; Track A gates landed) ·
> `dev/phase-6-soccer-hrl` @ `f3000d6` (pushed — policy recordings, docs, videos, exports).
> **GitHub:** https://github.com/thdhyan/booster_ws · **Plan:** `PLAN_PHASE6_SOCCER_HRL.md` · **Run log:** `TRAINING.md`
> **Drive:** [policy videos folder](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy) ·
> [Slide deck (4 videos embedded)](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit)

---

# 🟠 TRACK A — SOCCER (P3 head-track, P4 kick teacher) — *agent A owns this*

**Goal:** make the K1 see and kick a ball without ground truth in its inputs
(P3 keeps the ball in frame from camera detections; P4 teacher knows the ball;
the student will consume P3 + the vision estimator).

**Code map (all in `isaac_tasks/k1_velocity/source/k1_velocity/`):**
- `tasks/head/` — P3 `Isaac-HeadTrack-K1-v0` (head-only; obs 12 = YOLO
  detection(3)+head(4)+ang-vel(3)+action(2); centredness reward; ball-speed
  curriculum 0→0.8 m/s; CCW search helper), `tasks/head/agents/rsl_rl_ppo_cfg.py`
  (wandb `p3_head_track`).
- `tasks/kick/` — P4. Teacher id now drives head+legs (14-dim) with
  `ball_in_frame` + `head_ball_aim` (`K1KickTeacherEnvCfg`). Student id unchanged.
- Launchers: `scripts/spark_soccer_{host,container}.sh p3|p4t` (smoke-gated,
  tmux `k1_spark_p3`/`k1_spark_p4t`, logs `scripts/p3.soccer.log`/`p4t.soccer.log`).
- Smokes/verifiers: `scripts/smoke_p3_head.sh`, `scripts/smoke_p4_teacher.sh`,
  `scripts/zero_step.sh Isaac-HeadTrack-K1-v0 cameras` (zero-agent, ~5 min).

**Status (2026-09-24 15:03 UTC):**
- Deployability gate is green for **16 tasks**. P3 zero-step and the real
  camera+YOLO 16×3 smoke both pass; the smoke loaded `yolov8n`, exercised
  periodic resets, produced `model_2.pt` and three non-black clips.
- Fixed the P3 runtime blocker: `write_root_velocity_to_sim_index` requires
  linear+angular velocity `(N,6)` in this build. Also gave P3/P4 separate Isaac
  caches (the shared cache could hang OmniHub), switched to `--viz none`, and
  added log-marker gates because `python.sh` can return 0 after a crash.
- **P3 full is TRAINING on spark02**: tmux `k1_spark_p3`, 512 envs × 2000,
  log `scripts/p3.full.train.log`, run
  `logs/rsl_rl/p3_head_track/2026-09-24_14-58-47_p3_head_track/`, W&B
  [`oavkxiwf`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/oavkxiwf).
  Initial rate is ≈33–41 s/iter (full ETA ≈19 h; hard timeout 26 h); steady
  process memory ≈10 GB. Inference is 10 Hz, full-batch, FP16, 320 px.
  Videos are every 6400 control steps = 200 iterations.
- **P4 teacher 16×3 smoke PASSED** (`P4T_SMOKE_RC=0`,
  `P4T_SMOKE_MARKER=OK`): teacher obs `(55,)`, legs+head action `(14,)`,
  `model_2.pt` + video, W&B smoke
  [`b4f5ag2d`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/b4f5ag2d).
  Fixed per-environment goal replication/scoring, added teacher goal obs and
  the locked approach/kick/align shaping terms.
- Server-side tmux `k1_soccer_chain` waits for P3's explicit
  `SOC_FULL_MARKER=OK`, then runs `spark_soccer_host.sh p4t` (fresh smoke-gated
  512×3000 full). It aborts P4 if P3 lacks the final marker.
- Run-11 remains independently training on spark02 in tmux `k1_spark_hp`.

**Agent A — next steps (in order):**
1. Monitor without restarting healthy runs:
   ```bash
   ssh aim_spark02 'cd ~/Projects/booster_ws && grep -E "Learning iteration|Mean reward|SOC_FULL_MARKER|Traceback|Error" scripts/{p3,p4t}.full.train.log | tail -20'
   ```
2. Require final markers, not just process exit: P3 `SOC_FULL_MARKER=OK` plus
   `model_1999.pt`; P4T marker plus `model_2999.pt`. If either run dies, relaunch
   with the newest checkpoint via `scripts/train.py --checkpoint ...`.
3. Pull representative P3/P4 clips, play-record final checkpoints using the
   Track B recipe, mirror them to Drive, and add slides to the existing deck.
4. Replace these running/queued statuses with final W&B/reward/video evidence in
   `TRAINING.md` and this handoff.

**Watch-outs:** `WANDB_API_KEY` comes from `logs/.wandb_key`; ultralytics gets
a **list of HWC uint8 frames**; rigid-body root velocity is `(N,6)`; reward
terms return 1-D `(N,)`; P3 smoke/full share no Isaac cache with P4 or Run-11.

---

# 🟢 TRACK B — BOX PUSHING (P6: hierarchical velocity + wrist-IK controller) — *agent B owns this*

**Goal:** a policy that walks the K1 up to a box, places both wrist stubs on it
and pushes its corners to a moving goal set. Hierarchy: **velocity command
override → frozen Run-11 locomotion policy → legs**; **wrist-stub targets →
DifferentialIK → arms**.

**Code map:** `tasks/push/` — `push_env_cfg.py` (`K1PushEnvCfg`,
`K1PushReachEnvCfg`), `push_mdp.py` (frozen-base action term, wrist command,
corner/goal math, USD DR, green alpha), `agents/rsl_rl_ppo_cfg.py`
(wandb `p6_push_reach` → `p6_push`). Ids: `Isaac-Push-Reach-K1-v0`,
`Isaac-Push-K1-v0`. Frozen base: `models/k1_partialctrl_base.pt` (exported from
Run-10 `model_2999`, JIT parity 2.87e-03; present on local + spark02 + spark04;
re-export with `scripts/export_base_policy.sh` if Run-11 supersedes it).

**Design (locked):** action 9 = `[vel override 3 | left wrist EE delta 3 | right 3]`;
frozen base assembled with the exact 68-dim partial obs; IK `body_name=*_hand_link`,
`scale=0.05`, dls, position/relative; box = 1 m prototype with USD-time per-env DR
(edge 0.7–1.5 m, mass 3–25 kg / 12–25 for Reach, friction 0.3–1.2,
`replicate_physics=False`), green alpha 0.35; goals = box pose + cumulative
offset integrator (curriculum 0.3→1.5 m), rewards on 8 corners + corner
centroid + progress; teacher-group obs (fully observable by design).

**Status (2026-09-24 15:40 UTC) — TRACK B GREEN:**
- **Zero-step OK:** `ZERO_STEP_RESULT=OK` (3 steps, obs `teacher (n,108)`, rewards
  compute, no terminations; box rests at `z = half_extents`, DR prints
  `edge 0.71-1.47 m, mass 3.6-19.2 kg`). Gate log `scripts/pushzero.log`.
- **Smoke OK:** `scripts/smoke_push.sh push` 16×3 reached
  `Learning iteration 2/3`, video written, no Traceback. The last blocker was a
  **partial-reset broadcast bug**: `reset_wrist_targets` builds an
  `env_ids`-subset `(n,2,3)` target but `_to_base` subtracted the **full-batch**
  `root_pos_w` (4 envs never subset → zero-step hid it; 16-env smoke crashed at
  the first 2-env reset: `size of tensor a (2) must match (16)`). Fix:
  `_to_base(env, pts, env_ids=None)` indexes `root_pos_w`/`root_quat_w` by
  `env_ids`. Earlier rounds also fixed: `quat_apply` batching (`_rot_batch`),
  `goal_quat_bf` (`quat_mul(quat_inv(base),box)`), event modes (`prestartup`
  for USD geometry, `startup` for friction — material impls need `root_view`),
  USD `MassAPI` + `Gf.Vec3f` inertia DR, `stage.GetPrimAtPath`,
  `UsdShade.Material.Define` 2-arg, warp `joint_ids` int32,
  `_update_command()` no-args, `reset_box` z=half-extents, box-frame
  `wrist_box_proximity`, `find_joints` lists→tensor.
- **Deliverables shipped** (commit `6f7db13`, pushed `dev/soccer-p3p4`):
  README **P6 section** (obs/action/reward/DR tables) + `videos/push_smoke.mp4`
  + 2 PNG frames; TRAINING.md campaign rows 10–11; launchers
  `scripts/spark_push_{host,container}.sh`.
- **Drive:** `push_smoke.mp4` → folder
  [Booster videos](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy)
  (anyone/reader, file id `14lbLlTdANq6qNy1vGc5DcGoSJWjY4I-j`); deck slide
  added with `createVideo` source `DRIVE` (slide `slide_fe11a147`, Composio
  session `trip`).
- **Run-10 (P6 Reach 256×1500) RUNNING on spark04** — tmux `k1_spark_push_reach`,
  internal smoke gate green → full run at `Learning iteration N/1500`, log
  `scripts/reach.push.log`, wandb `p6_push_reach`.

**Remaining:**
1. When `PUSH_FULL_MARKER=OK` in `scripts/reach.push.log`, launch stage 2:
   `ssh aim_spark04 '~/Projects/booster_ws/scripts/spark_push_host.sh push'`
   (auto-picks latest `p6_push_reach` model via `--checkpoint`, gates on its own
   16×3 smoke, then 256×3000 with `PUSH_FULL_MARKER` check).
2. Monitor: `tmux ls` / `tail scripts/{reach,push}.push.log` on spark04.

**Loop recipe (used):** patch → `rsync …/tasks/push/ aim_spark04:…/tasks/push/`
(with `--no-owner --no-group --no-perms --exclude='__pycache__'`) → relaunch
`k1_pushzero` → grep markers (never rc: `PUSH_SMOKE_RC=0` prints even on
Traceback). Full smoke `k1_pushsmoke` only after zero-step OK.

---

# 🔧 SHARED INFRASTRUCTURE (both tracks)

- **Registration chain (a new family must touch all 4):**
  `tasks/__init__.py`, `source/k1_velocity/register_tasks.py`,
  `scripts/train.py`, `scripts/play_record.py` (+ `tests/test_deployability.py`
  conventions: obs group classes end in `Cfg`; `TeacherCfg` is the privileged
  exemption). Gate: container on spark04/02 —
  `python tests/test_deployability.py` → currently **HOLDS, 14 tasks**.
- **Container:** `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`, in-container
  python `/isaac-sim/python.sh`, repo mounted at `/workspace/booster_ws`, wandb
  project `booster_k1_soccer_hrl` (entity `thakk100-dhyan-home`).
- **Drive video mirror:** rclone linux-arm64 + `gdrive-thakk100` remote, tmux
  `k1_gdrive_sync` on spark02 AND spark04 (copies `logs/**/*.mp4` every 10 min).
  Config lives in `~/.config/rclone/rclone.conf` on both.
- **Server restart lesson (2026-09-24 ~12:55 UTC):** both sparks rebooted and
  killed every container/tmux session. Only `k1_gdrive_sync` survived. Salvage
  came from per-100-iter checkpoints + the new `train.py --checkpoint` resume
  flag (`RESUME_CKPT` env in `spark_force_*`). After any restart: list
  `logs/rsl_rl/*/*/model_*.pt`, then relaunch with `--checkpoint`.
  Run-10 had already finished cleanly (`HP_FULL_RC=0`).
- **Gotchas (full list was in the previous revision — top ones):**
  - `python.sh` returns rc=0 on crashes → gate on log markers/artifacts
    (`ZERO_STEP_RESULT=OK`, `REC_VERIFY_OK_*`), never rc.
  - Reward terms: 1-D `(N,)`. `(N,1)` broadcasts to `(N,N)` and the reward
    manager traceback does NOT name the term — guard or bisect.
  - Event/curriculum term signatures: `(env, env_ids, …)`; all names must be
    passed in cfg params (defaults make the validation fail).
  - `joint_pos[:, ids][:, 0]`; `joint_pos[:, ids, 0]` silently gives `(N, 2)`.
  - `randomize_rigid_body_scale`/mass/material are USD-cooked: `mode="usd"`,
    `replicate_physics=False`, per-env fixed.
  - ultralytics: pass a list of HWC uint8 frames, not a tensor.
  - `DifferentialInverseKinematicsActionCfg(body_name=...)`, not `body`.
  - configclass `class_type` must be set at decoration time.
  - Sparks: container files are root-owned; rsync with
    `--no-owner --no-group --no-perms --exclude='__pycache__'`.
    Occasional `Could not resolve hostname aim_sparkNN` → retry.
  - Never `git add src/k1_description/assets`.
  - tmux launchers that CREATE their own session (`spark_*_host.sh`,
    `record_policies_host.sh`) must be invoked directly, not wrapped in another
    `tmux new-session` (duplicate-session error).

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
