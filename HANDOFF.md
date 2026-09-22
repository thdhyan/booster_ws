# Booster K1 Workspace — Handoff

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
