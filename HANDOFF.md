# Booster K1 Workspace — Handoff

> **Date:** 2026-09-22 · **Branch:** `dev/phase-0` @ `dd3f6b2` (34 commits ahead of `main`)
> **GitHub:** https://github.com/thdhyan/booster_ws
> **Read first:** `STATE.md` (snapshot) → `PLAN_PHASE6_SOCCER_HRL.md` (the active plan) → `TASKS.md` T6.* (execution list)
> Supersedes the older two-agent handoff (`HANDOFF-AGENTS.md` — contact-sensor + kick-task jobs **landed**: `faefa3e`, `89b13a9`).

---

## Current State

### 🟥 BLOCKING — no training environment exists on this box
- Old `.venv-isaac` (Isaac Sim 6.0.1 + IL 3.0.0b2) and conda env `isaac` were **deleted from disk**.
- Fresh empty uv venv: `~/Projects/IsaacLab/isaac6/.venv` (py3.12, created 2026-09-22, `.envrc` targets **Isaac Sim 6.1.0**) — **zero packages installed**.
- IsaacLab checkout `~/Projects/IsaacLab` sits on branch `perf-2026-07-06` — **not** `v3.0.0-EA`. Plan: separate worktree `~/Projects/IsaacLab-ea` (shared checkout must not move; other projects use it).
- `zz-bw` HPC unreachable (ssh timeout). **Local-only training plan** it is.

### ✅ Landed (still valid)
- Velocity task trained (old env): student reward 40.89 (`models/k1_velocity_student.pt`), teacher 25.08 (`velocity_teacher_4999.pt`).
- Kick task `Isaac-Kick-Ball-K1-v0` (OmniReset, goal detection) — `89b13a9`.
- USD flatten + contact rewards re-enabled — `faefa3e` (`scripts/flatten_k1_usd.py`, `K1_flat.usd` 4.2 MB). *Contact end-to-end never verified; IL 3.0 importer may make flatten obsolete — re-check.*
- Fleet sim ×3 backends (Gazebo/MuJoCo/Isaac), stereo ZED 2i rig, RoboCup MuJoCo capture — see `PHASE0-DONE.md`, old details in git history.

### 🟡 Untracked drafts (commit with T6.0.1)
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

- **RTX 4060 8 GB / 16 GiB RAM / 54 GB disk.** One training at a time; ≤512 envs; `train_guard.sh` watchdog + `systemd-run --scope MemoryMax=9G`; GPU >7 GB ⇒ kill. Never 4096 envs locally.
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
