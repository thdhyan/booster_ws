# STATE — Booster K1 Workspace

> **Snapshot:** 2026-09-22 05:00 · **Branch:** `dev/phase-0` @ `dd3f6b2`
> **GitHub:** `git@github.com:thdhyan/booster_ws.git`
> Companion docs: `PLAN.md` (original), `PLAN_MULTILAYER.md` (soccer architecture),
> `PLAN_PHASE6_SOCCER_HRL.md` (current phase plan), `TASKS.md`, `HANDOFF.md`, `PHASE0-DONE.md`

---

## 1. Git

| Item | Value |
|---|---|
| Branch | `dev/phase-0` — **34 commits ahead of `main`, 0 behind** |
| Tags | none exist yet |
| Uncommitted (modified) | `ROBOTS.md`, `scripts/docker_isaac_fleet.sh`, `src/k1_sim_isaac/scripts/isaac_fleet_vis.py`, `src/k1_description/assets` (submodule untracked content) |
| Untracked | `.claude/`, `PLAN_MULTILAYER.md`, `isaac_tasks/k1_head_tracking/`, `isaac_tasks/k1_kicking/`, `src/k1_sim_isaac/scripts/k1_soccer_compose.py` |

**Pending git action (approved in plan):** commit leftovers → merge `dev/phase-0` → `main` → push → tag → open `dev/phase-6-soccer-hrl` for the new phase.

## 2. What exists and works

| Area | Status | Evidence |
|---|---|---|
| Velocity task + training | ✅ done (old env) | student reward 40.89, `models/k1_velocity_student.pt`, teacher `velocity_teacher_4999.pt` |
| Kick task `Isaac-Kick-Ball-K1-v0` | ✅ committed (`89b13a9`) | `isaac_tasks/k1_velocity/.../kick/` + `scripts/train_kick*.py` |
| USD flatten + contact rewards | ✅ committed (`faefa3e`) | `scripts/flatten_k1_usd.py`, `K1_flat.usd` (4.2 MB) |
| Fleet sim (Gazebo / MuJoCo / Isaac) | ✅ verified | HANDOFF.md table |
| Stereo head (ZED 2i replica) | ✅ | `stereo_cam_test.py` |
| RoboCup 3v3 MuJoCo capture | ✅ | `mujoco_robocup_demo.py` |
| Policy composition node (draft) | 🟡 untracked | `k1_soccer_compose.py` (needs head/kick policies) |
| Head-tracking env (draft) | 🟡 untracked, **not runnable** | `isaac_tasks/k1_head_tracking/.../head_tracking_env_cfg.py` — only one file, no `__init__.py`/agents/scripts; obs wiring uses velocity-command hack; `CurriculumCfg = CurrTerm()` is wrong type |
| Kicking env (draft) | 🟡 untracked, **not runnable** | `isaac_tasks/k1_kicking/.../kicking_env_cfg.py` — same, ball obs via `UniformVelocityCommand` hack, no scene ball physics link |

## 3. What is broken / missing right now

1. **No working Python training environment.**
   - Old `.venv-isaac` (Isaac Sim 6.0.1 + IsaacLab 3.0.0b2) — **deleted from disk**.
   - Old conda env `isaac` — deleted too.
   - Fresh `IsaacLab/isaac6/.venv` exists (uv, Python 3.12.13, created 2026-09-22 04:53) but **empty — no packages installed yet**. Its `.envrc` states it targets **Isaac Sim 6.1.0** and replaces the deleted envs.
   - IsaacLab repo checkout is on branch `perf-2026-07-06` (commit `f443e8ac5`) — **NOT** at `v3.0.0-EA`.
2. **`scripts/start_training.sh` is stale** — points at deleted `.venv-isaac` and wrong wandb entity (`thakk100`; valid entity is `thakk100-dhyan-home`).
3. **zz-bw (HPC training target) unreachable** — `ssh zz-bw` timed out (exit 124). The proven `~/run_k1_train.sh` + SIF flow cannot be used until the network recovers. `run_k1_train.sh` is not on this machine.
4. **No `k1_head_tracking` / `k1_kicking` packages** — just lone draft env-cfg files, unregistered, no agents or train scripts.
5. **IsaacLab 2.x→3.0-EA API migration not started** — see PLAN_PHASE6 §3 for the breaking-change list (quats WXYZ→XYZW, `ProxyArray`, `write_*_index/mask`, new `isaaclab train` CLI, `RecordVideo` removed → `VideoRecorderCfg`, importer now nests bodies).
6. **YOLO** — `ultralytics` not installed anywhere; no ball detector exists.

## 4. Hardware / resource budget (local box `dhyan-LOQ`)

| Resource | Value | Budget for training |
|---|---|---|
| GPU | RTX 4060 Laptop **8 GB**, ~1.7 GB used by desktop | training must stay **≤ 6.0 GB** VRAM |
| RAM | **15 GiB** total, ~5.3 GiB available now, 15 GiB swap (2.6 used) | training capped **≤ 9 GiB** RSS |
| CPU | 16 cores | `nice -n 10`, never >1 training process |
| Disk | **54 GB free** on `/` | Isaac Sim 6.1 install ≈ 15–25 GB + shader cache ≈ 3–5 GB + videos — fits, monitor |

**Hard rules for Phase 6:**
- Headless only, always.
- ONE training run at a time (sequential policy-by-policy).
- Every training process runs under a systemd scope with `MemoryMax=9G` + a watchdog that kills it if GPU > 7.0 GB or free RAM < 1 GiB → desktop never dies.
- `num_envs` capped for local box (256–512 typical; 4096 forbidden locally).

## 5. WandB

- Entity: **`thakk100-dhyan-home`** (`thakk100` / `thdhyan` are invalid).
- Existing project: `booster_k1_locomotion` (cached login present in `~/.cache/wandb`).
- Phase 6 project (planned): `booster_k1_soccer_hrl` — all 4 base policies + HRL top-level, one run per policy, videos logged as media.

## 6. Key gotchas carried forward

- Never source system ROS2 and Isaac ROS2 in the same shell.
- K1 joint A-prefix: `AAHead_yaw`, `ALeft_Shoulder_Pitch`, `ARight_Shoulder_Pitch`.
- URDF-imported USD nests link prims → contact sensors need flat hierarchy (`scripts/flatten_k1_usd.py`) — must be re-validated against the 3.0-EA importer (it now nests by design).
- rsl_rl ≥4 model schema: `actor=`/`critic=`, not `policy=`; deterministic distillation needs the `output_std` class-property guard.
- colcon: force `-DPython3_EXECUTABLE=/usr/bin/python3`.
