# TRAINING.md — Booster K1 soccer HRL campaign

**Created:** 2026-09-22. Living table of every planned training run: task ids, sizes,
estimates, checkpoint flow, status. Source of truth for run order = `PLAN_PHASE6_SOCCER_HRL.md` §5.

## Operating rules (user directives, 2026-09-22)

1. **tmux for all training** — launch via `scripts/tmux_train.sh` so runs survive quitting
   OpenCode / closing terminals.
2. **Backend = PhysX for ALL local runs** (user decision 2026-09-22, after measuring
   Newton ~2.5–3× slower @256 and PhysX→Newton checkpoint transfer exploding to NaN).
   Env var `K1_PHYSICS` (default **`physx`**; `newton` = opt-in for probes/other boxes)
   in `k1_velocity/sim_backend.py`, applied in each env cfg `__post_init__`.
   Headless via `HEADLESS=1` (auto-exported by the tmux launcher).
3. **No GT state** for deployable policies (T6.3.5 static test enforces); estimator in loop.
4. **Locked rewards** (PLAN §4) — no silent edits.

## Remote execution on dl (decided 2026-09-22, user directive)

**Why:** the laptop's 8 GB GPU + ~2.6–3.2 GB desktop baseline tripped the guard's
7000 MB cap mid-run-1 (7741 MB peak). All training moves to a server box, headless
(always), tmux, one run at a time. Target **dl** (4× RTX 6000 Ada 48 GB, 503 GB RAM,
64 cores, x86_64) once VPN routes exist — campus subnets (128.101/10.131) unreachable
from home without the tunnel; **zz-bw** (2× RTX PRO 6000 Blackwell 96 GB, no SLURM,
3 TB home) = verified fallback but both GPUs currently compute-saturated by another
user (`abrar008`). Sparks (GB10, aarch64) = Isaac Sim not supported on arm64.

**dl recipe (same paths as laptop):**
1. `git clone` booster_ws → `$HOME/Projects/booster_ws` + `git lfs pull`.
2. Clone isaac-sim/IsaacLab, `git fetch origin tag v3.0.0-EA --no-tags`,
   `git worktree add $HOME/Projects/IsaacLab-ea v3.0.0-EA` (tag = `ae37b028e`,
   what the laptop worktree runs; no local IsaacLab patches exist — all ours live
   in booster_ws).
3. `uv sync --extra isaacsim --extra wandb --extra video --extra importers
   --extra rsl-rl` + `uv pip install ultralytics "imageio[ffmpeg]"`
   (= `scripts/install_phase6_env.sh`, ≈ 30 GB incl. Isaac Sim 6.1.0 pip wheels).
4. rsync resume checkpoint `model_1500.pt` from laptop.
5. Launch via `scripts/tmux_train.sh` (exports `HEADLESS=1`; tmux server auto-started
   in its own systemd scope) with server guard env:

   | var | laptop default | dl value |
   |---|---|---|
   | `CUDA_VISIBLE_DEVICES` | (all) | chosen idle GPU (e.g. `2`) |
   | `GPU_IDX` (guard sample) | `0` | same index as `CUDA_VISIBLE_DEVICES` |
   | `GPU_MAX_MB` | `7000` | e.g. `46000` (48 GB card − headroom) |
   | `MEM_MAX_GB` / `SWAP_MAX_GB` | `9` / `4` | e.g. `64` / `64` |
   | `DISK_PATH` | `/` | `$HOME` (NVMe) |

   `GPU_IDX` MUST match `CUDA_VISIBLE_DEVICES` or the guard watches the wrong GPU.
6. `WANDB_API_KEY` exported at launch (same entity/project).

**dl active since 2026-09-22 11:55 (laptop clock)** — `scripts/setup_remote.sh` brought
the box up in 3 idempotent iterations (https rewrite for public submodules on a keyless
host; checkout `dev/phase-6-soccer-hrl` since `main` lacks phase-6 scripts; reuse the
pre-existing clean `~/Projects/IsaacLab` @ beta2 → fetch `v3.0.0-EA` tag → worktree
`ae37b028e`). uv venv + import sanity OK (`isaaclab`/`rsl_rl`/`ultralytics`).
`tmux_train.sh` now snapshots caller env into `logs/tmux_env_<name>` (600) — tmux
sessions inherit the SERVER's env, not the ssh client's. dl joins the **pre-existing**
tmux server (user's `g1`/`isaac-jupyter-*`); the laptop's cgroup/restart hazard (gotcha 4)
is OpenCode-local and does not apply to remote servers.

**Parallel scheme (user directive 2026-09-22):** independent runs on SEPARATE dl GPUs,
all in tmux, **no client-side polling** — wandb is the dashboard. Single-run guard lock
became per-run (`k1_train_guard_<name>.lock`) to allow this. `chain_train.sh` waits for
a session server-side and launches the next run (newest `model_*.pt` picked after the
wait). GDrive: rclone remote `gdrive-dhyan` mirrors `logs/` (guard logs, checkpoints,
videos) to `gdrive-dhyan:Booster/logs` every 10 min from tmux session `k1_gdrive_sync`
(wandb key `logs/.wandb_key` excluded, chmod 600).

| GPU | run | session | envs × iters |
|---|---|---|---|
| GPU1 | P1 teacher finish → **extension** (chained) | `k1_p1_teacher_dl` → `k1_p1_chain` → `k1_p1_teacher_dl_ext` | 256×2000 (running) → 512×3000 from newest ckpt |
| GPU2 | P2 velocity teacher (fresh) | `k1_p2_teacher_dl` | 512×3000 |
| GPU3 | ⚠️ not viable for our runs (20 GiB resident by neighbors vs 30 GiB guard cap) | — | — |
| GPU0 | kick smoke ✅ PASSED (16×2) → **P4 teacher** queued behind P3 | `k1_p4_kick_smoke` → planned | 16×2 → 512×3000 |
| GPU2 (frees when P2 ends) / GPU0 | **P3 head-track** — launches right after T6.3.x authoring | planned | 512×2000 |

Video interval per size: 256 envs → 4800; 512 envs → 2 457 600 env-steps (= every
200 iters). `max_iterations` counts FRESH iterations per launch (resume restarts the
counter: the 09:57-derived dl run does a full 0→2000 block from `model_1500`).

**Wall-time estimates below are laptop-measured PhysX**; dl measured ≈ **1.45 s/iter @256
with video, 4353 steps/s, GPU util 40 %, 6.5 GB VRAM** — update table rows as runs land.

**Run-1 handoff to dl:** resume point = `model_1500.pt` (the 09:57 window saved only
the start copy before tripping the guard at iter ~1574 — its ~74 thrashing iterations
are discarded; rewards were sane: mean −16, all 9 locked rewards logged). Remaining
≈ 430–500 iters × 256 envs.

## Launch / monitor cheat-sheet

```bash
# launch (session k1_<name>, done marker logs/tmux_<name>.done)
# tmux_train.sh exports HEADLESS=1 automatically — no Kit window ever opens
# (pass HEADLESS=0 to opt out and watch the window).
scripts/tmux_train.sh --name p1_teacher_newton -- isaaclab train --rl_library rsl_rl \
  --task <gym-id> --external_callback k1_velocity.register_tasks.register_tasks \
  --num_envs N --max_iterations M --seed 42 \
  [--checkpoint <path/to/model_K.pt>] \
  --headless --enable_cameras --video --video_length 1500 --video_interval 4800

# monitor
tmux attach -t k1_<name>        # detach: Ctrl-b d
tmux capture-pane -t k1_<name> -p | tail -30
tail -f logs/guard_<name>.log
```

- Video: `--video_length 1500` = 30 s @50 Hz; `--video_interval 4800` = every 200 iterations
  (24 env-steps/iter). Clips: `logs/rsl_rl/<wandb-run>/<timestamp>/videos/train/clip_*.mp4`,
  auto-uploaded to wandb as `wandb.Video`.
- One run at a time (guard lock `/tmp/k1_train_guard.lock`). Guard: GPU total < 7000 MB
  (desktop baseline ≈ 1.5 GB → process budget ≈ 5.5 GB), MemAvailable > 1024 MB, disk > 5 GB,
  3 consecutive 5 s samples before kill.
- 4096 envs forbidden locally; ≤ 512 envs. Heavy RL box = remote (currently unreachable).

## Campaign table (9 runs)

Wall-time estimates: **PhysX measured baseline** — 256 envs = 1.59–1.63 s/iter fresh
(≈ 2.1–2.5 s/iter with video capture + swap pressure); startup ≈ 2–3 min.
512-env tasks scale ≈ 1.6–2× per iteration + YOLO-env overhead where noted.
**Newton measured (2026-09-22):** 256 envs = **4.8–5.0 s/iter** steady (24 s first-iter
warp JIT), startup ≈ 2–4 min → 2000 it ≈ 2.8–3 h — **~2.5–3× SLOWER than PhysX on this
RTX 4060**, despite headless. 16 envs = 3.6 s/iter. Newton-at-256 = stable with a fresh
policy (probe `g3_newton_256_fresh` rc=0, 0 NaN) but **PhysX→Newton checkpoint transfer
explodes to NaN within 1 iteration** (run-1 resume attempt, 2026-09-22 09:37).

| # | Run | PLAN task id | actual gym id (registered) | envs | iters | backend | est. wall time | ckpt in | status |
|---|-----|--------------|----------------------------|------|-------|---------|----------------|---------|--------|
| 1 | P1 teacher | `Isaac-Basic-Teacher-K1-v0` | `Isaac-Basic-Teacher-K1-v0` | 256 | 2000 | physx | PhysX dl actual ≈ 46 min finish + ~68 min ext | `model_1500.pt` (physx runs 0→~1548) | **DONE 2026-09-22** — finish run GUARD_RC=0 (2000 it → model_3500) auto-chained ext GUARD_RC=0 (512 envs, 3000 it) → **best `model_6498.pt`** (`p1_basic_teacher/2026-09-22_13-41-33_p1_basic_teacher`); peak mean reward **−0.27** (single-iter spike), settles −16/−17; finish-run clips non-black (mean 89 / std 53) |
| 2 | P1 student | `Isaac-Basic-Student-K1-v0` | `Isaac-Basic-Student-K1-v0` via Path B `train_student.py --teacher_checkpoint` (EA `isaaclab train` distill strict-loads the PPO ckpt → `std_param` crash) | 256 | 1500 | physx | PhysX ≈ 45–60 min | #1 final `model_6498.pt` | **queue armed 2026-09-23** — `tmux k1_p1_student_queue` (GPU1): wait GPU <10 GB → 16×3 smoke → `GUARD_RC=0` marker gate → full run; auto-starts when tang0836's 8-proc job (33 GB/GPU) releases the GPU |
| 3 | P2 teacher | `Isaac-Move-Teacher-K1-v0` | `Isaac-Velocity-Rough-K1-Teacher-v0` | 512 | 3000 | physx | PhysX ≈ 2–3 h / dl actual ≈ 73 min | — | **DONE 2026-09-22 14:27 dl** — GUARD_RC=0, 3000 it → **`model_2999.pt`** (`k1_velocity_teacher/2026-09-22_13-14-06`), reward −2.94 → peak **16.22** → final 14.0; logs+ckpt synced to gdrive; ⚠ wandb run keeps legacy name `k1_velocity_teacher` (cfg = `p2_move_teacher`) |
| 4 | P2 student | `Isaac-Move-Student-K1-v0` | `Isaac-Velocity-Distill-K1-v0` via Path B `train_student.py --teacher_checkpoint` | 512 | 3000 | physx | PhysX ≈ 2–3 h | #3 final `model_2999.pt` | **queue armed 2026-09-23** — `tmux k1_p2_student_queue` (GPU2): wait GPU <10 GB → smoke → marker gate → full run; runs in parallel with #2 |
| 5 | P3 head-track | `Isaac-HeadTrack-K1-v0` | *not yet authored* (T6.3.3) | 512 (64 YOLO+vid) | 2000 | physx | PhysX ≈ 2–3 h | — | blocked: task authored just before this run → **authoring NOW** (T6.3.1→3.4 first); frozen legs = P1 teacher final; target GPU2 when P2 ends |
| 6 | P4 teacher | `Isaac-Kick-Teacher-K1-v0` | `Isaac-Kick-Ball-K1-Teacher-v0` | 512 (4 vid) | 3000 | physx | PhysX ≈ 2.5–3.5 h | — | kick smoke **PASSED** 2026-09-22 13:19 dl (16×2, GPU0, GUARD_RC=0, clip non-black → T6.2.3 closed); teacher **gated on T6.3.3 YOLO + trained P3 head (PLAN §P4/T6.4.6) + obs-53/goal env upgrade — will NOT train the blind env (retrain guaranteed)** |
| 7 | P4 student | `Isaac-Kick-Student-K1-v0` | `Isaac-Kick-Ball-K1-Distill-v0` `--distill` | 512 (64 YOLO+vid) | 3000 | physx | PhysX ≈ 2.5–3.5 h | #6 final | queued (Gate G5) |
| 8 | P5 teacher | `Isaac-HRL-Teacher-K1-v0` | *not yet authored* (T6.3.4) | 256 (32 YOLO+vid) | 2000 | physx | PhysX ≈ 1–1.5 h | #1–#7 | blocked: T6.3.4 |
| 9 | P5 student | `Isaac-HRL-Student-K1-v0` | *not yet authored* (T6.3.4) | 256 | 1500 | physx | PhysX ≈ 50–70 min | #8 final | queued (Gate G6) |

  **Totals (PhysX for all runs — user decision 2026-09-22):** ≈ **15–21 h sequential**.
Newton measured slower on this box (4.8–5.0 vs 1.6–2.1 s/iter @256) and can't resume
PhysX checkpoints → retained only as `K1_PHYSICS=newton` opt-in. Each run preceded by a
16-env smoke (gate); iter counts are ceilings — stop early on plateau; user reviews
videos every 200 iters.

### Notes

- Run 1 checkpoint chain: `2026-09-22_08-19-32` (PhysX, 0→~1100, SIGKILL at 1147 — no
  cgroup/OOM trace, swap 9.2 GB) → `2026-09-22_08-59-01` (PhysX resume, 1100→~1548) →
  newton resume from `model_1500.pt` **CRASHED: NaN obs @iter 1 (PhysX→Newton policy
  transfer explodes; fresh-policy probe proved scale@256 is fine)** → **PhysX headless
  finish from `model_1500.pt`** (tmux `k1_p1_teacher_physx_finish`, 09:57) → target
  `max_iterations 2000` absolute (rsl_rl counts from the checkpoint's iteration, so
  `--max_iterations 2000` = finish the campaign).
- Newton feasibility verified statically (2026-09-22): `NewtonCfg` selection pxr-free for all
  3 cfg families; `isaaclab_newton` has contact_sensor + ray_caster factories; custom
  BoosterDelayedPDActuator forced through Isaac Lab path (`use_newton_actuators=False`).
  Runtime = G3 smoke (`k1_g3_newton_smoke`, 16×3) before committing run 1.
- **Newton bring-up gotchas (G3 smoke, 2026-09-22):**
  1. `ContactSensorCfg.filter_prim_paths_expr=["/World/ground"]` crashes Newton init
     (`No bodies matched the counterpart pattern(s)` — ground plane isn't a body label
     in the Newton model). Fix: `sim_backend._clear_contact_filters` clears sensor
     filters under Newton only. Semantically safe: all consumers read **net** forces +
     air/contact timers (unfiltered in PhysX too); `filtered_forces`/`force_matrix`/
     `contact_pos` are never read in k1_velocity (verified).
  2. Scene cfg field `class_type: ResolvableString` has a lazy `__getattr__` that
     resolves the path template **and imports 43 pxr modules** — a plain
     `getattr(cfg, 'filter_prim_paths_expr', None)` walk over the scene would
     re-introduce the G2 Kit-poisoning pxr import pre-SimulationApp. Fix: read sensor
     fields from the instance `__dict__` directly (bypasses `__getattr__`).
     Static check after fix: newton → `NewtonCfg` + filter `[]` + pxr 0;
     physx → `NoneType` + filter kept + pxr 0.
  3. **`--headless` CLI arg is NOT honored by the unified `train` backend** (the
     export path forces `args_cli.headless = True` programmatically; train does not
     parse/forward it). Every run — including the morning PhysX runs — was actually
     **windowed**: the "Isaac Lab 3.0.0" window on the desktop was Isaac Lab's env UI
     (`base_env_window.py:73 "Creating window for environment."`). Fix: `HEADLESS=1`
     env var (AppLauncher `app_launcher.py:890`), now exported by
     `scripts/tmux_train.sh`. Verified: headless smoke log = 0 window lines,
     no Kit window at exit, clips still non-black + motion.
   4. **A plain detached `tmux` server is NOT restart-proof here**: the server
      inherits OpenCode's cgroup (`vte-spawn-*.scope`), so an OpenCode server restart
      kills the server → sessions die → training dies mid-run (observed 2026-09-22
      09:51: killed at iter ~1502, stale lock left behind — clear with
      `rmdir /tmp/k1_train_guard.lock`). Fix in `tmux_train.sh`: if no server is
      running, start one via `systemd-run --user --scope tmux new-session -d -s
      k1_tmux_holder 'while :; do sleep 3600; done'` → server lives in its own
      `run-*.scope`, sessions in `tmux-spawn-*.scope`, both outside OpenCode's
      cgroup (verified via `/proc/<pid>/cgroup`; 09:57 relaunch survived).
- **G3 smoke PASSED (2026-09-22 — two runs, rc=0, windowed then headless):**
  banner `Physics newton_mjwarp`; `Registered backend 'newton'` for Articulation +
  **ContactSensor** + **RayCaster**; 9 locked rewards logged; 2 clips/run written
  (32 frames, 640×360, first-vs-last frame diff 2.7 = motion; cubric warning is
  precautionary — frames NOT black); iter time @16 envs ≈ 3.6 s steady
  (first iter 7.7 s = warp JIT); startup ≈ 46–59 s; no guard LIMIT HIT.
- Env counts: 512-env rows planned per PLAN; actual envs capped by GPU guard (256 = ~6.3 GB
  device peak under PhysX+video). Bump only if measured peak leaves headroom — record here:
  - P1 (256): device peak 6315 MiB (PhysX+RTX, video every 200 it) → at cap, stay 256.
- Registry: 11 K1 gym ids registered (basic T/S, velocity rough/distill/play + teacher,
  kick ball base/teacher/distill). PLAN ids renamed → actual id column above (aliases
  `Move`/`HeadTrack`/`HRL` may be added later for PLAN parity).
