# Video → Motion on the Booster K1 — Implementation Plan

Date: 2026-09-27 · Status: plan (research complete, not yet implemented)
Scope: retarget recorded human motion (Makarena dance, arbitrary video) onto the
Booster K1, first in simulation, then on hardware.

---

## 1. Landscape — what each repo actually is

| Repo | Layer | Input → Output | K1 support | Verdict for us |
|---|---|---|---|---|
| **[GVHMR](https://github.com/zju3dv/GVHMR)** (zju3dv, SIGGRAPH Asia'24) | Video → 3D human | monocular RGB → SMPL-X params, **world-frame** (gravity-aligned, no drift) | N/A (human) | ✅ **Already installed locally** |
| **[GMR](https://github.com/YanjieZe/GMR)** (ICRA 2026) | 3D human → robot joints | SMPL-X/BVH/FBX → robot q via **mink + MuJoCo** IK | ✅ `booster_k1` built in | ✅ **Already installed locally**, K1 IK config exists |
| **[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)** (NVlabs) | BVH → robot CSV | SOMA BVH → root pose + joints CSV, **Newton/Warp**, foot-plant + contact | ⚠️ bundles `booster_t1`, **not K1** | 🔶 Best-in-class tooling; needs K1 config port |
| **[MotionBricks](https://nvlabs.github.io/motionbricks/)** (NVlabs, SIGGRAPH'26) | *generation* (not retargeting) | prompt → 15,000 FPS motion, 350k-clip backbone | ❌ G1 only (preview release) | 🔶 Long-term; wrong tool for a single dance clip |
| **Booster `booster_assets`** | data | retargeted motion **CSV** + `K1_JOINT_NAMES` | ✅ native | ✅ **The output format we must match** |
| **Booster `booster_train`** | RL tasks | BeyondMimic motion **tracking** tasks | ✅ `Booster-K1-MJ_Dance_002-v0` etc. | ✅ **The physically-feasible path** |
| **Booster `booster_deploy`** | sim2real | trained policy → real robot | ✅ | ✅ reuse for deployment |
| **Booster `booster_robotics_sdk_ros2`** | ROS2 SDK | official ROS2 wrapper | ✅ | ⚠️ **We hand-rolled a C++ bridge; reconsider** |

### Already on this machine (do not reinstall)

```
~/Projects/Thesis/workin_ws/
  third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_k1.json   ← K1 IK config
  src/gvhmr_ros/    → pub park_interfaces/GVHMRResult   (video → SMPL-X)
  src/gmr_ros/      → sub GVHMRResult, pub park_interfaces/GMRResult (SMPL-X → K1 q)
  src/twist_ros/    → whole-body teleop (the TWIST paper's runtime)
  src/park_interfaces/msg/GMRResult.msg
```

`GMRResult` already carries exactly what we need:
```
std_msgs/Header header
uint32 frame_id, float32 fps
float32[3] root_position
float32[4] root_orientation        (w,x,y,z)
string[]   joint_names             ← already name-tagged, no ambiguity
float32[]  joint_positions         ← radians
string     robot_type
float32    human_height
float32    retarget_time_ms
bool       is_valid
```

Booster's own motion CSV (what `booster_train` consumes) — **the contract we must hit**:
```
col 0-2   : root x, y, z
col 3-6   : root quat x, y, z, w     ← NOTE: xyzw, not wxyz
col 7-28  : 22 joint positions (rad) in K1_JOINT_NAMES order
sampled at 50 Hz
```
> ⚠️ **Quaternion order mismatch** — GMRResult publishes `wxyz`, booster CSV uses `xyzw`.
> The bridge node must reorder. Easy to get wrong; assert it in code.

---

## 2. The central decision: replay vs. RL tracking

There are exactly two ways to get a dance onto the robot, and they are not
equivalent. This is the decision that determines the whole project.

```
PATH A — KINEMATIC REPLAY  (fast, days, no learning)
═══════════════════════════════════════════════════
  video ──GVHMR──► SMPL-X ──GMR──► K1 joint targets ──► SDK ──► ROBOT
                                          │
                                          └─ open-loop, no balance feedback

  + works immediately, no training
  + exact pose reproduction
  − ROBOT FALLS on anything dynamic
  − no foot-force feedback → feet slip, float, penetrate floor
  − cannot recover from any perturbation
  ✅ VALID FOR: slow, small-amplitude upper-body motion (arm wave, head turn)
  ❌ INVALID FOR: Macarena dance, any step, any turn, any kick


PATH B — RL MOTION TRACKING  (slow, days of GPU, robust)
══════════════════════════════════════════════════════
  video ──GVHMR──► SMPL-X ──GMR──► reference q(t)
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  BeyondMimic tracker │  learns to follow q(t)
                            │  PPO, 4096 envs      │  while staying balanced
                            └──────────┬───────────┘
                                       │  trained policy
                                       ▼
                                  SDK ──► ROBOT

  + dynamically feasible — policy learns real balance
  + recovers from pushes, slips, mis-estimates
  + full 22 DoF (arms + head + legs) expressive
  − GPU-days of training
  − needs its own sim validation loop
  ✅ VALID FOR: dance, RoboCup, anything with steps
```

**Recommendation: Path B for the dance.** Booster already ships the BeyondMimic
tasks and two K1 dance/fight motions, so the scaffolding is done — we are adding
data, not a framework.

### A note on DoF (ties back to `docs/policy_io_reference.md`)

| Policy | Action DoF | Can it dance? |
|---|---|---|
| P1 / P2 velocity | 12 (legs) | ❌ no arms |
| Partial control | 14 (legs + head) | ❌ no arms |
| **BeyondMimic tracker** | **22 (full body)** | ✅ |

A dance needs arms. That is a *second* reason Path A fails for it and Path B is
required: our locomotion policies physically cannot express a Macarena arm pose,
and the WBC that would supply the arms has no knowledge of the choreography.

---

## 3. Staged plan

### Phase 0 — Recon & tooling audit  (0.5 day)
- [ ] **Re-evaluate `booster_robotics_sdk_ros2`.** We hand-wrote
      `src/k1_control/k1_control/sdk_bridge_node.cpp`; the official repo may
      replace it outright. If it covers LowCmd/LowState/IMU/odom, delete ours.
      If not, keep ours. *Do this before building anything on top of the bridge.*
- [ ] Verify the Macarena video: single clearly-visible full body, static or
      tripod camera, 5–10 s, no occlusion. If the camera moves, GVHMR's
      visual-odometry stage is needed (`-s` skips it — do **not** skip).
- [ ] Check GMR runs headless on this laptop and measure retarget latency
      (target < 20 ms/frame for 50 Hz).

### Phase 1 — Offline retarget, sim replay (2 days)  ← **the fast win**
Get *a picture* of the dance on the robot before any training.

- [ ] Run the existing pipeline unchanged:
      ```bash
      # in ~/Projects/Thesis/workin_ws
      python src/gvhmr_ros/... # or scripts/demo_folder.py on the video dir
      ```
      then `gmr_node` with `robot_type: "booster_k1"` (currently `unitree_g1`
      in `gmr_config.yaml` — **must change**).
- [ ] Write `scripts/gmr_pkl_to_csv.py` → emit the booster CSV contract
      (7 root cols + 22 joints, 50 Hz, `xyzw`).
- [ ] Write `src/k1_sim_gazebo/scripts/motion_replay_node.py`:
      sub a `k1_interfaces`-style motion topic (or CSV reader),
      pub `/{ns}/joint_commands` + root pose, interpolate 50→200 Hz.
      **Reuse the existing `/{ns}/joint_commands` plumbing — no new interfaces.**
- [ ] Visualise in MuJoCo first (fastest, no ROS): `mujoco_fleet_node.py` +
      replay. Then Gazebo. Then Isaac.
- [ ] **Record debug video + extract a frame and look at it** (per workspace rule).
- [ ] ⚠️ **K1 ankle caveat:** GMR's `smplx_to_k1.json` was authored for the
      `k1_serial.xml` model. Our K1 uses a **4-bar ankle linkage**
      (`CrankUp`/`CrankDown` — see `docs/policy_io_reference.md`), which the IK
      config does not model. Foot orientation will be approximate. Budget time
      to hand-tune the ankle entries, or accept foot-roll error on flat ground.

**Exit criterion:** Macarena dance plays back on the K1 in MuJoCo, feet
approximately planted. Expect visible foot slide/float — that is expected and is
exactly what Phase 2 fixes.

### Phase 2 — Make the motion trackable (1–2 days)
Feed the physics engine, not just the renderer.

- [ ] Convert CSV → npz with Booster's own tool:
      ```bash
      python isaac_tasks/booster_train_ref/source/booster_train/scripts/csv_to_npz.py \
        --headless --input_file=<motion>.csv --input_fps=50 --output_name=<motion>.npz
      ```
- [ ] Register a new task beside the existing ones in `booster_train`:
      `Booster-K1-Makarena-v0` + `-Play`, mirroring
      `Booster-K1-MJ_Dance_002-v0`.
- [ ] **Sanity-check trackability offline before spending GPU-days:** open the
      Play task and look at the reference vs. a PD-held robot. If the robot
      double-support fraction is < ~40 %, or foot velocity is unbounded, the
      retarget is not physically reachable → go back to Phase 1 IK tuning.
      GMR-MotionLab (a community GMR fork) exists specifically because stock GMR
      uses one IK config for a whole clip; a dance clip mixes slow stances with
      fast steps and usually needs per-phase weights.

### Phase 3 — Train the tracker (3–7 GPU-days on a spark)
- [ ] Register on the fleet, not the laptop. `dl` (4× RTX 6000 Ada) or a spark
      with >500 GB free. Use the container path already in `docker/apptainer/`.
- [ ] Train `Booster-K1-Makarena-v0` with the BeyondMimic PPO config.
- [ ] Record debug video every iteration (tiled: reference vs. actual vs. error).
- [ ] Watch: mean episode length (falling early = bad), tracking error per body
      part, foot-slip term.

### Phase 4 — Sim validation ladder (2 days)
Our existing tooling makes this cheap:
- [ ] MuJoCo (`mujoco_fleet_node.py`) — same policy code as training's Sim2Sim.
- [ ] Gazebo — `ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo`
- [ ] Isaac — `… backend:=isaac`
- [ ] Domain randomise: push the robot mid-dance, perturb PD gains ±30 %,
      randomise friction. A dance policy that only works with zero noise is not
      ready.

### Phase 5 — Hardware (1–2 days + supervision)
- [ ] Deploy via `booster_deploy` (official) or our
      `real.launch.py` (see `guide_real.md`).
- [ ] **The dance is full-body, so the SDK bridge must command all 22 DoF.**
      Our `sdk_bridge_node.cpp` currently maps only the 12 policy legs. Extend
      it with the head (`kHeadYaw`, `kHeadPitch`) and 8 arm indices from
      `JointIndexK1`, and use the BeyondMimic export's own 22-DoF action vector.
- [ ] Start at 0.25× speed, robot on a stand, e-stop live. Then 0.5×, then full.
- [ ] On failure: return to the **last gait** — you need a safe stand-down
      transition, not a hard cut.

---

## 4. Files this plan adds to `booster_ws`

```
isaac_tasks/k1_motion/                       # new package
  source/k1_motion/tasks/makarena/
      makarena_env_cfg.py                    # mirrors booster_train dance cfg
      __init__.py                            # gym.register Booster-K1-Makarena-v0[-Play]
  motions/K1/makarena.csv                    # retargeted (7 root + 22 joint cols)
  motions/K1/makarena.npz                    # via csv_to_npz.py

scripts/
  gmr_pkl_to_csv.py                          # GMR pkl → booster CSV contract
  extract_gvhmr_smplx.py                     # video → SMPL-X npz (thin wrapper)

src/k1_control/
  k1_control/motion_replay_node.py           # CSV → /{ns}/joint_commands (kinematic, debug)
  k1_control/sdk_bridge_node.cpp             # EXTEND to 22 DoF (head + arms)

docs/
  video_to_motion_plan.md                    # this file
```

Reused, not duplicated: `/{ns}/joint_commands`, `sim_validate.launch.py`,
`booster_assets.K1_JOINT_NAMES`, `booster_train/scripts/csv_to_npz.py`,
`k1_wbc` (for Phase-5 upper-body merging if we composite trackers).

---

## 5. Risks and known traps

| Risk | Severity | Mitigation |
|---|---|---|
| **Open-loop replay falls over** | certain | Phase 2 RL tracking; treat Phase 1 as visualisation only |
| **K1 4-bar ankle not in GMR config** | high | hand-tune ankle IK entries; verify foot roll on flat ground |
| **Quat order** `wxyz` vs `xyzw` | high | assert in the converter; unit-test one frame against a hand-checked value |
| **Dance physically unreachable for K1** | medium | check double-support fraction in Phase 2 **before** training |
| **Stock GMR one-config-per-clip** | medium | per-phase IK weights, or GMR-MotionLab fork |
| **GVHMR licence** (non-commercial research) + MPI-gated SMPL-X | blocker for product use | keep research scope; commercial needs licensed data |
| **MotionBricks is G1-only** | n/a now | out of scope for one clip; revisit for a *style* library later |
| **BONES-SEED motion licence** | medium | SOMA bundled data is NVIDIA Sample Data Eval Licence — not Apache-2.0 |
| **22-DoF hardware export mismatch** | high | extend SDK bridge before Phase 5, test in Sim2Sim first |
| **No safe fall→dance transition** | high | author a blend; test it repeatedly in sim before hardware |

---

## 6. Where the "smart" NVIDIA stack fits (later, not now)

The MotionBricks / SOMA / GEAR-SONIC / GR00T-WBC family is the *right* long-term
answer, but the wrong tool today:

- **MotionBricks** generates motion from prompts at 15 k FPS. We don't need
  generation — we have one specific dance we want reproduced faithfully.
  Its checkpoints are G1-specific (23+ DoF, different proportions) and the full
  robotics-integration release is ~1 month out.
- **SOMA Retargeter** is the more immediately useful piece: its **robot
  configurator** and **IK-weight optimiser** are better tooling than anything we
  would write. It bundles `booster_t1`; adding K1 is a documented guided
  process. Recommend: after Phase 1 works, port the K1 config and compare
  foot-plant quality against GMR. Keep whichever wins.

**Recommended order:** GMR (day 1) → SOMA K1 config (if GMR foot-plating is
poor) → BeyondMimic RL (the actual product) → MotionBricks (only if we ever want
a style library rather than specific clips).

---

## 7. Open questions for you

1. **Which video exactly?** Please drop the Macarena clip into the repo
   (`assets/motions/video/`) — camera static or moving changes the GVHMR path.
2. **Do you want the arms in the first version?** Arms-only replay (Path A) is a
   1-day demo that cannot fall over. Full-body dance (Path B) is ~2 GPU-weeks.
   Doing the arms-only demo first de-risks the whole pipeline cheaply.
3. **Which retargeter should be the reference implementation?** Recommend GMR
   (already installed, K1 config exists); SOMA is the quality ceiling to compare
   against later.
4. **Booster `booster_robotics_sdk_ros2` — should I audit it now** and possibly
   replace our hand-written C++ bridge?
