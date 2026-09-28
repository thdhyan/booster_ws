# Video → Motion on the Booster K1 — Implementation Plan

Date: 2026-09-27 · Status: research complete; feasibility gate implemented
Branch: `feat/video-to-motion`
Scope: retarget recorded human motion (Makarena dance, arbitrary video) **and**
text-generated motion onto the Booster K1, first in sim, then on hardware.

---

## 0. TL;DR — the two questions you asked

**"How do we know SOMA retargeting won't crash the robot?"**

We don't — and SOMA says so itself:

> "Generated motion is **kinematic**; validate its safety, feasibility, and
> controller compatibility in simulation before hardware use." — SOMA README
>
> "Optimizer loss measures kinematic tracking and smoothness. **It does not prove
> physical feasibility or controller performance.**" — `compare-and-tune.md`

So the answer is a **gate**, not a guarantee. Built:
[`scripts/motion_feasibility_gate.py`](../scripts/motion_feasibility_gate.py) —
10 checks (joint limits / velocity / accel / torque bound, foot penetration,
float, slip, root height, double support, CoM-over-support) against the real K1
URDF limits and real MuJoCo forward kinematics.

It already earned its keep — run against **Booster's own shipped K1 motions**:

```
k1_mj2_seg1_50fps.csv    7 blocking failures
k1_fight_001_30fps.csv   8 blocking failures
```

Those are *RL training references*, not hardware-safe playback scripts. A dance
retargeted by GMR or SOMA needs the same treatment.

**"I want both text-to-motion and video-to-motion."** They converge:

```
VIDEO  ──GVHMR──►  SMPL-X  ─┐
                             ├──►  retarget  ──► K1 CSV ──► GATE ──► RL tracker
TEXT   ──Kimodo──►  SOMA/G1 ─┘   (GMR and/or SOMA)
```

`Kimodo-SMPLX` emits AMASS npz → GMR consumes directly.
`Kimodo-SOMA` emits SOMA skeleton → SOMA Retargeter consumes directly.
Use **both** retargeters and compare (§6).

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
  motion_feasibility_gate.py                   # ← IMPLEMENTED. the safety gate.
  gmr_pkl_to_csv.py                            # GMR pkl → booster CSV contract
  extract_gvhmr_smplx.py                       # video → SMPL-X npz (thin wrapper)

src/k1_control/
  k1_control/motion_replay_node.py             # CSV → /{ns}/joint_commands (kinematic, debug)
  k1_control/sdk_bridge_node.cpp               # EXTEND to 22 DoF (head + arms)
  k1_control/imu_source_node.py                # rpy ImuState → quat sensor_msgs/Imu

external (cloned outside the repo, per their instructions)
  ~/Projects/soma-retargeter/                  # + assets/robotics/booster/booster_k1/
  ~/Projects/kimodo/                           # text-to-motion

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
| **`ImuState` has rpy, not quat** | high | our `_imu_cb` reads a quaternion → projected gravity silently dies. See `docs/sdk_ros2_audit.md` |
| **SOMA `sole_normal_local` for K1** | medium | K1 foot box is at local `[0.026, 0, -0.02]`, so sole-up is **-Z**; default assumes +Z and silently skips flattening |
| **Kimodo VRAM** | medium | ~17 GB, or `TEXT_ENCODER_DEVICE=cpu` (<3 GB). Laptop 4060 is 8 GB — run on `dl`/spark |
| **Registering a new SOMA *source* is a code change** | low | we only need a new *robot* target, which is config-only |
| **Gate is kinematic, not physics** | by design | MuJoCo → Gazebo → Isaac rollout is still mandatory (Phase 4) |

---

## 5b. SOMA Retargeter — adding K1 as a target

SOMA bundles `unitree_g1`, `unitree_h2`, `booster_t1`, `agibot_x2ultra`,
`agibot-a3t3`. **K1 is not there**, but adding it is a documented guided
process, not a research project. `booster_t1` is the closest sibling (same
vendor, 23 DoF, same leg topology) and is a good template.

```bash
git clone https://github.com/NVIDIA/soma-retargeter
cd soma-retargeter && git lfs install
uv run python app/tools/robot_config_generator.py --viewer gl   # configurator
```

We already have the inputs it wants — `K1_22dof.urdf` + `meshes/` in
`src/k1_description/assets/robots/K1/`. Note the configurator **cannot read
xacro**, so use the plain `.urdf` (we have it), and it needs
`+Z up, +X forward, +Y left`.

What it produces per robot:
```
assets/robotics/booster/booster_k1/
  manifest.json
  desc/K1_22dof.urdf + meshes/
  configs/
    soma_to_booster_k1_scaler_config.json
    soma_to_booster_k1_retargeter_config.json
    booster_k1_post_processing_config.json
```

Mapping for K1 (22 DoF, 12 legs + 8 arms + 2 head — vs H1's 19):

| SOMA joint | K1 link | note |
|---|---|---|
| `Hips` | `Trunk` | **must** be the root link |
| `Chest` | `Trunk` | K1 has no separate torso link — competes with Hips; expect to leave one unmapped |
| `LeftArm` / `RightArm` | `Left_Arm_3` / `Right_Arm_3` | from GMR's K1 config, these are the shoulder links |
| `LeftForeArm` / `RightForeArm` | `left_hand_link` / `right_hand_link` | |
| `LeftLeg` / `RightLeg` | `Left_Hip_Yaw` / `Right_Hip_Yaw` | GMR's K1 config uses these as the hip landmark |
| `LeftShin` / `RightShin` | `Left_Shank` / `Right_Shank` | |
| `LeftFoot` / `RightFoot` | `left_foot_link` / `right_foot_link` | |
| `Head` | `Head_2` | K1 **has** a head, unlike H1 |

Then run the **IK weight optimizer** on the bundled 15 BVH motions, and finally
enable the foot-plant post-processing (see §5c).

### 5c. Why foot planting is the whole ballgame

SOMA's `documentation/foot-contact.md` is the most important page in the repo
for our purposes. Retargeting without it produces visible skating and the gate
will (correctly) reject the clip. Key knobs:

| Stage | Param | Default | What it does |
|---|---|---|---|
| detect | `velocity_contact` | 0.1 m/s | foot speed below this opens a contact window |
| detect | `jerk_contact` | -0.05 m/s³ | jerk trigger |
| accept | `max_flatness_deg` | 25° | max sole tilt from world up to accept a plant |
| accept | `plant_speed_threshold` | 0.15 m/s | max foot speed for a valid plant |
| accept | `min_plant_frames` | 3 | drop shorter runs |
| correct | `propagation_ratio` | 0.4 | fraction of adjacent swing that takes position correction |
| correct | `enable_flatten_foot_plant` | true | flatten sole to horizontal |

Requires two enable flags in `*_retargeter_config.json`:
`enable_post_processing` and `enable_contact_processing`.

For K1 specifically, `sole_normal_local` must be set if the foot effector
link's local `+Z` is not sole-up at the zero pose — the K1 foot box sits at
local `[0.026, 0, -0.02]`, so the sole normal is **-Z**, not +Z. Getting this
wrong silently disables flattening.

## 5d. Text-to-motion — Kimodo

[Kimodo](https://github.com/nv-tlabs/kimodo) (NVIDIA, arXiv 2603.15546) is a
kinematic motion **diffusion** model: text prompt → 3D motion. Apache-2.0 code,
3.5k stars, checkpoints on HuggingFace under the NVIDIA Open Model licence.

```bash
git clone https://github.com/nv-tlabs/kimodo && cd kimodo
conda create -n kimodo python=3.10 -y && conda activate kimodo
pip install -e .
kimodo_gen --prompt "a person dances the macarena with arms raised" \
           --model Kimodo-SOMA-SEED-v1.1 --duration 6
```

Needs ~17 GB VRAM, or set `TEXT_ENCODER_DEVICE=cpu` (<3 GB). Our laptop has an
8 GB 4060 — so **run this on `dl` or a spark, not the laptop**.

Model choice for our pipeline:

| Model | Output | Feeds |
|---|---|---|
| `Kimodo-SMPLX-RP-v1` | AMASS npz | → **GMR** → K1 (NVIDIA documents this exact path) |
| `Kimodo-SOMA-SEED-v1.1` | SOMA skeleton | → **SOMA Retargeter** → K1 (needs our K1 target) |
| `Kimodo-G1-*` | MuJoCo qpos CSV | G1 only — wrong robot, skip |

Take `Kimodo-SMPLX` first: it needs no new robot target because GMR already
knows `booster_k1`. Use `Kimodo-SOMA` later once the SOMA K1 config exists —
that route is the one that can emit **robot-native** motion, avoiding the
SMPL-X detour entirely.

Also relevant: **BONES-SEED** (142,220 clips, HF `bones-studio/seed`) is the
public corpus both SOMA and Kimodo-SEED are built from, in SOMA BVH and G1
formats. It is a *dataset* option if we ever want to train our own policy
without writing a single prompt. Text annotations are in
`nvidia/SEED-Timeline-Annotations`.

**ARDY** (2026-07-10, `nv-tlabs/ardy`) is Kimodo's real-time successor — worth
knowing about, not needed now.

---

## 6. Where the "smart" NVIDIA stack fits

The MotionBricks / SOMA / GEAR-SONIC / GR00T-WBC family is the *right* long-term
answer. Updated for the SOMA + text-to-motion decision:

- **MotionBricks** generates motion from prompts at 15 k FPS. We don't need
  generation — we have one specific dance we want reproduced faithfully.
  Its checkpoints are G1-specific (23+ DoF, different proportions) and the full
  robotics-integration release is ~1 month out.
- **SOMA Retargeter** is the tooling win. Its **robot configurator**,
  **IK-weight optimiser** and **foot-plant correction** are better than anything
  we would write, and K1 is a guided add (§5b). It bundles `booster_t1`.
- **MotionBricks** stays out of scope *for this clip*. It generates motion from
  prompts at 15 k FPS — we want one specific dance reproduced faithfully, not
  generated. Checkpoints are G1-specific and the full robotics release is ~1
  month out. It becomes interesting later, as a **style library**
  (zombie walk, injured walk) rather than specific clips.

**Recommended order:**
```
1. GMR + Kimodo-SMPLX          day 1     — no new robot config needed
2. feasibility gate             done      — scripts/motion_feasibility_gate.py
3. SOMA K1 target (§5b)         1-2 days  — configurator + IK optimiser
4. compare GMR vs SOMA          0.5 day   — keep whichever foot-plants better
5. BeyondMimic RL tracker       3-7 GPU-days  ← the actual product
6. MotionBricks                 later     — only for a style library
```

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
