# Booster K1 Workspace — Handoff

> **Date:** 2026-08-19  
> **Author:** thdhyan  
> **GitHub:** https://github.com/thdhyan/booster_ws

---

## Current State

### Phase 0 — COMPLETE ✅
Branch `dev/phase-0` pushed to GitHub. All scaffolding done.

### Active Training Run
```
tmux session: k1_train
Task:         Isaac-Velocity-Rough-K1-v0
Envs:         1024
Max iter:     5000
WandB:        booster_k1_locomotion project
Log:          logs/train_k1_velocity.log
```

**Monitor:**
```bash
tmux attach -t k1_train          # live output
tail -f logs/train_k1_velocity.log
```

---

## Environment Setup

### Python / IsaacLab (for training + Isaac Sim)
```bash
source /home/thakk100/Projects/IsaacLab/.venv-isaac/bin/activate
# IsaacSim 6.0.1 — Python 3.12
# IsaacLab 3.0.0b2 + rsl_rl + isaaclab_tasks
```

### System ROS2 (for all k1_* packages)
```bash
source /opt/ros/jazzy/setup.bash
cd /home/thakk100/Projects/booster_ws
source install/setup.bash   # after colcon build
```

**NEVER source both in the same shell.**

---

## Repo Layout

```
booster_ws/
├── src/
│   ├── k1_interfaces/       # msgs: JointCommand, LocomotionCommand, RobotStatus, PolicyObs
│   ├── k1_description/      # URDF xacro stub + ros2_control.yaml
│   │   └── assets/          # submodule → thdhyan/booster_assets (K1 URDF, meshes)
│   ├── k1_control/          # sdk_bridge_node (stub), sim_bridge_node
│   ├── k1_locomotion/       # 50 Hz TorchScript policy node
│   ├── k1_wbc/              # decoupled WBC stub (Phase 5+)
│   ├── k1_sim_gazebo/       # Gazebo Harmonic — flat.sdf, ros_gz_bridge, spawn
│   ├── k1_sim_isaac/        # Isaac Sim 6.0.1 standalone + ROS2 bridge config
│   └── k1_bringup/          # top-level launches: gazebo, isaac, real, fleet
├── sdk/
│   └── booster_robotics_sdk/ # submodule → upstream (read-only, pinned d5d8f7a)
├── isaac_tasks/
│   ├── k1_velocity/          # OUR RSL-RL task (Isaac-Velocity-Rough-K1-v0)
│   └── booster_train_ref/    # submodule → thdhyan/booster_train (reference)
├── models/                   # trained policies (Git LFS: *.pt)
├── logs/                     # training logs (gitignored)
├── PLAN.md                   # full architecture + phase plan
├── TASKS.md                  # task checklist with status
└── CLAUDE.md                 # coding rules
```

---

## Submodules

| Path | Repo | Type |
|------|------|------|
| `sdk/booster_robotics_sdk` | `BoosterRobotics/booster_robotics_sdk` | read-only |
| `src/k1_description/assets` | `thdhyan/booster_assets` | editable fork |
| `isaac_tasks/booster_train_ref` | `thdhyan/booster_train` | reference fork |

```bash
git submodule update --init --recursive   # init all after fresh clone
```

---

## Training

### Restart / resume
```bash
source /home/thakk100/Projects/IsaacLab/.venv-isaac/bin/activate
PYTHON=/home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12
TRAIN=isaac_tasks/booster_train_ref/scripts/rsl_rl/train.py

tmux new-session -d -s k1_train \
  "$PYTHON $TRAIN \
    --task Isaac-Velocity-Rough-K1-v0 \
    --num_envs 1024 \
    --headless \
    --max_iterations 5000 \
    --seed 42 2>&1 | tee logs/train_k1_velocity.log"
```

### Key files
| File | Purpose |
|------|---------|
| `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py` | Env config (obs, actions, rewards, terrain) |
| `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/agents/rsl_rl_ppo_cfg.py` | PPO hyperparams + wandb config |
| `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/__init__.py` | Gym registration |

### Robot config used
`BOOSTER_K1_CFG` from `booster_train.assets.robots.booster`:
- URDF: `booster_assets/robots/K1/K1_22dof.urdf`
- Root link: `Trunk` (height z=0.57m)
- Foot links: `left_foot`, `right_foot`
- Real actuator models: BoosterDelayedPDActuatorCfg per joint type

### After training — export policy
```bash
$PYTHON isaac_tasks/booster_train_ref/scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-K1-Play-v0 \
  --checkpoint logs/k1_velocity_rough/*/model_5000.pt \
  --headless

# Copy to models/ (tracked via Git LFS)
cp <checkpoint>.pt models/k1_velocity_policy.pt
git add models/k1_velocity_policy.pt
git commit -m "feat: add trained K1 velocity policy (5000 iter)"
```

---

## Next Steps (TASKS.md reference)

| Task | Priority |
|------|---------|
| T1.1 — Fix URDF xacro to include actual K1 URDF from assets/ | High |
| T1.2 — Wire gz_ros2_control + test Gazebo spawn | High |
| T1.4 — End-to-end Gazebo bringup test | High |
| T4.1 — Wire policy into locomotion_node (load .pt, publish JointCommand) | After training |
| T3.4 — Verify training reward increases; play eval | During training |
| T0.G5 — Pin SDK submodule to tagged release | Low |

---

## Package Install (venv-isaac)

```bash
VENV=/home/thakk100/Projects/IsaacLab/.venv-isaac
uv pip install \
  --python $VENV/bin/python3.12 \
  -e src/k1_description/assets/ \
  -e isaac_tasks/booster_train_ref/source/booster_train/ \
  -e isaac_tasks/k1_velocity/

# Path fix (editable install bug workaround):
echo "$(pwd)/isaac_tasks/k1_velocity/source" > \
  $VENV/lib/python3.12/site-packages/k1_velocity.pth
```

---

## Git Workflow

```bash
# Current branch
git checkout dev/phase-0

# Phase 1 (Gazebo)
git checkout -b dev/phase-1
# ... work ...
git push origin dev/phase-1
gh pr create --base main --title "feat: Phase 1 — Gazebo sim"

# Milestone tag after merge
git checkout main && git merge dev/phase-1
git tag v0.2-gazebo && git push origin main --tags
```

WandB project: **booster_k1_locomotion** (https://wandb.ai/thdhyan/booster_k1_locomotion)
