# K1 Velocity Locomotion Task

RSL-RL PPO training task for Booster K1 humanoid velocity control on rough terrain.

## Overview

This task trains a neural network policy to control the **12 leg joints** of the Booster K1 humanoid robot to follow linear and angular velocity commands. The policy learns to traverse rough terrain via reward shaping and curriculum learning.

**Key specs:**
- Robot: Booster K1 (22 DoF humanoid, 12 leg DoF for policy)
- Framework: Isaac Lab 2.2 + RSL-RL PPO
- Observation: 72D (base state, commands, joint state, height scan, last action)
- Action: 12D (leg joint position offsets)
- Reward: velocity tracking, gait quality, regularization
- Terrain: procedurally generated rough (curriculum) → flat (eval)

## Setup

### Prerequisites

1. **Isaac Lab 2.2** installed and environment activated
2. **RSL-RL** package: `pip install rsl-rl`
3. **Booster K1 asset** (USD path to be provided)

### Install k1_velocity Task

From `booster_ws` root:

```bash
pip install -e isaac_tasks/k1_velocity/
```

This registers the task:
- `Isaac-Velocity-Rough-K1-v0` — training env (rough terrain, 4096 envs)
- `Isaac-Velocity-Rough-K1-Play-v0` — eval env (flat terrain, 50 envs)

## Training

### Command

```bash
cd booster_ws
python isaac_tasks/k1_velocity/scripts/train.py \
    --task Isaac-Velocity-Rough-K1-v0 \
    --num_envs 4096 \
    --max_iterations 5000 \
    --device cuda:0 \
    --log_dir logs/k1_velocity
```

### Smoke Test (5 iterations)

```bash
python isaac_tasks/k1_velocity/scripts/train.py \
    --task Isaac-Velocity-Rough-K1-v0 \
    --max_iterations 5 \
    --headless
```

### Output

- Checkpoints: `logs/k1_velocity/k1_velocity_rough/<seed>_<iter>.pt`
- Tensorboard: `logs/k1_velocity/k1_velocity_rough/` (watch with `tensorboard --logdir=...`)

## Evaluation / Play

### Command

```bash
python isaac_tasks/k1_velocity/scripts/play.py \
    --task Isaac-Velocity-Rough-K1-Play-v0 \
    --checkpoint logs/k1_velocity/k1_velocity_rough/model_5000.pt \
    --num_envs 50
```

This runs the policy on flat terrain with 50 parallel environments.

### Export to TorchScript

```bash
python isaac_tasks/k1_velocity/scripts/play.py \
    --checkpoint logs/k1_velocity/k1_velocity_rough/model_5000.pt \
    --export models/k1_velocity_policy.pt
```

The exported model can be deployed on the real K1 for zero-shot control or further finetuning.

## Configuration Details

### Joint Configuration

**Leg joints (12 DoF policy control):**
- Left: Hip (Pitch, Roll, Yaw) + Knee (Pitch) + Ankle (Pitch, Roll) = 6
- Right: Hip (Pitch, Roll, Yaw) + Knee (Pitch) + Ankle (Pitch, Roll) = 6

**Arm/Head joints (10 DoF, passive regularization):**
- Head: Yaw, Pitch
- Each shoulder: Pitch, Roll (2 each × 2 sides = 4)
- Each elbow: Pitch, Yaw (2 each × 2 sides = 4)

These non-locomotion joints are penalized for deviation from default pose to keep arms/head stable.

### Observation Space (72D)

| Component | Dim | Notes |
|-----------|-----|-------|
| Base linear velocity | 3 | x, y, z (noisy) |
| Base angular velocity | 3 | roll, pitch, yaw rates (noisy) |
| Projected gravity | 3 | base frame gravity vector (noisy) |
| Velocity commands | 3 | vx, vy, yaw (target) |
| Joint positions (rel) | 12 | 12 leg joints relative to default (noisy) |
| Joint velocities | 12 | 12 leg joints (noisy) |
| Height scan | 16 | 4×4 raycaster grid (cropped to [-1, 1]) |
| Last action | 12 | previous step command |
| **Total** | **72** | |

### Action Space (12D)

- Joint position offsets: one per leg joint, normalized to [-1, 1]
- Scale: 0.25 rad (±14.3 degrees)
- Executed via PD controller with K=100–150, D=1–4 (tuned per joint)

### Reward Function

| Term | Weight | Purpose |
|------|--------|---------|
| `track_lin_vel_xy_exp` | 1.0 | Follow x, y velocity commands |
| `track_ang_vel_z_exp` | 2.0 | Follow yaw rate command (higher weight) |
| `feet_air_time` | 0.25 | Encourage proper gait rhythm |
| `feet_slide` | -0.1 | Penalize foot slipping during stance |
| `flat_orientation_l2` | -1.0 | Keep base upright |
| `action_rate_l2` | -0.005 | Smooth control |
| `dof_acc_l2` | -1.25e-7 | Minimize joint accelerations |
| `dof_torques_l2` | -1.5e-7 | Minimize joint torques |
| `dof_pos_limits` | -1.0 | Stay away from joint limits |
| `joint_deviation_arms` | -0.05 | Keep arms/head near default |
| `termination_penalty` | -200.0 | Penalize episode failure |

### Curriculum

Episodes start on **terrain level 0** (flat) and gradually increase to **level 5** (very rough) based on task success. This helps the policy learn stable locomotion before tackling difficult terrain.

### Randomization

At each reset:
- Robot joint positions randomized ±50%
- Random mass (±2 kg) added to base
- Periodic pushes (10–15s intervals, ±0.5 m/s velocity impulses)

## Asset Configuration

### K1 URDF → USD Conversion

The USD path is currently a placeholder:
```python
K1_USD_PATH = "{ISAACLAB_NUCLEUS_DIR}/Robots/Booster/K1/k1.usd"
```

**To update:** Once the K1 USD is available from booster_assets:
1. Update `K1_USD_PATH` in `source/k1_velocity/tasks/velocity/velocity_env_cfg.py`
2. Verify joint names match the K1 URDF:
   - Leg joints: `Left_Hip_Pitch`, `Left_Hip_Roll`, etc.
   - Arm/head joints: `Left_Shoulder_Pitch`, `Head_yaw`, etc.
3. Tune actuator limits/stiffness from K1 datasheet

### Actuator Tuning

Current placeholder values (from G1 reference):

```python
"hip_knee": effort_limit=120.0, stiffness=100–150, damping=2–4
"ankle":    effort_limit=40.0,  stiffness=40,      damping=1
"arms_head": effort_limit=20.0, stiffness=20,      damping=1
```

Update based on K1 motor specs (datasheet).

## Troubleshooting

### Error: "Isaac Lab not found"
Activate IsaacLab environment:
```bash
source ~/IsaacLab/setup_conda_env.sh
conda activate isaaclab
```

### Error: "Task not registered"
Ensure `k1_velocity` is installed:
```bash
pip install -e isaac_tasks/k1_velocity/
```

### Slow simulation
- Reduce `num_envs` (e.g., 1024 instead of 4096)
- Use `--headless` flag
- Check GPU memory with `nvidia-smi`

### Policy not learning
- Check observation/action noise levels
- Verify reward scaling (some terms may be too small)
- Increase `num_steps_per_env` (currently 24)
- Lower learning rate or increase `desired_kl` threshold

## References

- **Ported from:** `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`
- **RSL-RL docs:** https://github.com/leggedrobotics/rsl_rl
- **Isaac Lab docs:** https://docs.isaaclab.io

## License

Apache-2.0 (Isaac Lab project standard)
