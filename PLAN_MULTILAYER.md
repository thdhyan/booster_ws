# Multi-Layered Policy Architecture for K1 Soccer

## Production System Reference (robocup_demo)

The official Booster RoboCup demo uses:
- **Behavior Trees** (BT) for high-level decision making
- **YOLOv8** for vision (ball/robot/field detection)
- **Traditional control** for locomotion
- **RLVisionKick** for kicking (firmware ≥1.5.2)

Key states from `game.xml`:
```
INITIAL → READY → SET → PLAY → END
                    ↓
              FREE_KICK / PENALTY
```

## Our Architecture: RL Policies + BT Coordinator

Replace traditional locomotion with RL policies while keeping the BT framework:

```
┌─────────────────────────────────────────────────────────────┐
│                    Behavior Tree (BT)                        │
│  game.xml: StrikerPlay, GoalKeeperPlay, etc.                │
│  Handles: game states, ball finding, role switching          │
└──────────┬──────────────────┬──────────────────┬───────────┘
           │                  │                  │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Velocity   │   │   Head      │   │  Kicking    │
    │  Policy     │   │  Tracking   │   │  Policy     │
    │  (12 legs)  │   │  (2 head)   │   │ (12 legs)   │
    └─────────────┘   └─────────────┘   └─────────────┘
           │                  │                  │
    ┌──────▼──────────────────▼──────────────────▼──────┐
    │              Joint Command Publisher               │
    │         /robot_N/joint_commands (ROS2)            │
    └───────────────────────────────────────────────────┘
```

## Layer 1: Velocity Policy (Walk)
- **Observations**: 48-dim (base_vel + gravity + cmd + joints + last_action)
- **Actions**: 12 leg joint positions
- **Training**: PPO with teacher-student distillation
- **Status**: ✅ Trained (reward 40.89)
- **File**: `models/k1_velocity_policy.pt`

## Layer 2: Head Tracking Policy
- **Observations**: 11-dim (ball_angle + head_joints + base_ang_vel + last_action)
- **Actions**: 2 head joint positions (AAHead_yaw, Head_pitch)
- **Training**: PPO, tracking reward
- **Status**: 🔲 To train
- **Config**: `isaac_tasks/k1_head_tracking/`

## Layer 3: Kicking Policy
- **Observations**: 48-dim (ball_state + base_vel + gravity + joints + last_action)
- **Actions**: 12 leg joint positions (kick motion)
- **Training**: PPO with ball velocity reward
- **Status**: 🔲 To train
- **Config**: `isaac_tasks/k1_kicking/`

## Integration with robocup_demo

The BT already handles high-level decisions. We add RL policy calls:

### Modified StrikerPlay BT Node
```xml
<ReactiveSequence name="striker_play">
    <!-- Track ball with head -->
    <TrackBallHead />  <!-- NEW: calls head tracking policy -->
    
    <!-- Chase ball -->
    <SimpleChase vx_limit="1.0" stop_dist="0.5" />
    
    <!-- Kick when close -->
    <Kick speed_limit="1.2" />  <!-- MODIFIED: uses kicking policy -->
</ReactiveSequence>
```

### RL Policy Integration Node
```cpp
// rl_policy_node.cpp
class RLPolicyNode : public BT::SyncActionNode {
    bool tickTick() override {
        // Get ball state from vision
        auto ball = getBallPosition();
        
        // Build observations
        auto obs = buildObservation(ball, jointStates);
        
        // Run appropriate policy
        if (ball.distance > 2.0) {
            // WALK: velocity policy
            auto action = velocityPolicy(obs);
            publishJointCommands(action);
        } else if (ball.distance > 0.5) {
            // TRACK: velocity + head tracking
            auto legAction = velocityPolicy(obs);
            auto headAction = headTrackingPolicy(ballAngle);
            publishJointCommands(legAction, headAction);
        } else {
            // KICK: kicking policy
            auto action = kickingPolicy(obs);
            publishJointCommands(action);
        }
        return true;
    }
};
```

## Training Pipeline

### Step 1: Train Head Tracking Policy
```bash
cd ~/Projects/thdhyan/IsaacLab
./isaaclab.sh -p source/isaaclab/isaaclab/scripts/rsl_rl/train.py \
    --task Isaac-HeadTracking-K1-v0 \
    --num_envs 4096 \
    --max_iterations 3000
```

### Step 2: Train Kicking Policy
```bash
./isaaclab.sh -p source/isaaclab/isaaclab/scripts/rsl_rl/train.py \
    --task Isaac-Kicking-K1-v0 \
    --num_envs 4096 \
    --max_iterations 5000
```

### Step 3: Distill to Student (optional)
```bash
./isaaclab.sh -p source/isaaclab/isaaclab/scripts/rsl_rl/train.py \
    --task Isaac-HeadTracking-K1-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --teacher呕吐 models/k1_head_tracking_teacher.pt
```

## Contact Sensor Setup (from booster_train reference)

The `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)` places sensors on ALL rigid bodies. The `undesired_contacts` reward uses regex to exclude feet:

```python
undesired_contacts = RewTerm(
    func=mdp.undesired_contacts,
    weight=-10.0,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
        ),
        "threshold": 1.0,
    },
)
```

**Yes, the simulation calculates contact forces for ALL bodies** — the regex only filters which forces are used in the reward calculation.

## Key Design Decisions

1. **Separate policies** (not multi-head): Easier to train, debug, and replace individual components
2. **BT coordinator** (not learned): Deterministic transitions based on game state — more predictable
3. **Velocity policy unchanged**: The existing 48→12 policy works well, no retraining needed
4. **Head tracking independent**: Head joints are decoupled from legs, can train separately
5. **Kicking overrides legs**: When close to ball, kicking policy takes full control of legs

## Next Steps

1. [ ] Register `Isaac-HeadTracking-K1-v0` and `Isaac-Kicking-K1-v0` tasks
2. [ ] Train head tracking policy (3000 iterations)
3. [ ] Train kicking policy (5000 iterations)
4. [ ] Test composition in Isaac Sim Docker
5. [ ] Integrate with robocup_demo BT framework
6. [ ] Deploy on real robots
