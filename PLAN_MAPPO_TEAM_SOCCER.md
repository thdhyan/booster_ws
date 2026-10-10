# PLAN: MAPPO team soccer for K1 (and the multi-agent push stepping stone)

Status: plan only, 2026-10-10. Builds on the single-robot stack:

- the velocity student (walk, `models/k1_velocity_student_gaitfix.pt`, frozen)
- the residual skills P8 ball-track and P9 kick (`tasks/skills/`, training since 2026-10-10)
- the push stack (`tasks/push/`), as the first multi-robot testbed

## 0. Goal

Teams of K1s (2v2 first, then 3v3 with a keeper) play soccer in Isaac Lab. Every robot runs one
shared high-level policy, trained with MAPPO (shared actor and a centralised critic), on top of
frozen low-level skills. The result must deploy: the actor sees only egocentric,
camera-plausible inputs and teammate messages, never the global state.

## 1. Architecture

```
 MAPPO actor (shared weights, one row per robot, 50 Hz; 10 Hz decision hold later)
   obs_i = ego proprio summary | ball (rel pos/vel, visible flag, last-seen age)
           | teammates (rel pose/vel) | opponents (rel pose, FOV-masked) | goals
           | field-edge distances | agent role one-hot | game clock
   act_i = [vx, vy, wz | head yaw, pitch | kick gate]          (6 continuous)
            |                      |            |
            v                      v            v
   frozen velocity student <- residual skill (track / kick, chosen by kick gate > 0)
            |
         12 leg targets + 2 head targets

 MAPPO critic (training only): global state = every robot's pose and velocity in the field
           frame, ball state, score, clock, plus the agent id of the row being valued
```

- **Skills stay frozen at first.** MAPPO learns *where* to go and *when* to kick. The P8/P9
  residual actors are loaded as TorchScript inside the action term, the same way the frozen
  push base is loaded (`FrozenBaseVelocityAction`) and the student is loaded
  (`FrozenStudentResidualAction`). A late stage may unfreeze them with a low learning rate.
- **Kick gate:** a continuous scalar thresholded at 0, so the policy stays a plain Gaussian
  PPO. rsl_rl has no hybrid discrete+continuous action space, and we do not need one yet.

## 2. Trainer: agents-in-batch MAPPO on rsl_rl

Keep the existing toolchain (rsl_rl 5, wandb, `train.py`, `play_record.py`, checkpoints) and
add no new framework:

1. **Env:** a `ManagerBasedRLEnv` with `R` robots per cell (`robot_0..robot_{R-1}`), one
   ball and one field. Each robot gets its own action term instance (`asset_name=robot_k`),
   and every obs term takes an `asset_name` parameter.
2. **`AgentsInBatchVecEnv` wrapper:** reshapes `(E, R*d)` to `(E*R, d)` for the actor group
   and broadcasts the global `critic` group to every row with the agent id appended. Each row
   gets its own agent's reward; dones are per cell, broadcast to all R rows.
   - To rsl_rl this is plain PPO over `E*R` "envs". Shared parameters plus a centralised
     critic is exactly MAPPO (Yu et al. 2022).
3. **Opponents:** opponent rows are taken out of the learning batch. A league wrapper drives
   them with a frozen snapshot, so only the learning team's rows reach PPO.
   - Snapshots: the latest checkpoint, the last k, and scripted bots.
   - Sampling: PFSP, weighted toward opponents we beat less than 50 % of the time.

Alternative considered: skrl MAPPO with a `DirectMARLEnv`. It is ready-made, but it means a
second training stack, a second checkpoint format and rewriting the video tools. It is the
fallback if the wrapper turns out wrong.

## 3. Rewards

| Kind | Term | Notes |
|---|---|---|
| team, sparse | goal for +100, goal against −100 | dominant, never annealed |
| team, dense | potential-based ball progress toward the opponent goal | annealed 1.0 → 0.2 |
| team | possession: our closest robot nearer the ball than theirs | small |
| individual | approach ball, **paid only to the teammate closest to the ball** | stops everyone swarming the ball |
| individual | spacing: penalty while a teammate is within 1.0 m | spreads the team |
| individual | fall −50, out of bounds −10, energy, action rate | same as the skills |
| individual | kick quality (ball speed toward the goal after a gate) | early stages only |

## 4. Curriculum (each stage gates the next)

| Stage | Setup | Gate to advance |
|---|---|---|
| S0 | 1v0, MAPPO with R=1 (= PPO) on frozen P8/P9 | scores ≥ 70 % within 20 s |
| S1 | 2v0, shared actor | scores ≥ 70 %, teammate overlap < 10 % of steps, a pass emerges |
| S2 | 2v1 vs a scripted defender (BT chaser) | win rate ≥ 60 % |
| S3 | 2v2 self-play league (PFSP) | Elo rising vs the frozen pool for 500 iters |
| S4 | 3v3 + keeper role (role one-hot) | beats the S3 champion ≥ 60 % |
| S5 | distil to a deployable student: FOV-masked opponents, estimated ball, 100 ms teammate-message latency | ≤ 10 % win-rate drop vs the teacher |

Field: 9 × 6 m (KidSize-like), goal 2.6 m wide. Episode 60 s at S3+, ending early on a goal
(kick-off reset) or on a team fall cascade.

## 5. Compute and scene budget

- One cell = R robots + ball + field. The K1 articulation dominates the cost. Starting
  budget: 2v2 at 1024 cells = 4096 robots per GPU, about the size of today's single-robot runs.
- Frozen MLPs (student 500→12, skills) run batched over every robot inside the action term:
  negligible next to physics.
- Robot-robot collisions ON. PhysX GPU pair buffers must be scaled (`ensure_physx_gpu_capacity`
  already scales with env count; extend it with robots per cell).

## 6. Multi-agent push: the testbed before soccer (M0 to M2)

Push already has a trained high-level policy over a frozen base. It is the cheapest place to
debug the multi-robot plumbing (per-robot action terms, per-robot obs, the agents-in-batch
wrapper) before a ball and opponents are involved.

| Step | What | Done when |
|---|---|---|
| M0 | Generalise `push_mdp` from `scene["robot"]` to an `asset_name` param. That covers 9 hard-coded sites plus the per-robot height scanner, wrist IK terms and `_state` (h_cmd per robot). | single-robot Push-SG reproduces its reward with `asset_name="robot"` |
| M1 | `Isaac-Push-Multi-K1-Play-v0`: R robots, each driven by the **frozen trained push policy** (the p6 checkpoint, exported) | video: each robot pushes, with no cross-talk between robots |
| M2 | Cooperative heavy box: a box too heavy or wide for one robot, a wrist-target slot per robot on the near face, shared goal; MAPPO fine-tune from the frozen single-robot policy | 2 robots reach the goal where 1 robot fails |

Open choice for M1, pending the user's decision: **one box per robot**, sharing the arena
(tests the plumbing only), or **one shared box** (true cooperation, and leads straight to M2).
Default if nobody decides: shared box.

## 7. Verification and videos (compulsory)

Every smoke, eval and play run gets a tiled debug video. Panels:

- overview
- top-down field view
- one follow-cam per robot

Each panel is 1920×1080, denoised (spp 8, 4 subframes). The status header shows score, clock,
stage and each robot's gate/skill. Scene markers:

- ball last-seen and predicted positions
- each robot's FOV wedge
- field boundary and goal mouths
- teammate spacing radius

Long runs save `.pt` every 100 iterations, and videos come from play runs on those checkpoints.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Swarming (everyone chases the ball) | closest-only approach reward, spacing penalty, role one-hot |
| Non-stationary self-play, cycling strategies | PFSP league, keep old snapshots, Elo tracking |
| Frozen skills cap play quality (kick strength, turning) | S4+: unfreeze the residuals at LR × 0.1 |
| Robot-robot contact destabilises the frozen student (never saw contact) | push-robot DR already in skills; add robot-contact DR in P8/P9 fine-tune |
| Deployability gap (GT opponents, global state leak) | critic-only global state; S5 distillation with FOV masks and latency |

## 9. Execution order

1. P8/P9 residual skills converge (running), then export them to TorchScript.
2. M0 → M1 → M2 multi-agent push (plumbing + wrapper).
3. S0 → S1 on the skills (soccer env with R robots, no opponents).
4. Scripted defender, then S2.
5. League + S3 → S4.
6. S5 distillation, then the ROS 2 deployment node (`k1_` package, `robot_ns` per robot).
