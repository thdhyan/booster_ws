#!/usr/bin/env python3
"""K1 Soccer — Multi-layered policy composition at runtime.

Composes three policies:
  1. Velocity policy → leg joints (always active)
  2. Head tracking policy → head joints (tracks ball)
  3. Kicking policy → leg joints (overrides velocity when close to ball)

State machine:
  WALK:  ball_distance > 2.0m → velocity policy
  TRACK: 0.5m < ball_distance ≤ 2.0m → velocity + head tracking
  KICK:  ball_distance ≤ 0.5m + aligned → kicking policy

Usage:
  python k1_soccer_compose.py --n_robots 2

Requires:
  - Trained velocity policy: models/k1_velocity_policy.pt
  - Trained head tracking policy: models/k1_head_tracking_policy.pt
  - Trained kicking policy: models/k1_kicking_policy.pt
  - ROS_DOMAIN_ID=0, CycloneDDS for discovery
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────
# Joint order (22 DoF)
ALL_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]
HEAD_IDX = [ALL_JOINTS.index(j) for j in HEAD_JOINTS]

LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
LEG_IDX = [ALL_JOINTS.index(j) for j in LEG_JOINTS]

# Policy dimensions
VEL_OBS_DIM = 48   # velocity policy
HEAD_OBS_DIM = 12  # head tracking: detection(3) + head(2+2) + ang_vel(3) + action(2)
KICK_OBS_DIM = 48  # kicking policy

# CCW search: when the camera has no ball for LOST_FRAMES steps, issue an
# in-place counter-clockwise velocity command (0, 0, +omega) to the locomotion
# policy so the robot rotates to re-acquire an out-of-FOV ball.
CCW_OMEGA = 0.6
LOST_FRAMES = 25   # 0.5 s at 50 Hz

ACTION_SCALE = 0.25
DEFAULT_LEG_POS = np.zeros(12, dtype=np.float32)
DEFAULT_HEAD_POS = np.zeros(2, dtype=np.float32)

# State machine thresholds
WALK_THRESHOLD = 2.0   # meters — switch to TRACK
KICK_THRESHOLD = 0.5   # meters — switch to KICK
KICK_ALIGN_THRESHOLD = 0.3  # radians — ball must be aligned for kick


class SoccerState:
    """State machine for soccer behavior."""
    WALK = 0
    TRACK = 1
    KICK = 2


class PolicyComposer:
    """Composes velocity, head tracking, and kicking policies."""

    def __init__(self, vel_path, head_path, kick_path, device="cpu"):
        self.device = device

        # Load velocity policy (legs)
        self.vel_policy = torch.jit.load(vel_path, map_location=device)
        self.vel_policy.eval()

        # Load head tracking policy (head)
        self.head_policy = torch.jit.load(head_path, map_location=device)
        self.head_policy.eval()

        # Load kicking policy (legs override)
        self.kick_policy = torch.jit.load(kick_path, map_location=device)
        self.kick_policy.eval()

        # Per-robot state
        self.state = SoccerState.WALK
        self.last_leg_action = np.zeros(12, dtype=np.float32)
        self.last_head_action = np.zeros(2, dtype=np.float32)
        self.kick_timer = 0  # frames since kick started
        # CCW search state
        self.lost_frames = 0
        self.active_cmd = np.zeros(3, dtype=np.float32)  # what locomotion tracks

    def update_search(self, ball_visible: bool) -> bool:
        """Track detection loss; return True while searching (CCW in place)."""
        self.lost_frames = 0 if ball_visible else self.lost_frames + 1
        if self.lost_frames >= LOST_FRAMES:
            self.active_cmd = np.array([0.0, 0.0, CCW_OMEGA], dtype=np.float32)
            return True
        self.active_cmd = np.zeros(3, dtype=np.float32)
        return False

    def get_state(self, ball_distance, ball_angle):
        """Determine state based on ball position."""
        if ball_distance <= KICK_THRESHOLD and abs(ball_angle) < KICK_ALIGN_THRESHOLD:
            return SoccerState.KICK
        elif ball_distance <= WALK_THRESHOLD:
            return SoccerState.TRACK
        else:
            return SoccerState.WALK

    def compute_action(self, obs_vel, obs_head, obs_kick, ball_distance, ball_angle):
        """Compute combined action based on state.

        obs_head[0] is the detector's visible flag; while it is lost for
        LOST_FRAMES steps the locomotion policy receives the in-place CCW
        command via self.active_cmd (build obs_vel with it).
        """
        state = self.get_state(ball_distance, ball_angle)
        self.state = state
        self.update_search(bool(obs_head[0] > 0.5))

        # Build full 22-DoF action
        full_action = np.zeros(22, dtype=np.float32)

        if state == SoccerState.WALK:
            # Velocity policy controls legs
            leg_action = self._run_vel_policy(obs_vel)
            full_action[LEG_IDX] = leg_action
            # Head stays at default
            full_action[HEAD_IDX] = DEFAULT_HEAD_POS

        elif state == SoccerState.TRACK:
            # Velocity policy controls legs
            leg_action = self._run_vel_policy(obs_vel)
            full_action[LEG_IDX] = leg_action
            # Head tracking policy controls head
            head_action = self._run_head_policy(obs_head)
            full_action[HEAD_IDX] = head_action

        elif state == SoccerState.KICK:
            # Kicking policy controls legs
            kick_action = self._run_kick_policy(obs_kick)
            full_action[LEG_IDX] = kick_action
            # Head looks at ball
            head_action = self._run_head_policy(obs_head)
            full_action[HEAD_IDX] = head_action
            self.kick_timer += 1

        self.last_leg_action = full_action[LEG_IDX]
        self.last_head_action = full_action[HEAD_IDX]
        return full_action

    def _run_vel_policy(self, obs):
        """Run velocity policy for leg control."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.vel_policy(obs_t).cpu().numpy().flatten()
        return (action * ACTION_SCALE).astype(np.float32)

    def _run_head_policy(self, obs):
        """Run head tracking policy for head control."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.head_policy(obs_t).cpu().numpy().flatten()
        return (action * 0.5).astype(np.float32)  # head has different scale

    def _run_kick_policy(self, obs):
        """Run kicking policy for leg control."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.kick_policy(obs_t).cpu().numpy().flatten()
        return (action * ACTION_SCALE).astype(np.float32)

    def build_obs_vel(self, base_lin_vel, base_ang_vel, projected_gravity,
                      cmd_vel, joint_pos, joint_vel):
        """Build velocity policy observation (48-dim)."""
        return np.concatenate([
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            cmd_vel,           # 3
            joint_pos,         # 12
            joint_vel,         # 12
            self.last_leg_action,  # 12
        ]).astype(np.float32)

    def build_obs_head(self, detection, head_pos, head_vel,
                       base_ang_vel):
        """Build head tracking observation (12-dim).

        detection = (visible, du, dv) from the P3 detector — the ONLY ball
        signal (no ground truth). visible==0 is what arms the CCW search.
        """
        return np.concatenate([
            np.asarray(detection, dtype=np.float32).reshape(3),  # visible, du, dv
            head_pos,        # 2
            head_vel,        # 2
            base_ang_vel,    # 3
            self.last_head_action,  # 2
        ]).astype(np.float32)

    def build_obs_kick(self, ball_distance, ball_angle,
                       base_lin_vel, base_ang_vel, projected_gravity,
                       joint_pos, joint_vel):
        """Build kicking observation (48-dim)."""
        return np.concatenate([
            np.array([ball_distance], dtype=np.float32),  # 1
            np.array([ball_angle], dtype=np.float32),     # 1
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            joint_pos,         # 12
            joint_vel,         # 12
            self.last_leg_action,  # 12
        ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="K1 Soccer Multi-Policy Composer")
    parser.add_argument("--vel-policy", default="models/k1_velocity_policy.pt")
    parser.add_argument("--head-policy", default="models/k1_head_tracking_policy.pt")
    parser.add_argument("--kick-policy", default="models/k1_kicking_policy.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    composer = PolicyComposer(
        vel_path=os.path.join(ws, args.vel_policy),
        head_path=os.path.join(ws, args.head_policy),
        kick_path=os.path.join(ws, args.kick_policy),
        device=args.device,
    )

    print("[soccer] Multi-policy composer loaded")
    print(f"[soccer] Velocity policy: {args.vel_policy}")
    print(f"[soccer] Head tracking policy: {args.head_policy}")
    print(f"[soccer] Kicking policy: {args.kick_policy}")
    print("[soccer] Ready for integration with Isaac Sim fleet")


if __name__ == "__main__":
    main()
