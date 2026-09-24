"""MDP terms for P3 head tracking.

Detection pipeline (the only ball signal the policy sees):
  * camera path  : head cam (Head_2) RGB -> YOLO `sports ball` bbox centre
                   -> (visible, du, dv), refreshed every DETECT_EVERY control
                   steps (10 Hz at 50 Hz control), held between detections.
  * fallback     : ball pose projected through the current head joint angles
                   with a pinhole model (same units, tiny noise/dropout) so
                   smoke/deployability still work without ultralytics/weights.

The head camera *tracks the head joints* (mounted on Head_2), so turning the
head moves the view - exactly the task ("keep the detection in frame").

State buffers live in ``env.head_track`` (SimpleNamespace), created lazily so
ManagerBasedRLEnv stays untouched.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from ..kick.mdp import BALL_RADIUS, ball_pos_in_robot_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]
HEAD_CAM_PRIM = "head_cam"
DETECT_EVERY = 5          # control steps between detections (50 Hz -> 10 Hz)
DETECT_EPS = 2e-3         # per-env chunking to bound YOLO batch memory
CAM_HFOV_DEG = 69.4       # ZED 2c @ 320x240 crop (approx, f=238px)
YOLO_CLASS = 32           # COCO "sports ball"
_yolo_model = None        # lazily loaded ultralytics YOLO (None -> fallback)
_yolo_failed = False


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    st = getattr(env, "head_track", None)
    if st is None:
        n = env.num_envs
        dev = env.device
        st = SimpleNamespace(
            yolo=torch.zeros((n, 3), device=dev),      # (visible, du, dv), held
            visible=torch.zeros(n, dtype=torch.bool, device=dev),
            lost_steps=torch.zeros(n, dtype=torch.long, device=dev),
            ball_dir=torch.zeros((n, 3), device=dev),  # unit XY heading
            ball_speed=0.0,                             # curriculum-owned (m/s)
            frame=0,
            yolo_active=False,
        )
        env.head_track = st
    return st


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _load_yolo():
    """Load yolov8n (COCO) once. Returns None when unavailable -> fallback."""
    global _yolo_model, _yolo_failed
    if _yolo_model is not None or _yolo_failed:
        return _yolo_model
    try:
        import os

        from ultralytics import YOLO

        weights = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
        _yolo_model = YOLO(weights)
        print(f"[head] YOLO loaded from {weights} (COCO sports ball)")
    except Exception as exc:  # no ultralytics / no weights / no net
        _yolo_failed = True
        print(f"[head] YOLO unavailable ({exc}); using geometric detection proxy")
    return _yolo_model


def _project_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Geometric detection proxy: project ball into the (head-tracking) camera.

    Returns (n, 3): visible, du, dv  with du/dv in [-1, 1] (image edges).
    Uses true ball pose (reward-time GT is fine; this is only a *sensor model*
    fallback and the policy still never sees pose/velocity directly).
    """
    st = _state(env)
    ball_rf = ball_pos_in_robot_frame(env)                      # (n, 3) xyzw
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(HEAD_JOINTS, preserve_order=True)
    head_pos = robot.data.joint_pos[:, ids]                      # (n, 2)
    yaw_q, pitch_q = head_pos[:, 0], head_pos[:, 1]

    # Camera sits slightly in front/above the head pitch axis (Head_2 origin).
    cam_off = torch.tensor([0.10, 0.0, 0.12], device=env.device)
    rel = ball_rf - cam_off

    hfov = math.radians(CAM_HFOV_DEG)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * 240.0 / 320.0)

    # azimuth/elevation relative to the head pointing direction
    yaw_ball = torch.atan2(rel[:, 1], rel[:, 0])
    pitch_ball = torch.atan2(rel[:, 2] - 0.55, torch.hypot(rel[:, 0], rel[:, 1]))
    yaw_err = yaw_ball - yaw_q
    pitch_err = pitch_ball - pitch_q
    du = yaw_err / (hfov / 2.0)
    dv = pitch_err / (vfov / 2.0)

    in_front = rel[:, 0] > 0.05
    visible = in_front & (du.abs() < 1.0) & (dv.abs() < 1.0)
    du = du.clamp(-1.5, 1.5) + 0.02 * torch.randn_like(du)
    dv = dv.clamp(-1.5, 1.5) + 0.02 * torch.randn_like(dv)
    # 5 % detector dropouts
    keep = torch.rand_like(du) > 0.05
    visible = visible & keep

    out = torch.stack([visible.float(), du, dv], dim=-1)
    st.visible = visible
    return out


def _detect_yolo(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Real camera path: batched YOLO over the head-cam RGB buffer."""
    st = _state(env)
    cam = env.scene.sensors.get(HEAD_CAM_PRIM)
    rgb = cam.data.output["rgb"].torch[..., :3]                # (n, H, W, 3) uint8
    n = rgb.shape[0]
    out = torch.zeros((n, 3), device=rgb.device)
    model = _load_yolo()
    with torch.inference_mode():
        for s in range(0, n, 256):
            e = min(s + 256, n)
            batch = rgb[s:e].permute(0, 3, 1, 2).contiguous()
            res = model.predict(batch, verbose=False, device=0, classes=[YOLO_CLASS])
            for j, r in enumerate(res):
                boxes = getattr(r, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                best = int(torch.argmax(boxes.conf).item())
                cx = float((boxes.xyxy[best, 0] + boxes.xyxy[best, 2]) / 2.0)
                cy = float((boxes.xyxy[best, 1] + boxes.xyxy[best, 3]) / 2.0)
                w, h = rgb.shape[2], rgb.shape[1]
                out[s + j, 0] = 1.0
                out[s + j, 1] = 2.0 * cx / w - 1.0
                out[s + j, 2] = 1.0 - 2.0 * cy / h
    st.visible = out[:, 0] > 0.5
    return out


def ball_detection(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Obs term: (visible, du, dv) from the detector, held between detections."""
    st = _state(env)
    cam = env.scene.sensors.get(HEAD_CAM_PRIM)
    model = _load_yolo() if cam is not None else None
    if model is not None and st.frame % DETECT_EVERY == 0:
        try:
            st.yolo = _detect_yolo(env)
            st.yolo_active = True
        except Exception as exc:  # any detector hiccup -> hold + proxy
            print(f"[head] YOLO inference failed once ({exc}); falling back")
            st.yolo = _project_ball(env)
            st.yolo_active = False
    elif st.frame % DETECT_EVERY == 0 or not st.yolo_active:
        st.yolo = _project_ball(env)
    # lost / found bookkeeping for the CCW search consumer
    now = st.yolo[:, 0] > 0.5
    st.lost_steps = torch.where(now, torch.zeros_like(st.lost_steps), st.lost_steps + 1)
    st.frame += 1
    return st.yolo.clone()


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def ball_centered(env: ManagerBasedRLEnv, std: float = 0.35) -> torch.Tensor:
    """Primary: exp kernel on the *detection* offset (du, dv). No GT."""
    st = _state(env)
    du, dv = st.yolo[:, 1], st.yolo[:, 2]
    return torch.exp(-(du * du + dv * dv) / (std * std)).unsqueeze(-1)


def ball_in_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary: the detector currently sees the ball."""
    st = _state(env)
    return st.yolo[:, 0:1]


def track_ball_angle_exp(env: ManagerBasedRLEnv, std: float = 0.35) -> torch.Tensor:
    """Geometric head-pointing reward (true ball pose, reward-time only)."""
    st = _state(env)
    ball_rf = ball_pos_in_robot_frame(env)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(HEAD_JOINTS, preserve_order=True)
    yaw_q, pitch_q = robot.data.joint_pos[:, ids, 0], robot.data.joint_pos[:, ids, 1]
    yaw_err = torch.atan2(ball_rf[:, 1], ball_rf[:, 0]) - yaw_q
    pitch_err = torch.atan2(ball_rf[:, 2] - 0.55, torch.hypot(ball_rf[:, 0], ball_rf[:, 1])) - pitch_q
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
    pitch_err = torch.atan2(torch.sin(pitch_err), torch.cos(pitch_err))
    return torch.exp(-(yaw_err * yaw_err + pitch_err * pitch_err) / (std * std)).unsqueeze(-1)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones((env.num_envs, 1), device=env.device)


# ---------------------------------------------------------------------------
# curriculum: ball speed 0 -> v_max (static centring first, then rolling)
# ---------------------------------------------------------------------------
def ball_speed_curriculum(env: ManagerBasedRLEnv, start_iter: int, end_iter: int, v_max: float) -> float:
    st = _state(env)
    frac = min(max(env.common_step_counter / max(end_iter - start_iter, 1), 0.0), 1.0)
    if env.common_step_counter < start_iter:
        frac = 0.0
    st.ball_speed = v_max * frac
    return st.ball_speed


# ---------------------------------------------------------------------------
# reset / maintenance events
# ---------------------------------------------------------------------------
def reset_ball_for_tracking(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Ball 1.0-2.5 m ahead, then rolls at the curriculum speed."""
    st = _state(env)
    n = len(env_ids)
    dev = env.device
    dist = 1.0 + 1.5 * torch.rand(n, device=dev)
    lat = 0.5 * (2.0 * torch.rand(n, device=dev) - 1.0)
    pos = torch.zeros((n, 3), device=dev)
    pos[:, 0], pos[:, 1], pos[:, 2] = dist, lat, BALL_RADIUS + 0.01

    theta = (torch.rand(n, device=dev) - 0.5) * 1.2 - math.pi / 2.0  # mostly lateral
    direction = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(n, device=dev)], dim=-1)
    st.ball_dir[env_ids] = direction

    ball = env.scene["ball"]
    root = ball.data.default_root_state.torch[env_ids].clone()
    root[:, :3] = pos + env.scene.env_origins[env_ids]
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    root[:, 7:10] = direction * st.ball_speed
    root[:, 10:13] = 0.0
    ball.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)

    st.yolo[env_ids] = 0.0
    st.lost_steps[env_ids] = 0


def maintain_ball_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Interval: re-assert the commanded roll speed (friction would eat it)."""
    st = _state(env)
    ball = env.scene["ball"]
    vel = ball.data.root_lin_vel_w.torch[env_ids].clone()
    vel[:, :2] = st.ball_dir[env_ids, :2] * st.ball_speed
    vel[:, 2] = 0.0
    ball.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# CCW search: conditional in-place rotation command when the ball is lost
# ---------------------------------------------------------------------------
CCW_OMEGA = 0.6        # rad/s, in place
LOST_AFTER_STEPS = 25  # 0.5 s at 50 Hz without a detection


def ccw_search_command(lin_vel: torch.Tensor, ang_vel: torch.Tensor, lost_steps: torch.Tensor,
                       omega: float = CCW_OMEGA, lost_after: int = LOST_AFTER_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """If a camera has no ball for ``lost_after`` steps -> (0, 0, +omega) command.

    Counter-clockwise = positive yaw (viewed from +z). Linear velocity is zeroed
    so the robot rotates in place without drifting while searching for an
    out-of-FOV ball. Otherwise the sampled velocity command passes through.

    Used by the compose runtime and (later) the P4-student/P5 env command term.
    """
    lost = (lost_steps >= lost_after).unsqueeze(-1)
    lin = torch.where(lost, torch.zeros_like(lin_vel), lin_vel)
    ang = torch.where(lost, torch.full_like(ang_vel, omega), ang_vel)
    return lin, ang
