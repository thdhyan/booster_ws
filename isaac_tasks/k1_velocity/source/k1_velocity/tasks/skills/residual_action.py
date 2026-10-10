"""Residual skill action on top of the FROZEN velocity student.

The velocity student (models/k1_velocity_student_gaitfix.pt, a distillation
checkpoint: student_state_dict = MLP 500 -> 512 -> 256 -> 128 -> 12, ELU, no obs
normalisation) walks from a 3-dim velocity command. A skill policy drives it:

    action = [vx, vy, wz | residual 12 legs | head 2 (optional)]

* vx/vy/wz in [-1, 1] scale to the student's training ranges (+-0.5, +-0.5, +-1.0)
  and are written into the student's command slot.
* legs: target = default + clip(student_out + residual_scale * residual, +-1)
  (the student's own JointPositionAction: scale 1.0, clip +-1).
* head: target = default + head_scale * a (absolute, the student never moved it).

The student's 50-dim frame is rebuilt here in its training order
    lin 3 | ang 3 | grav 3 | cmd 3 | leg_pos_rel 12 | leg_vel 12 | last_action 12 | clock 2
and stacked over 10 steps TERM-MAJOR (each term's 10 frames oldest -> newest, then
the next term), which is how ObservationManager group history concatenates.
``last_action`` is the student's OWN raw output, not student + residual, so the
student keeps seeing the action stream it was trained on.

Leg joints resolve with preserve_order=False (articulation order), matching the
velocity env's JointPositionAction and joint_pos/vel obs terms; see the
FrozenBaseVelocityAction note in push_mdp for what the wrong order does.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from k1_velocity.tasks.velocity import gait_clock

try:
    import isaaclab_tasks.core.velocity.mdp as vmdp
except ImportError:  # older layout
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp

K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
K1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]

CMD_RANGES = (0.5, 0.5, 1.0)     # velocity env UniformVelocityCommandCfg ranges
HISTORY = 10                     # STUDENT_HISTORY_LEN
FRAME_TERMS = (3, 3, 3, 3, 12, 12, 12, 2)


def load_student(path: str, device) -> nn.Module:
    """Rebuild the student MLP from a distillation checkpoint (or load TorchScript)."""
    try:
        return torch.jit.load(path, map_location=device).eval()
    except RuntimeError:
        pass
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))
    sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers: list[nn.Module] = []
    for j, i in enumerate(idx):
        w = sd[f"{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict({f"{2 * j}.{p}": sd[f"{i}.{p}"] for j, i in enumerate(idx) for p in ("weight", "bias")})
    return net.to(device).eval()


def stack_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(n, H, 50) frame history (oldest first) -> (n, 50*H) term-major layout."""
    out, k = [], 0
    for d in FRAME_TERMS:
        out.append(hist[:, :, k:k + d].reshape(hist.shape[0], -1))
        k += d
    return torch.cat(out, dim=-1)


class FrozenStudentResidualAction(ActionTerm):
    def __init__(self, cfg: "FrozenStudentResidualActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        dev = self._asset.device
        self._policy = load_student(cfg.student_path, dev)
        leg_ids, _ = self._asset.find_joints(K1_LEG_JOINTS, preserve_order=False)
        self._leg_ids = torch.as_tensor(leg_ids, dtype=torch.long, device=dev)
        self._default12 = self._joint("default_joint_pos")[:, self._leg_ids].clone()
        self._n_head = 2 if cfg.head else 0
        if cfg.head:
            head_ids, _ = self._asset.find_joints(K1_HEAD_JOINTS, preserve_order=True)
            self._head_ids = torch.as_tensor(head_ids, dtype=torch.long, device=dev)
            self._default_head = self._joint("default_joint_pos")[:, self._head_ids].clone()
        n = env.num_envs
        self._dim = 3 + 12 + self._n_head
        self._raw = torch.zeros(n, self._dim, device=dev)
        self._processed = torch.zeros(n, self._dim, device=dev)
        self.cmd = torch.zeros(n, 3, device=dev)              # scaled velocity command
        self.student_out = torch.zeros(n, 12, device=dev)     # student's raw output (its last_action)
        self.residual = torch.zeros(n, 12, device=dev)        # scaled residual actually added
        self._hist = torch.zeros(n, HISTORY, sum(FRAME_TERMS), device=dev)
        self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self._leg_targets = self._default12.clone()
        self._head_targets = self._default_head.clone() if cfg.head else None
        self._scale = torch.tensor(CMD_RANGES, device=dev)
        print(f"[skills] frozen student loaded: {cfg.student_path} action_dim={self._dim} "
              f"residual_scale={cfg.residual_scale} head={cfg.head}")

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def _joint(self, name):
        x = getattr(self._asset.data, name)
        return x.torch if hasattr(x, "torch") else x

    def _frame(self) -> torch.Tensor:
        env = self._env
        lin = vmdp.base_lin_vel(env)
        ang = vmdp.base_ang_vel(env)
        grav = vmdp.projected_gravity(env)
        jp, jv = self._joint("joint_pos"), self._joint("joint_vel")
        leg_pos = jp[:, self._leg_ids] - self._default12
        leg_vel = jv[:, self._leg_ids]
        clock = gait_clock.phase_clock_from_magnitude(env, self.cmd[:, :2].norm(dim=-1))
        return torch.cat([lin, ang, grav, self.cmd, leg_pos, leg_vel, self.student_out, clock], dim=-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        self._processed[:] = a
        self.cmd = a[:, :3] * self._scale
        frame = self._frame()
        fresh = self._fresh
        self._hist = torch.roll(self._hist, shifts=-1, dims=1)
        self._hist[:, -1] = frame
        if fresh.any():   # CircularBuffer semantics: first push after reset fills every slot
            self._hist[fresh] = frame[fresh].unsqueeze(1).expand(-1, HISTORY, -1)
            self._fresh[:] = False
        with torch.inference_mode():
            out = self._policy(stack_term_major(self._hist))
        self.student_out = out.clone()
        self.residual = cfg.residual_scale * a[:, 3:15]
        self._leg_targets = self._default12 + (out + self.residual).clamp(-1.0, 1.0)
        if cfg.head:
            self._head_targets = self._default_head + cfg.head_scale * a[:, 15:17]

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids.to(torch.int32))
        if self.cfg.head:
            self._asset.set_joint_position_target(self._head_targets, joint_ids=self._head_ids.to(torch.int32))

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.student_out[env_ids] = 0.0
        self.residual[env_ids] = 0.0
        self.cmd[env_ids] = 0.0
        self._raw[env_ids] = 0.0
        self._fresh[env_ids] = True


@configclass
class FrozenStudentResidualActionCfg(ActionTermCfg):
    class_type: type = FrozenStudentResidualAction
    asset_name: str = "robot"
    student_path: str = "models/k1_velocity_student_gaitfix.pt"
    residual_scale: float = 0.1
    head: bool = True
    head_scale: float = 1.0
