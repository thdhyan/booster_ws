#!/usr/bin/env python3
"""HMD-Poser body -> SOMA -> K1 arm joints, live (replaces the K1 arm IK when
the teleop node runs with ``--arm-src soma``). Ported from G1_sim
``teleop/soma_retarget_node.py`` (9c3c429); the ``booster_k1`` target comes from
``scripts/mk_soma_k1_target.py`` (branch feat/retarget-soma-k1).

    /{ns}/teleop/smpl_pose (Float32MultiArray, 22 x 3x3 row-major; [0] = SMPL root
        in the canonical frame: y up, facing +z, the operator's pelvis yaw removed)
      -> SMPL forward (smplx, neutral)                       posed vertices
      -> SOMA-X PoseInversion.fit (SMPL identity)            78 SOMA local rotations
      -> NVIDIA soma-retargeter (Newton IK, one reseeded     K1 22-DoF joint_q
         step per message, CUDA graph)
      -> /{ns}/teleop/upper_body_cmd (8 arm joints, rate limited) while
         /{ns}/teleop/status says arms on, not paused, arm_source soma
         (the teleop node keeps sending the head joints on the same topic)
      -> /{ns}/teleop/soma_k1 (all joints, always; for inspection)

Runs in G1_sim's soma-retarget image (docker/Dockerfile.soma-retarget) with the
booster_k1 target generated into soma_retargeter's assets. The SMPL
model is licensed: mount it, never commit it (SMPL_MODEL, default
/body_models/smpl_np/SMPL_NEUTRAL.pkl - a numpy-only re-pickle, so chumpy is
not needed at load time).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np

SMPL_MODEL = os.environ.get("SMPL_MODEL", "/body_models/smpl_np/SMPL_NEUTRAL.pkl")
SOMA_ASSETS = os.environ.get("SOMA_ASSETS", "/opt/SOMA-X/assets")
# URDF names, sort prefix included (K1_22dof.urdf)
ARM_JOINTS = [f"{p}{s}_{j}" for s in ("Left", "Right")
              for p, j in (("A", "Shoulder_Pitch"), ("", "Shoulder_Roll"), ("", "Elbow_Pitch"), ("", "Elbow_Yaw"))]


class SmplToSoma:
    """SMPL pose (root + 21 body rotations) -> 78 SOMA joint local rotations."""

    def __init__(self, device: str = "cuda", body_iters: int = 2, full_iters: int = 1) -> None:
        import smplx
        import torch
        from soma.body import SOMALayer
        from soma.fitting.pose_inversion import PoseInversion

        self.torch, self.device = torch, device
        self.iters = {"body_iters": body_iters, "finger_iters": 0, "full_iters": full_iters}
        self.smpl = smplx.create(model_type="smpl", model_path=SMPL_MODEL, batch_size=1).to(device)
        # low_lod layer up front: PoseInversion's own low-LOD fallback has a broken
        # relative import at SOMA-X cc1f396 (fitting/pose_inversion.py `from .body`).
        soma = SOMALayer(SOMA_ASSETS, identity_model_type="smpl", identity_model_kwargs={"model_path": SMPL_MODEL},
                         device=device, mode="warp", low_lod=True)
        self.inv = PoseInversion(soma, low_lod=True)
        self.inv.prepare_identity(torch.zeros(1, 10, device=device))

    def __call__(self, R: np.ndarray) -> tuple[np.ndarray, float]:
        """(22, 3, 3) -> ((78, 3, 3) SOMA local rotations, mean vertex error m)."""
        from scipy.spatial.transform import Rotation

        aa = Rotation.from_matrix(R).as_rotvec().astype(np.float32)
        body = np.zeros((1, 69), np.float32)                  # SMPL joints 1..23; hands (22, 23) stay rest
        body[0, :63] = aa[1:].reshape(-1)
        t = self.torch
        with t.no_grad():
            out = self.smpl(body_pose=t.from_numpy(body).to(self.device),
                            global_orient=t.from_numpy(aa[None, 0]).to(self.device),
                            betas=t.zeros(1, 10, device=self.device))
        r = self.inv.fit(out.vertices, **self.iters)
        return r["rotations"][0].cpu().numpy(), float(r["per_vertex_error"].mean())


class LiveRetargeter:
    """soma-retargeter's SomaRetargetingPipeline, stepped one frame at a time: the
    same model, objectives and BaseIKSolver its execute() builds, with the solve
    captured as a CUDA graph. ``reseed`` (default) restarts every frame from the
    retargeter's reference pose: warm-starting from the last solution got stuck
    after an odd pose (right arm at shoulder -2.6 rad for arms-forward, live
    2026-09-28), a reseeded solve does not depend on history."""

    def __init__(self, robot: str = "booster_k1", iterations: int = 48, reseed: bool = True) -> None:
        import newton
        import warp as wp
        from soma_retargeter.io.bvh import load_bvh
        from soma_retargeter.pipelines import utils as pu
        from soma_retargeter.pipelines.ik_solver import BaseIKSolver
        from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
        from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

        self.wp = wp
        self.skel, _ = load_bvh(pu.get_source_zero_pose_asset_path(pu.get_source_type_from_str("soma")))
        p = SomaRetargetingPipeline(self.skel, "soma", robot)
        model = p._build_model(1)
        self.state = model.state()
        self.pos, self.rot, joint_limit, _ = p._create_ik_objectives(1, model, self.state)
        self.solver = BaseIKSolver(p.ik_model, 1, model, position_objectives=self.pos, rotation_objectives=self.rot,
                                   extra_objectives=[joint_limit] if p.joint_limit_weight > 0 else [],
                                   reference_joint_q=p.human_robot_scaler.reference_joint_q)
        self.p, self.iterations, self.reseed = p, iterations, reseed
        self.joint_q = wp.clone(self.solver.joint_q)
        self.seed = wp.clone(self.solver.joint_q)                   # reference joint_q
        self.offset = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())   # app default
        self.ref = np.array([list(t) for t in self.skel.reference_local_transforms], np.float32)       # (78, 7)
        with wp.ScopedCapture() as cap:
            self._solve()
        self.graph = cap.graph
        types = p.ik_model.joint_type.numpy()
        labels = list(p.ik_model.joint_label)
        starts = p.ik_model.joint_q_start.numpy()
        self.q_index = {lab.split("/")[-1]: int(starts[i]) for i, lab in enumerate(labels)
                        if types[i] == int(newton.JointType.REVOLUTE)}

    def _solve(self) -> None:
        self.solver.solve(self.state, self.iterations)
        self.wp.copy(self.joint_q, self.solver.joint_q)

    def __call__(self, soma_R: np.ndarray) -> dict[str, float]:
        """(78, 3, 3) SOMA local rotations -> {robot joint name: angle}."""
        from scipy.spatial.transform import Rotation
        from soma_retargeter.animation.animation_buffer import AnimationBuffer

        wp = self.wp
        local = self.ref.copy()
        local[:, 3:7] = Rotation.from_matrix(soma_R).as_quat()     # xyzw; positions stay the reference skeleton's
        frames = np.zeros((1, len(local)), dtype=wp.transform)
        frames[0] = [wp.transform(*t) for t in local]
        buf = AnimationBuffer(self.skel, 1, 60.0, frames)
        eff = np.asarray(self.p.human_robot_scaler.compute_effectors_from_buffer(buf, True, self.offset), np.float32)
        eff = eff.reshape(1, -1, 7)[:, self.p.target_effector_indices]
        for i in range(len(self.pos)):
            self.pos[i].set_target_positions(wp.array(eff[:, i, 0:3], dtype=wp.vec3))
            self.rot[i].set_target_rotations(wp.array(eff[:, i, 3:7], dtype=wp.vec4))
        if self.reseed:
            wp.copy(self.solver.joint_q, self.seed)
            self.solver.solver.reset()
        wp.capture_launch(self.graph)
        q = self.joint_q.numpy()
        self.p.joint_limit_clamper.apply(wp.array(q, dtype=wp.float32))
        q = q[0]
        return {name: float(q[i]) for name, i in self.q_index.items()}


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray, String

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ns", default="k1_0")
    ap.add_argument("--robot", default="booster_k1", help="soma-retargeter robot target")
    ap.add_argument("--max-speed", type=float, default=2.0, help="rad/s per arm joint towards the retargeted pose")
    ap.add_argument("--body-iters", type=int, default=1)
    ap.add_argument("--full-iters", type=int, default=0)
    a = ap.parse_args()

    fit = SmplToSoma(body_iters=a.body_iters, full_iters=a.full_iters)
    retarget = LiveRetargeter(a.robot)
    missing = [j for j in ARM_JOINTS if j not in retarget.q_index]
    assert not missing, f"retargeter joints lack {missing}; has {sorted(retarget.q_index)}"

    rclpy.init()
    ns = a.robot_ns
    node = Node("k1_soma_retarget")
    pub_arm = node.create_publisher(JointState, f"/{ns}/teleop/upper_body_cmd", 10)
    pub_all = node.create_publisher(JointState, f"/{ns}/teleop/soma_k1", 10)
    latest: dict = {}
    status = {"arms": False, "paused": True, "arm_source": "ik"}
    joints: dict[str, float] = {}
    node.create_subscription(Float32MultiArray, f"/{ns}/teleop/smpl_pose",
                             lambda m: latest.__setitem__("R", np.asarray(m.data, np.float64).reshape(22, 3, 3)), 10)
    node.create_subscription(String, f"/{ns}/teleop/status", lambda m: status.update(json.loads(m.data)), 10)
    node.create_subscription(JointState, f"/{ns}/joint_states",
                             lambda m: joints.update(zip(m.name, (float(p) for p in m.position))), 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    node.get_logger().info(f"ready: SMPL {SMPL_MODEL} -> SOMA -> {a.robot} ({len(retarget.q_index)} joints)")

    sent: np.ndarray | None = None
    t_prev, n, t_log = time.time(), 0, time.time()
    while rclpy.ok():
        R = latest.pop("R", None)
        if R is None:
            time.sleep(0.005)
            continue
        t0 = time.perf_counter()
        soma_R, err = fit(R)
        t1 = time.perf_counter()
        q = retarget(soma_R)
        t2 = time.perf_counter()
        pub_all.publish(JointState(name=list(q), position=list(q.values())))
        now = time.time()
        dt, t_prev = min(now - t_prev, 0.2), now
        active = status.get("arms") and not status.get("paused") and status.get("arm_source") == "soma"
        target = np.array([q[j] for j in ARM_JOINTS])
        if not active:
            sent = None
        else:
            if sent is None:                                        # start from where the arms are
                sent = np.array([joints.get(j, 0.0) for j in ARM_JOINTS])
            sent = sent + np.clip(target - sent, -a.max_speed * dt, a.max_speed * dt)
            msg = JointState(name=ARM_JOINTS, position=[float(v) for v in sent])
            msg.header.stamp = node.get_clock().now().to_msg()
            pub_arm.publish(msg)
        n += 1
        if now - t_log > 5.0:
            node.get_logger().info(f"{n / (now - t_log):.1f} Hz  fit {1e3 * (t1 - t0):.0f} ms (err {100 * err:.1f} cm)  "
                                   f"retarget {1e3 * (t2 - t1):.0f} ms  arms {'ON' if active else 'off'}")
            n, t_log = 0, now


if __name__ == "__main__":
    main()
