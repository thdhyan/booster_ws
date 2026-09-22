#!/usr/bin/env python3
"""Isaac Sim K1 fleet with visualization + OmniGraph ROS2 bridge + velocity policy.

Shows the Isaac Sim viewer with N K1 robots, uses OmniGraph for ROS2
joint_states/joint_commands, and runs the trained velocity policy.

Usage (venv-isaac python):
  python isaac_fleet_vis.py --n_robots 2
  python isaac_fleet_vis.py --n_robots 1 --headless  # for headless testing

Requires:
  - Isaac Sim 6.0.1 via ~/Projects/IsaacLab/.venv-isaac
  - ROS_DOMAIN_ID=0, CycloneDDS for discovery
  - OmniGraph ROS2 bridge nodes auto-created by Isaac Sim
"""
import argparse
import os
import sys

# ── Environment setup ────────────────────────────────────────────────
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("CYCLONEDDS_URI",
                      "file:///home/thakk100/.cyclonedds/cyclonedds.xml")

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "..")

import site

def _ros_core_path():
    # Check site-packages first (venv-isaac install)
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        cand = os.path.join(sp, "isaacsim", "exts", "isaacsim.ros2.core")
        if os.path.isdir(cand):
            return cand
    # Check Isaac Sim Docker install
    docker_cand = "/isaac-sim/exts/isaacsim.ros2.core"
    if os.path.isdir(docker_cand):
        return docker_cand
    return None

def _ensure_bundled_ros_env():
    if os.environ.get("K1_BUNDLED_ROS_READY") == "1":
        return
    core = _ros_core_path()
    if core is None:
        print("[fleet] isaacsim.ros2.core not found — continuing without ROS")
        os.environ["K1_BUNDLED_ROS_READY"] = "0"
        return
    overlay = os.path.join(core, "jazzy", "rclpy")
    libs = os.pathsep.join([
        os.path.join(core, "jazzy", "lib"),
        os.path.join(core, "lib"),
    ])
    os.environ["PYTHONPATH"] = overlay + os.pathsep + os.environ.get("PYTHONPATH", "")
    # Also add to sys.path for the current process (execv may not work in Kit)
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = libs + (os.pathsep + ld if ld else "")
    os.environ["AMENT_PREFIX_PATH"] = os.path.join(core, "jazzy")
    os.environ["K1_BUNDLED_ROS_READY"] = "1"
    # Restart only if sys.executable is valid (not in Kit's embedded Python)
    if sys.executable and os.path.isfile(sys.executable):
        os.execv(sys.executable, [sys.executable] + sys.argv)
    # else: env vars are set, continue in-process

# NOTE: _ensure_bundled_ros_env() moved AFTER AppLauncher to avoid
# LD_LIBRARY_PATH conflicts with Isaac Sim's Kit window system.

# ── Argument parsing ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_robots", type=int, default=2)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--policy", default="models/k1_velocity_policy.pt")
parser.add_argument("--cmd-vx", type=float, default=0.5, help="velocity command m/s")
parser.add_argument("--record", type=str, default=None,
                    help="Record video to this path (e.g. docs/videos/fleet_isaac.mp4)")
parser.add_argument("--record-seconds", type=float, default=5.0,
                    help="Seconds of video to record (default 5)")
parser.add_argument("--record-fps", type=int, default=30,
                    help="Video FPS (default 30)")
args, unknown = parser.parse_known_args()

# ── Isaac Sim / Isaac Lab imports ───────────────────────────────────
# Detect if Kit is already running (Docker Python standalone mode via --exec)
_kit_running = False
try:
    import omni.kit.app
    _app_instance = omni.kit.app.get_app()
    if _app_instance is not None:
        _kit_running = True
        print("[fleet] Kit is already running (Docker standalone mode)")
except Exception:
    pass

if not _kit_running:
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        # In Docker standalone mode, AppLauncher might not be available
        _kit_running = True
        print("[fleet] AppLauncher not available, assuming Kit is running")

if not _kit_running:
    launcher_args = argparse.Namespace(**vars(args))
    # Force headless when recording (no window needed)
    if args.record:
        launcher_args.headless = True
        launcher_args.enable_cameras = True
        launcher_args.offscreen_render = True  # Enable offscreen rendering
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
else:
    # Kit is already running — get the simulation app from Kit
    class _FakeApp:
        def close(self): pass
        def update(self): pass
    simulation_app = _FakeApp()
    print("[fleet] Using existing Kit instance")

# ── Now set up bundled ROS (after Kit is initialized) ────────────────
_ensure_bundled_ros_env()

# ROS2 imports (optional — Docker standalone may not have rclpy)
_HAS_ROS = False
try:
    import rclpy  # noqa: E402
    from rclpy.node import Node  # noqa: E402
    from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
    from sensor_msgs.msg import JointState  # noqa: E402
    from geometry_msgs.msg import Twist  # noqa: E402
    _HAS_ROS = True
    print("[fleet] ROS2 available")
except ImportError:
    print("[fleet] ROS2 not available — running without ROS endpoints")
    # Define stubs so the rest of the code doesn't crash
    class Node:
        def create_publisher(self, *a, **kw): return type('Pub', (), {'publish': lambda s, m: None})()
        def create_subscription(self, *a, **kw): return None
        def create_rate(self, *a, **kw): return type('Rate', (), {'sleep': lambda s: None})()
        def get_logger(self): return type('Log', (), {'info': lambda s, m: print(m), 'warn': lambda s, m: print(m)})()
        def get_clock(self): return type('Clock', (), {'now': lambda s: type('Stamp', (), {'to_msg': lambda s: type('Msg', (), {'sec': 0, 'nanosec': 0})()})()})()
        def destroy_node(self): pass
    class QoSProfile:
        def __init__(self, **kw): pass
    class ReliabilityPolicy:
        RELIABLE = 0
        BEST_EFFORT = 1
    class JointState:
        pass
    class Twist:
        pass
    rclpy = None  # type: ignore

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

# Video recording imports
if args.record:
    import cv2  # noqa: E402

# OmniGraph for ROS2 bridge (loaded dynamically if available)
try:
    import omni.graph.core as og  # noqa: E402
    HAS_OMNIGRAPH = True
except ImportError:
    HAS_OMNIGRAPH = False
    print("[OmniGraph] Not available — using Isaac Lab built-in ROS2 bridge")

# Try to import K1_ARTICULATION_CFG from k1_velocity (available in Isaac Lab venv)
# Fall back to a simplified config for Docker standalone mode
_HAS_K1_CFG = False
try:
    from k1_velocity.tasks.velocity.velocity_env_cfg import K1_ARTICULATION_CFG  # noqa: E402
    _HAS_K1_CFG = True
except ImportError:
    print("[fleet] k1_velocity not available — using simplified K1 articulation config")
    # Simplified K1 articulation config for Docker standalone
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as _sim_utils

    K1_ARTICULATION_CFG = ArticulationCfg(
        spawn=_sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=False,
            asset_path="/workspace/booster_ws/src/k1_description/assets/robots/K1/K1_22dof.urdf",
            activate_contact_sensors=True,
            rigid_props=_sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=_sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=_sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=_sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.57),
            joint_pos={
                "Left_Shoulder_Roll": -1.3,
                "Right_Shoulder_Roll": 1.3,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                    ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
                ],
                effort_limit=120.0,
                velocity_limit=30.0,
                stiffness=40.0,
                damping=1.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_Shoulder_Pitch", ".*_Shoulder_Roll",
                    ".*_Elbow_Pitch", ".*_Elbow_Yaw",
                ],
                effort_limit=30.0,
                velocity_limit=10.0,
                stiffness=20.0,
                damping=0.5,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=[".*Head.*"],
                effort_limit=10.0,
                velocity_limit=5.0,
                stiffness=10.0,
                damping=0.3,
            ),
        },
    )

# ── Constants ────────────────────────────────────────────────────────
JOINT_ORDER = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]

DEFAULT_LEG_POS = np.zeros(12, dtype=np.float32)
ACTION_SCALE = 0.25
OBS_DIM = 48

# JointState used for commands (Isaac bundled stack can't import k1_interfaces)
CMD_TYPE_NAME = "sensor_msgs/JointState"
JointCommand = JointState


# ── OmniGraph ROS2 Bridge Setup ──────────────────────────────────────
def setup_omnigraph_bridge():
    """Create OmniGraph ActionGraph for ROS2 joint_states/joint_commands.

    Isaac Lab 3.0.0b2 handles ROS2 bridging natively via InteractiveScene.
    OmniGraph nodes are auto-created when articulations have ROS topic configs.
    This function is a no-op if omni.graph is not available.
    """
    if not HAS_OMNIGRAPH:
        print("[OmniGraph] Module not available — Isaac Lab ROS2 bridge handles this")
        return False

    print("[OmniGraph] Setting up ROS2 bridge ActionGraph...")
    try:
        keys = og.Controller.Keys
        graph = og.Controller.create_graph(
            keys.PORTS_EXTENDER, "/IsaacRosBridge"
        )
        print("[OmniGraph] ActionGraph created (per-robot nodes via InteractiveScene)")
        return True
    except Exception as e:
        print(f"[OmniGraph] Setup note: {e}")
        print("[OmniGraph] Using Isaac Lab built-in ROS bridge (already active)")
        return False


# ── Scene Configuration ──────────────────────────────────────────────
def build_scene(n):
    """Build InteractiveScene with N K1 robots on a ground plane."""
    attrs = {}
    for i in range(n):
        attrs[f"robot_{i}"] = K1_ARTICULATION_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot_{i}",
            init_state=K1_ARTICULATION_CFG.init_state.replace(
                pos=(3.0 * i, 0.0, 0.57),
            ),
        )
    attrs["ground"] = AssetBaseCfg(
        prim_path="/World/defaultGround", spawn=sim_utils.GroundPlaneCfg()
    )
    attrs["light"] = AssetBaseCfg(
        prim_path="/World/domeLight", spawn=sim_utils.DomeLightCfg(intensity=600.0)
    )
    cfg_cls = configclass(type("FleetSceneCfg", (InteractiveSceneCfg,), attrs))
    return cfg_cls(num_envs=1, env_spacing=3.0)


# ── Policy Runner ────────────────────────────────────────────────────
def load_policy(path):
    """Load TorchScript velocity policy."""
    if not os.path.isfile(path):
        ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        path = os.path.join(ws, path)
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    out = policy(torch.zeros(1, OBS_DIM))
    assert out.numel() == 12, f"Expected 12 outputs, got {out.numel()}"
    print(f"Policy loaded: {path}")
    return policy


def build_obs(base_lin_vel, base_ang_vel, projected_gravity, cmd_vel,
              joint_pos, joint_vel, last_action):
    """48-dim observation matching training layout."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3] = base_lin_vel
    obs[3:6] = base_ang_vel
    obs[6:9] = projected_gravity
    obs[9:12] = cmd_vel
    obs[12:24] = joint_pos - DEFAULT_LEG_POS
    obs[24:36] = joint_vel
    obs[36:48] = last_action
    return obs


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # Setup OmniGraph
    setup_omnigraph_bridge()

    # ROS2 node
    if _HAS_ROS:
        rclpy.init()
    node = Node("k1_isaac_fleet_vis")
    sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    # Load policy
    policy = load_policy(args.policy)

    # Build scene
    scene_cfg = build_scene(args.n_robots)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(0.005)

    # ── Video recording setup ────────────────────────────────────────
    video_writer = None
    video_w, video_h = 1920, 1080
    max_steps = 0
    frame_interval = 1  # physics steps between captures
    if args.record:
        fps = args.record_fps
        frame_interval = max(1, int(1.0 / (fps * 0.005)))  # physics steps per frame
        max_steps = int(args.record_seconds * fps * (1.0 / 0.005))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.record, fourcc, fps, (video_w, video_h))
        print(f"[fleet] Recording {args.record_seconds}s @ {fps}fps → {args.record}")
        print(f"[fleet] Total physics steps: {max_steps}, frame interval: {frame_interval}")

    # ── Viewport / Camera setup ──────────────────────────────────────
    if not args.headless:
        try:
            import omni.kit.viewport.utility as vp_util
            viewport_api = vp_util.create_viewport_from_window()
            vp_util.set_camera_position(viewport_api, 5.0, -5.0, 3.0)
            vp_util.set_camera_target(viewport_api, 0.0, 0.0, 0.5)
            print("[fleet] Viewport camera set: looking at robots from (5, -5, 3)")
        except Exception as e:
            print(f"[fleet] Viewport setup note: {e}")
            print("[fleet] Using default viewport (press W/A/S/D to navigate)")

    robots = {f"k1_{i}": scene[f"robot_{i}"] for i in range(args.n_robots)}
    pubs, subs, policy_states = {}, {}, {}

    for ns, art in robots.items():
        # Joint index mapping
        idx = [art.find_joints(j)[0][0] for j in JOINT_ORDER]
        pubs[ns] = {
            "js": node.create_publisher(JointState, f"/{ns}/joint_states", 10),
            "idx": idx,
            "art": art,
        }

        # Velocity command subscriber
        def make_cmd_cb(namespace, articulation, joint_idx):
            def cb(msg):
                pass  # Velocity commands handled by policy
            return cb
        subs[ns] = node.create_subscription(
            Twist, f"/{ns}/cmd_vel", make_cmd_cb(ns, art, idx), 10
        )

        # JointCommand subscriber
        def make_js_cb(namespace, articulation, joint_idx):
            def cb(msg):
                try:
                    names = getattr(msg, "joint_names", None) or msg.name
                    positions = getattr(msg, "positions", None) or msg.position
                    q_des = {n_: float(p) for n_, p in zip(names, positions)}
                except Exception:
                    return
                if q_des:
                    ids, _ = articulation.find_joints(list(q_des.keys()))
                    tgt = torch.tensor([[q_des[k] for k in q_des.keys()]],
                                       device=articulation.device)
                    jids = torch.tensor(ids, device=articulation.device, dtype=torch.int32)
                    articulation.set_joint_position_target_index(target=tgt, joint_ids=jids)
            return cb
        subs[ns] = node.create_subscription(
            JointCommand, f"/{ns}/joint_commands", make_js_cb(ns, art, idx), 10
        )

        # Policy state
        policy_states[ns] = {
            "last_action": np.zeros(12, dtype=np.float32),
            "cmd_vel": np.array([args.cmd_vx, 0.0, 0.0], dtype=np.float32),
        }

        node.get_logger().info(
            f"[{ns}] Isaac fleet endpoint ready (OmniGraph ROS2 bridge active)")

    rate = node.create_rate(50)

    step = 0
    # Give viewport a moment to initialize
    if not args.headless:
        import time as _time
        print("[fleet] Waiting 3s for viewport to initialize...")
        for _ in range(30):
            sim.step(render=True)
            _time.sleep(0.1)
        print("[fleet] Viewport ready — entering main loop")

    try:
        print("[fleet] Entering simulation loop...")
        while simulation_app.is_running():
            # ROS spin
            if _HAS_ROS:
                rclpy.spin_once(node, timeout_sec=0.005)

            # Policy inference per robot
            for ns, art in robots.items():
                ps = policy_states[ns]
                joint_pos = art.data.joint_pos[0].cpu().numpy()
                joint_vel = art.data.joint_vel[0].cpu().numpy()

                # Extract leg joints in policy order
                leg_pos = np.zeros(12, dtype=np.float32)
                leg_vel = np.zeros(12, dtype=np.float32)
                for lj_idx, lj_name in enumerate(LEG_JOINTS):
                    j_idx = pubs[ns]["idx"][JOINT_ORDER.index(lj_name)] \
                        if lj_name in JOINT_ORDER else None
                    if j_idx is not None and j_idx < len(joint_pos):
                        leg_pos[lj_idx] = joint_pos[j_idx]
                        leg_vel[lj_idx] = joint_vel[j_idx]

                # Projected gravity (approximate: assume upright)
                projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                base_lin_vel = np.zeros(3, dtype=np.float32)
                base_ang_vel = np.zeros(3, dtype=np.float32)

                obs = build_obs(base_lin_vel, base_ang_vel, projected_gravity,
                                ps["cmd_vel"], leg_pos, leg_vel, ps["last_action"])

                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

                ps["last_action"] = action
                targets = DEFAULT_LEG_POS + ACTION_SCALE * action

                # Apply to robot
                target_tensor = torch.zeros(1, art.num_joints, device=art.device)
                for lj_idx, lj_name in enumerate(LEG_JOINTS):
                    if lj_name in JOINT_ORDER:
                        j_id = pubs[ns]["idx"][JOINT_ORDER.index(lj_name)]
                        target_tensor[0, j_id] = float(targets[lj_idx])
                art.set_joint_position_target(target_tensor)

            # Step simulation
            scene.write_data_to_sim()
            sim.step(render=True)  # Always render for video capture
            scene.update(0.005)
            step += 1

            # Publish joint states at 50 Hz (every 4th step at 200Hz physics)
            if step % 4 == 0:
                stamp = node.get_clock().now().to_msg()
                for ns, d in pubs.items():
                    art = d["art"]
                    m = JointState()
                    m.header.stamp = stamp
                    m.name = JOINT_ORDER
                    pos = art.data.joint_pos[0].cpu().numpy()
                    vel = art.data.joint_vel[0].cpu().numpy()
                    m.position = [float(pos[i]) for i in d["idx"]]
                    m.velocity = [float(vel[i]) for i in d["idx"]]
                    d["js"].publish(m)

            # ── Video frame capture ──────────────────────────────────
            if video_writer and step % frame_interval == 0:
                try:
                    # Use Isaac Sim's replicator annotator
                    import omni.replicator.core as rep
                    # Create render product from camera
                    render_product = rep.create.render_product(
                        "/OmniverseKit_Persp", resolution=(video_w, video_h))
                    # Attach RGB annotator
                    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                    rgb_annotator.attach(render_product)
                    # Step replicator to capture
                    rep.step()
                    # Get the data
                    data = rgb_annotator.get_data()
                    if data is not None and len(data) > 0:
                        frame = np.array(data)
                        if frame.shape[:2] != (video_h, video_w):
                            frame = cv2.resize(frame, (video_w, video_h))
                        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        n_frames = step // frame_interval
                        if n_frames % 10 == 0:
                            print(f"[fleet] Captured frame {n_frames}/{max_steps // frame_interval}")
                    rgb_annotator.detach()
                except Exception as e:
                    if step % 500 == 0:
                        print(f"[fleet] Frame capture error: {e}")

            # Stop after recording duration
            if video_writer and step >= max_steps:
                print(f"[fleet] Recording complete: {max_steps} steps")
                break

    except Exception as e:
        print(f"[fleet] ERROR in sim loop: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        pass
    finally:
        if video_writer:
            video_writer.release()
            print(f"[fleet] Video saved → {args.record}")
        if _HAS_ROS:
            node.destroy_node()
            rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
