"""Small tiled debug-video helper for K1 play/eval runs.

Provides three/four fixed USD cameras (overview, top-down, follow, side),
labelled panels, a status header, and transparent task markers.  It follows the
same cheap pattern as the PE debug recorder: render only while a clip is being
captured, and pump Kit with physics disabled.
"""
from __future__ import annotations

import os

import numpy as np


def pump_app(subframes: int = 1) -> None:
    """Update Kit ``subframes`` times without advancing physics.

    Extra updates let the real-time path tracer's temporal denoiser accumulate
    on the current pose, which is what removes the grain from the panels.
    """
    import carb
    import omni.kit.app

    settings = carb.settings.get_settings()
    previous = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    for _ in range(max(1, int(subframes))):
        omni.kit.app.get_app().update()
    settings.set_bool("/app/player/playSimulations", bool(previous))


def set_clean_render(spp: int = 8) -> None:
    """Raise samples per pixel and keep the denoiser on for clear debug videos."""
    import carb

    settings = carb.settings.get_settings()
    settings.set_int("/rtx/rtpt/spp", int(spp))
    settings.set_bool("/rtx/newDenoiser/enabled", True)
    settings.set_bool("/rtx/rtpt/fireflyFilter/enabled", True)


class PanelCameras:
    """Replicator RGB cameras posed from world-space eye/target pairs."""

    def __init__(self, names=("overview", "top_down", "follow", "side"), size=(1920, 1080), focal_length=16.0):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._rep = rep
        self._stage = omni.usd.get_context().get_stage()
        self._ops = {}
        self._annotators = {}
        self._render_products = []
        self.names = tuple(names)
        for index, name in enumerate(self.names):
            path = f"/Visuals/K1DebugCams/{name}"
            camera = UsdGeom.Camera.Define(self._stage, path)
            camera.CreateFocalLengthAttr(focal_length)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
            xform = UsdGeom.Xformable(camera)
            xform.ClearXformOpOrder()
            self._ops[name] = xform.AddTransformOp()
            product = rep.create.render_product(path, size)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([product])
            self._render_products.append(product)
            self._annotators[name] = annotator

    def set_views(self, views) -> None:
        """Pose cameras from ``[(name, eye, target), ...]``."""
        Gf = self._Gf
        for name, eye, target in views:
            if name not in self._ops:
                continue
            matrix = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*[float(v) for v in eye]),
                Gf.Vec3d(*[float(v) for v in target]),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            self._ops[name].Set(matrix.GetInverse())

    def read(self, name):
        raw = self._annotators[name].get_data()
        if isinstance(raw, dict):
            raw = raw.get("data")
        if raw is None:
            return None
        raw = np.asarray(raw, dtype=np.uint8)
        return raw if raw.ndim == 3 else None

    def close(self) -> None:
        for product in self._render_products:
            try:
                self._rep.remove_render_product(product)
            except Exception:
                pass


class TiledPanelRecorder:
    """Write labelled panels and a status header to one MP4."""

    # quality=5 maps to roughly crf 25 in imageio's ffmpeg writer. The previous 8
    # (crf ~10) produced a 55 Mbit/s stream -- RTX path-tracing noise is close to
    # incompressible -- and writing that onto the bind-mounted logs filesystem
    # corrupted the mdat: the moov index still advertised 700 frames while only 39
    # decoded, which ffmpeg reports as "Invalid NAL unit size".
    QUALITY = 5

    def __init__(self, path, names, size=(1920, 1080), cols=2, fps=25):
        import imageio.v2 as imageio
        from PIL import ImageFont

        self.path = path
        self.names = tuple(names)
        self.size = tuple(size)
        self.cols = cols
        self.frames_added = 0
        # Encode to a container-local path, then move into place on close(). The logs
        # dir is a bind mount from the host; streaming tens of Mbit/s at it was the
        # suspected source of the bitstream damage.
        import tempfile

        self.tmp_path = os.path.join(tempfile.gettempdir(), f"_panel_{os.getpid()}.mp4")
        self.writer = imageio.get_writer(
            self.tmp_path, fps=fps, codec="libx264", quality=self.QUALITY, macro_block_size=1
        )
        # label/header text scales with the panel height (16 px at 270 p)
        self.font_px = max(16, self.size[1] * 16 // 270)
        self.header_px = max(16, self.size[1] * 10 // 270)
        try:
            self.header_font = ImageFont.load_default(size=self.header_px)
        except TypeError:
            self.header_font = ImageFont.load_default()
        try:
            self.font = ImageFont.load_default(size=self.font_px)
        except TypeError:
            self.font = ImageFont.load_default()

    def add(self, images, status: str) -> None:
        from PIL import Image, ImageDraw

        cells = []
        for name in self.names:
            image = images.get(name)
            if image is None:
                image = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            else:
                image = np.asarray(image[:, :, :3], dtype=np.uint8)
            image = Image.fromarray(image).resize(self.size)
            draw = ImageDraw.Draw(image)
            f = self.font_px
            draw.rectangle((0, 0, int(0.75 * f * len(name)) + f, int(1.5 * f)), fill=(0, 0, 0))
            draw.text((f // 3, f // 4), name, fill=(255, 255, 255), font=self.font)
            cells.append(np.asarray(image))

        while len(cells) % self.cols:
            cells.append(np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8))
        rows = [np.concatenate(cells[i:i + self.cols], axis=1) for i in range(0, len(cells), self.cols)]
        grid = np.concatenate(rows, axis=0)
        # wrap the status over as many lines as it needs (glyphs are ~0.55 em wide)
        import textwrap

        f = self.header_px
        lines = textwrap.wrap(status, width=max(40, int(grid.shape[1] / (0.55 * f)))) or [""]
        height = (len(lines) * int(1.25 * f) + f) // 2 * 2   # even, for yuv420p
        header = Image.new("RGB", (grid.shape[1], height), (18, 18, 18))
        draw = ImageDraw.Draw(header)
        for i, line in enumerate(lines):
            draw.text((f // 2, f // 2 + i * int(1.25 * f)), line, fill=(240, 240, 240), font=self.header_font)
        self.writer.append_data(np.concatenate([np.asarray(header), grid], axis=0))
        self.frames_added += 1

    def close(self) -> None:
        """Close, verify, publish, verify again.

        The pre-move check is a bisect, not ceremony. Two recordings came out corrupt
        ("Invalid NAL unit size (117004 > 116125)", 505 decode errors) with the moov
        atom intact and the correct duration, so file existence and ffprobe duration are
        both worthless as evidence. Checking at the container-local tmp path and again
        after the move to the bind mount isolates which stage does the damage:
        the encoder, or the copy onto /workspace/mounts.
        """
        import shutil

        self.writer.close()

        pre = self._count_decodable(self.tmp_path)
        print(f"[PANEL_VERIFY] pre-move  frames_added={self.frames_added} "
              f"decodable={pre} bytes={os.path.getsize(self.tmp_path)}")

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        shutil.move(self.tmp_path, self.path)
        post = self._count_decodable(self.path)

        # -1 means the check itself could not run (no ffmpeg, timeout, probe failure).
        # That is NOT the same as zero decodable frames and must not be reported as
        # corruption, which is how the first fixed run printed CORRUPT for a file that
        # had merely gone unchecked.
        if pre < 0 or post < 0:
            verdict = "UNCHECKED"
        elif post == self.frames_added:
            verdict = "OK"
        else:
            verdict = "CORRUPT"
        blame = ""
        if verdict == "CORRUPT" and pre == self.frames_added:
            blame = " (damaged by the move onto the bind mount)"
        elif verdict == "CORRUPT":
            blame = " (already corrupt in the encoder's own output)"
        print(
            f"[PANEL_VERIFY] post-move frames_added={self.frames_added} "
            f"decodable={post} {verdict}{blame} bytes={os.path.getsize(self.path)} {self.path}"
        )

    def _count_decodable(self, path) -> int:
        """Count frames a decoder can actually produce. -1 = could not check."""
        import shutil as _shutil
        import subprocess

        ffmpeg = _shutil.which("ffmpeg")
        ffprobe = _shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            # imageio drives ffmpeg through the imageio-ffmpeg wheel, which ships its
            # own binary; the system one is often absent inside the container.
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:  # noqa: BLE001
                print(f"[PANEL_VERIFY] no ffmpeg available: {type(exc).__name__}: {exc}")
                return -1
            ffprobe = ffmpeg  # ffprobe is not shipped; fall back to ffmpeg's own count

        try:
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-i", path, "-f", "null", "-"],
                capture_output=True, text=True, timeout=900,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[PANEL_VERIFY] decode check failed: {type(exc).__name__}: {exc}")
            return -1

        if proc.stderr.strip():
            errs = [l for l in proc.stderr.splitlines() if l.strip()]
            print(f"[PANEL_VERIFY] ffmpeg emitted {len(errs)} error line(s), first: {errs[0][:90]}")
            return 0

        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=900,
            )
            return int(probe.stdout.strip().splitlines()[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[PANEL_VERIFY] frame count failed: {type(exc).__name__}: {exc}")
            return -1


class K1DebugMarkers:
    """Transparent robot/target/boundary markers for the debug panels."""

    def __init__(self):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        self._Gf = Gf
        self._Vt = Vt
        self._stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(self._stage, "/Visuals/K1DebugMarkers")
        self.robot = self._sphere("robot", Gf.Vec3f(0.1, 0.8, 1.0))
        self.target = self._sphere("target", Gf.Vec3f(1.0, 0.8, 0.1))
        self.estimate = self._sphere("estimate", Gf.Vec3f(1.0, 0.1, 0.8))
        # A solid translucent cube would sit between the cameras and the robot,
        # so the arena boundary is drawn as a flat square outline instead.
        boundary = UsdGeom.BasisCurves.Define(self._stage, "/Visuals/K1DebugMarkers/boundary")
        half = 5.0
        points = [Gf.Vec3f(x, y, 0.0) for x, y in
                  ((-half, -half), (half, -half), (half, half), (-half, half))]
        boundary.CreatePointsAttr(Vt.Vec3fArray(points))
        boundary.CreateWidthsAttr([0.03])
        boundary.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.5, 1.0)])
        boundary.CreateDisplayOpacityAttr([0.7])
        self.boundary = boundary
        # Goal: wireframe of the push box's goal pose (12 edges) or, for carry,
        # a ring at the success radius. Points are rewritten each update.
        goal = UsdGeom.BasisCurves.Define(self._stage, "/Visuals/K1DebugMarkers/goal")
        goal.CreateTypeAttr(UsdGeom.Tokens.linear)
        goal.CreateWidthsAttr([0.025])
        goal.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        goal.CreateDisplayColorAttr([Gf.Vec3f(0.1, 1.0, 0.2)])
        goal.CreateDisplayOpacityAttr([0.8])
        goal.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        self.goal = goal
        self.goal_centre = self._sphere("goal_centre", Gf.Vec3f(0.1, 1.0, 0.2))
        self.goal_centre.GetRadiusAttr().Set(0.05)
        UsdGeom.Imageable(self.goal_centre).MakeInvisible()
        self._xforms = {
            "robot": UsdGeom.Xformable(self.robot),
            "target": UsdGeom.Xformable(self.target),
            "estimate": UsdGeom.Xformable(self.estimate),
            "boundary": UsdGeom.Xformable(boundary),
            "goal_centre": UsdGeom.Xformable(self.goal_centre),
        }
        del root

    def _sphere(self, name, color):
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(self._stage, f"/Visuals/K1DebugMarkers/{name}")
        sphere.CreateRadiusAttr(0.11)
        sphere.CreateDisplayColorAttr([color])
        sphere.CreateDisplayOpacityAttr([0.45])
        return sphere

    def mark(self, name, position, color=(1.0, 0.5, 0.05), radius=0.06) -> None:
        """Place a named translucent sphere, creating it on first use (extra robots,
        per-robot wrist targets)."""
        from pxr import Gf, UsdGeom

        if name not in self._xforms:
            sphere = self._sphere(name, Gf.Vec3f(*color))
            sphere.GetRadiusAttr().Set(radius)
            self._xforms[name] = UsdGeom.Xformable(sphere)
        self._translate(name, position)

    def _translate(self, name, position) -> None:
        xform = self._xforms[name]
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(self._Gf.Vec3d(*[float(v) for v in position]))

    def _show_goal(self, points, counts, centre) -> None:
        from pxr import UsdGeom

        Gf, Vt = self._Gf, self._Vt
        self.goal.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in points]))
        self.goal.GetCurveVertexCountsAttr().Set(Vt.IntArray(counts))
        UsdGeom.Imageable(self.goal).MakeVisible()
        UsdGeom.Imageable(self.goal_centre).MakeVisible()
        self._translate("goal_centre", centre)

    def _update_goal(self, base_env):
        """Draw the task goal if the env has one; return a header string or ''."""
        push = getattr(base_env, "push_state", None)
        if push is not None:
            pos = push.goal_pos[0].detach().cpu().numpy()
            x, y, z, w = (float(v) for v in push.goal_quat[0].detach().cpu().numpy())  # xyzw
            half = push.half_extents[0].detach().cpu().numpy()
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners = pos + (signs * half) @ rot.T
            edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                     if int(np.abs(signs[a] - signs[b]).sum()) == 2]
            self._show_goal([corners[i] for e in edges for i in e], [2] * len(edges), pos)
            box_xy = None
            try:
                box_pos = base_env.scene["box"].data.root_pos_w
                box_xy = getattr(box_pos, "torch", box_pos)[0, :2].detach().cpu().numpy()
            except Exception:
                pass
            dist = "" if box_xy is None else f" box->goal={float(np.linalg.norm(pos[:2] - box_xy)):.2f}m"
            return f"goal ({pos[0]:+.2f},{pos[1]:+.2f}){dist}"
        try:
            term = base_env.command_manager.get_term("base_velocity")
        except Exception:
            term = None
        if term is not None and hasattr(term, "goal_pos_w"):
            pos = term.goal_pos_w[0].detach().cpu().numpy()
            radius = float(getattr(term.cfg, "success_radius", 0.25))
            angles = np.linspace(0.0, 2.0 * np.pi, 33)
            ring = [pos + np.array([radius * np.cos(a), radius * np.sin(a), 0.02]) for a in angles]
            self._show_goal(ring, [len(ring)], pos + np.array([0.0, 0.0, 0.05]))
            return f"goal ({pos[0]:+.2f},{pos[1]:+.2f}) dist={float(term.goal_dist[0]):.2f}m"
        return ""

    def update(self, base_env, robot, task: str) -> dict:
        """Update markers and return status/knowledge fields for the header."""
        scene = base_env.scene
        root = robot.data.root_pos_w[0].detach().cpu().numpy()
        origin = scene.env_origins[0].detach().cpu().numpy()
        self._translate("robot", root + np.array([0.0, 0.0, 0.18]))
        self._translate("boundary", origin + np.array([0.0, 0.0, -0.03]))

        target = root + np.array([0.8, 0.0, 0.15])
        knowledge = "command target"
        ball = None
        try:
            ball = scene["ball"]
        except Exception:
            pass
        if ball is not None:
            target = ball.data.root_pos_w[0].detach().cpu().numpy()
            knowledge = "ball GT (reward only)"
        else:
            try:
                cmd = base_env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy()
                norm = float(np.linalg.norm(cmd[:2]))
                if norm > 0.05:
                    target = root + np.array([cmd[0], cmd[1], 0.0]) / norm * min(1.5, max(0.5, norm))
                    knowledge = f"command ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})"
            except Exception:
                pass
        self._translate("target", target)
        self._translate("estimate", target + np.array([0.0, 0.0, 0.12]))
        goal = self._update_goal(base_env)
        if goal:
            knowledge = f"{knowledge} | {goal}"
        return {"target": target, "knowledge": knowledge, "task": task}
