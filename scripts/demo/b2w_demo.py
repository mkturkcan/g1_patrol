"""Record the B2-W demo: the B2-W navigating the deployment lab, indoor and New York scenarios among animated people.

As ``nyc_demo.py`` (the same dressing, people, cameras and per-clip checks), for the B2-W tasks: one robot per chosen
scenario tile runs the navigation policy; the kinematic box of the simulation is drawn as Unitree's B2-W
(``b2w_look.py``: legs posed for the episode's base height, wheels turning with the motion); indoor tiles are roofless
rooms filmed from above their walls. With ``--shield`` the policy's commands pass through the safety shield as on the
robot (``scripts/b2w/shielded.py``). Every episode is saved as its own clip, named after its scenario and how it ended.
``--camera d435`` films from the robot's front RealSense (its mount, the color stream's field of view) and ``--camera
robot`` from a forward-looking camera on the front of the body; both also write the depth (``*_depth.mp4``: inverse
depth, 0.5 m white to 20 m black) as a control video for video-to-video models.

Usage:
    python scripts/demo/b2w_demo.py --task G1Patrol-B2W-Navigate-Lab-Fused --checkpoint <model.pt> \\
        --description <path to unitree_ros/robots/b2w_description> --scenarios makerlab lab sidewalk --shield
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import math
import os
import sys

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    add_common_play_args,
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "scripts", "b2w"))
from nyc_demo import TITLES as STREET_TITLES  # noqa: E402
from nyc_demo import FollowCamera, Overlay, _camera_bounds, robot_in_structure  # noqa: E402

TITLES = {
    **STREET_TITLES,
    "makerlab": "The lab",
    "civil_lab": "Civil engineering lab",
    "lab": "Laboratory",
    "corridor": "Office corridor",
    "doorway": "Doorways",
    "office": "Open-plan office",
    "warehouse": "Warehouse aisles",
    "lobby": "Building lobby",
}
INDOOR = ("makerlab", "civil_lab", "lab", "corridor", "doorway", "office", "warehouse", "lobby")
HALLS = ("makerlab", "civil_lab")
"""Indoor tiles that are sections of a hall continuing beyond the tile's ends."""
DEPTH_RANGE = (0.5, 20.0)
"""Depth (m) drawn white and black in the depth videos (inverse depth in between)."""
OUTCOMES = ("goal_reached", "pedestrian_collision", "obstacle_collision", "roadway", "time_out")
DESCRIPTION = os.path.join(os.path.dirname(PROJECT), "assets_dl", "repos_ref", "unitree_ros", "robots", "b2w_description")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="RL agent config entry point.")
    parser.add_argument("--scenarios", nargs="+", default=["makerlab", "lab", "sidewalk"], choices=list(TITLES))
    parser.add_argument("--level", type=int, default=6, help="Difficulty level (terrain row) of the filmed tiles.")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Per-scenario levels (overrides --level).")
    parser.add_argument("--camera", default="chase", choices=["chase", "drone", "side", "d435", "robot"])
    parser.add_argument("--no_overlay", action="store_true", help="Record clean frames without titles or labels.")
    parser.add_argument("--variant", type=int, default=0, help="Which of the scenario's terrain columns to use.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds to record.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--render_interval", type=int, default=1, help="Physics steps (50 Hz) per frame.")
    parser.add_argument("--out", default=None, help="Output directory (default: <checkpoint dir>/demo).")
    parser.add_argument("--label", default="Unitree B2-W · simulation · RL navigation policy")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--snapshot_every", type=float, default=0.0, help="Also save a PNG still every N seconds.")
    parser.add_argument("--description", default=DESCRIPTION, help="Unitree's b2w_description (URDF and meshes).")
    parser.add_argument("--shield", action="store_true", help="The safety shield in the loop, as on the robot.")
    parser.add_argument("--speed_limit", type=float, default=None, help="Speed cap (m/s; default: drawn per episode).")
    parser.add_argument("--room_light", type=float, default=600.0, help="Intensity of the light over indoor tiles.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    args.video = True  # turns on headless rendering; frames are captured by this script, not the env's recorder
    enable_cameras_for_video(args)
    set_hydra_args(remaining)
    return args


class B2WCamera(FollowCamera):
    """The follow camera, framed for the B2-W (0.76 m tall); over indoor tiles it looks down from above the walls."""

    CHASE = {  # (back, lateral, eye height above the ground)
        "sidewalk": (2.6, 1.0, 1.6),
        "scaffold": (2.6, 0.3, 1.4),
        "crosswalk": (2.7, 1.1, 2.0),
        "construction": (2.8, 1.3, 2.6),
        "bottleneck": (2.8, 1.3, 2.6),
        "plaza": (3.0, 1.4, 2.6),
    }

    def __init__(self, stage, path: str, kind: str, width: int, height: int, mode: str = "chase"):
        super().__init__(stage, path, kind, width, height, mode)
        if kind in INDOOR and mode == "chase":  # a longer lens from higher up: tall things near it splay less
            self.focal = 20.0
            self.camera.GetFocalLengthAttr().Set(self.focal)

    def update(self, robot_pos, robot_yaw, tile_center, dt, bounds=None, ground=0.0):
        self.t += dt
        heading = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
        if self.mode != "chase":
            return self._update_cinematic(robot_pos, heading, tile_center, dt, bounds)
        if self.kind in INDOOR:  # above the walls (2.7 m), behind the robot, looking down at it
            back, lateral, up = 3.6, 0.5, 5.2
            side = np.array([-heading[1], heading[0]])
            side = side if side @ (tile_center[:2] - robot_pos[:2]) >= 0 else -side
        else:
            back, lateral, up = self.CHASE[self.kind]
            side = np.array([0.0, -1.0])
            if self.kind == "crosswalk":
                side = np.array([1.0, 0.0]) if robot_pos[0] < tile_center[0] else np.array([-1.0, 0.0])
            elif self.kind == "plaza":
                side = np.array([-heading[1], heading[0]])
                side = side if side @ (tile_center[:2] - robot_pos[:2]) >= 0 else -side
        eye = np.array([*(robot_pos[:2] - back * heading + lateral * side), ground + up])
        if bounds is not None:
            x0, x1, y0, y1, z1 = bounds
            clamped = np.array([np.clip(eye[0], x0, x1), np.clip(eye[1], y0, y1)])
            eye[2] = min(eye[2] + 0.5 * np.linalg.norm(eye[:2] - clamped), z1)
            eye[:2] = clamped
        target = np.array([*(robot_pos[:2] + (0.9 if self.kind in INDOOR else 1.6) * heading), ground + 0.35])
        if self.eye is None:
            self.eye, self.target = eye, target
        self.eye = self.eye + (eye - self.eye) * min(1.0, dt / 0.7)
        self.target = self.target + (target - self.target) * min(1.0, dt / 0.35)
        self._look_at(self.eye, self.target)


class RobotCamera(FollowCamera):
    """A camera on the robot: ``d435`` at the front RealSense's mount (45 deg down) with the color stream's field of
    view (69 x 42 deg at 16:9), ``robot`` looking ahead from the front of the body (12 deg down, same lens). Also
    renders the depth (distance to the image plane)."""

    def __init__(self, stage, path: str, kind: str, width: int, height: int, mode: str):
        import omni.replicator.core as rep

        super().__init__(stage, path, kind, width, height, "chase")
        self.mode = mode
        self.focal = self.aperture[0] / (2 * math.tan(math.radians(69.0) / 2))
        self.camera.GetFocalLengthAttr().Set(self.focal)
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth.attach(self.render_product)

    def update(self, robot_pos, robot_quat, dt):
        """``robot_quat`` (w, x, y, z) of the base."""
        from g1_patrol.b2w import DEPTH_CAMERAS, rpy_matrix

        self.t += dt
        w, x, y, z = robot_quat
        base = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])  # fmt: skip
        if self.mode == "d435":
            mount, rpy = DEPTH_CAMERAS["front"]
            axis = np.asarray(rpy_matrix(rpy))[:, 2]  # the optical axis in base_link
        else:
            mount, pitch = (0.58, 0.0, 0.08), math.radians(12.0)
            axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        self.eye = np.asarray(robot_pos) + base @ np.asarray(mount)
        self.target = self.eye + base @ axis
        self._look_at(self.eye, self.target)

    def depth_frame(self) -> np.ndarray | None:
        """Inverse depth as 8-bit gray (white at :data:`DEPTH_RANGE` [0], black at [1] and beyond)."""
        data = self.depth.get_data()
        if data is None or len(data) == 0:
            return None
        d = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=DEPTH_RANGE[1], posinf=DEPTH_RANGE[1])
        near, far = DEPTH_RANGE
        v = (1.0 / np.clip(d, near, far) - 1.0 / far) / (1.0 / near - 1.0 / far)
        return np.repeat((255 * v).astype(np.uint8)[..., None], 3, axis=-1)


def _indoor_bounds(center) -> tuple:
    """Where an indoor tile's camera may go: over the room (it is roofless), up to 6 m."""
    from g1_patrol.tasks.navigation.nyc.indoor import IN_L, IN_W

    cx, cy = center
    return cx - IN_L + 0.4, cx + IN_L - 0.4, cy - IN_W + 0.4, cy + IN_W - 0.4, 6.0


def footprint_points(pos: np.ndarray, yaw: float, ground: float) -> np.ndarray:
    """Points on the robot's sides (corners and edge midpoints, low and at the body), world frame, for the clip check
    of the robot inside a structure."""
    from g1_patrol.b2w import FOOTPRINT

    hx, hy = FOOTPRINT[0] - 0.06, FOOTPRINT[1] - 0.03  # (the footprint has a margin around the body)
    local = np.array([(sx * hx, sy * hy) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if sx or sy])
    c, s = math.cos(yaw), math.sin(yaw)
    xy = pos[:2] + local @ np.array([[c, s], [-s, c]])
    return np.concatenate([np.c_[xy, np.full(len(xy), ground + h)] for h in (0.12, pos[2] - ground)])


def main(argv):
    args = parse_args(argv)
    installed_version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=True)
    env_cfg.scene.num_envs = len(args.scenarios)
    env_cfg.curriculum.levels = None
    env_cfg.sim.render_interval = args.render_interval
    env_cfg.commands.nyc.debug_vis = False
    if args.speed_limit is not None:
        env_cfg.actions.base.speed_limit_range = (args.speed_limit, args.speed_limit)
    pre_launch_video_config(env_cfg, args)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        import imageio.v2 as imageio
        import omni.usd
        from pxr import Usd, UsdGeom

        import b2w_look
        import nyc_look
        from people import AnimatedCrowd

        from g1_patrol.tasks.navigation.nyc.kinds import ALL_KINDS
        from g1_patrol.tasks.navigation.nyc.terrain import NycTerrainGenerator

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        apply_env_overrides(args, env_cfg)
        env_cfg.scene.num_envs = len(args.scenarios)
        env_cfg.seed = agent_cfg.seed + args.seed_offset
        checkpoint = os.path.abspath(args.checkpoint)
        out_dir = args.out or os.path.join(os.path.dirname(checkpoint), "demo")
        os.makedirs(out_dir, exist_ok=True)

        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        scenario = base.command_manager.get_term("nyc")
        action = base.action_manager.get_term("base")
        terrain = base.scene.terrain
        stage = omni.usd.get_context().get_stage()
        shield = None
        if args.shield:
            from shielded import ShieldInLoop

            shield = ShieldInLoop(base)

        # one robot per scenario, on that scenario's tile at the requested difficulty
        kinds = scenario.layout.kind
        levels = args.levels or [args.level] * len(args.scenarios)
        assert len(levels) == len(args.scenarios), "--levels needs one level per scenario"
        tiles = []
        for e, (name, level) in enumerate(zip(args.scenarios, levels)):
            cols = torch.nonzero(kinds[level] == ALL_KINDS.index(name)).flatten()
            assert len(cols), f"no {name} tile at level {level} in {args.task}"
            col = int(cols[min(args.variant, len(cols) - 1)])
            terrain.terrain_levels[e], terrain.terrain_types[e] = level, col
            tiles.append((level, col))
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

        # looks: the filmed tiles and copies of them along the street; people; the robots; cameras
        generator = NycTerrainGenerator.latest
        b2w_look.register_indoor_materials()
        placements = {}
        for (r, c), name in zip(tiles, args.scenarios):
            if name not in INDOOR or name in HALLS:  # streets, and the lab halls, continue beyond the tile's ends
                placements.update({(r + k, c): (r, c) for k in (-2, -1, 1, 2)})
        placements.update({t: t for t in tiles})
        nyc_look.dress(stage, generator, placements)
        for (r, c), source in placements.items():  # roofless rooms: light from above, as from a ceiling
            if args.scenarios[tiles.index(source)] in INDOOR:
                b2w_look.add_room_light(stage, f"/World/Demo/RoomLights/r{r}_c{c}", nyc_look.tile_center(generator, r, c),
                                        intensity=args.room_light)
        stairs = [[item[1:] for item in generator.layouts[t].decor if item[0] == "stairs"] for t in tiles]
        clips_dir = os.path.join(PROJECT, "src/g1_patrol/assets/data/people")
        filmed = list(range(len(args.scenarios)))
        crowd = AnimatedCrowd(scenario, filmed, clips_dir, stairs=stairs)
        models = [b2w_look.B2WModel(stage, f"/World/Demo/B2W/robot_{e}", args.description) for e in filmed]
        for e in filmed:  # the kinematic box the simulation moves: hidden (it still collides and is ray-cast)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")):
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
        on_robot = args.camera in ("d435", "robot")
        camera_class = RobotCamera if on_robot else B2WCamera
        cameras = [camera_class(stage, f"/World/Demo/Cameras/cam_{e}", name, args.width, args.height, args.camera)
                   for e, name in enumerate(args.scenarios)]  # fmt: skip
        centers0 = [nyc_look.tile_center(generator, *t) for t in tiles]
        camera_bounds = [
            _indoor_bounds(centers0[e]) if name in INDOOR
            else _camera_bounds(generator.layouts[tiles[e]], centers0[e], inside_shed=args.camera == "chase")
            for e, name in enumerate(args.scenarios)
        ]
        overlay = Overlay(args.width, args.height, args.label)
        frame_dt = base.physics_dt * args.render_interval
        fps = 1.0 / frame_dt

        clips = [None] * len(args.scenarios)
        people_events = [[0, 0, 0, 0] for _ in args.scenarios]
        touches = [collections.Counter() for _ in args.scenarios]
        prev_people = [None] * len(args.scenarios)
        clip_t = [0.0] * len(args.scenarios)
        clip_frames = [0] * len(args.scenarios)
        clip_count = [0] * len(args.scenarios)

        depth_clips = [None] * len(args.scenarios)

        def open_clip(e):
            clip_count[e] += 1
            path = os.path.join(out_dir, f"{args.scenarios[e]}_L{levels[e]}_{args.camera}_ep{clip_count[e]:02d}.mp4")
            clips[e] = (imageio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=8), path)
            if on_robot:
                depth_path = path.replace(".mp4", "_depth.mp4")
                writer = imageio.get_writer(depth_path, fps=fps, codec="libx264", quality=9, macro_block_size=8)
                depth_clips[e] = (writer, depth_path)
            clip_t[e], clip_frames[e] = 0.0, 0
            people_events[e], prev_people[e] = [0, 0, 0, 0], None
            touches[e].clear()

        def close_clip(e, outcome):
            writer, path = clips[e]
            writer.close()
            if on_robot:
                depth_clips[e][0].close()
            if clip_frames[e] == 0:
                print(f"[DEMO] {args.scenarios[e]} episode {clip_count[e]}: {outcome} after {clip_t[e]:.1f} s, no frames", flush=True)
                return
            final = path.replace(".mp4", f"_{outcome}.mp4")
            os.replace(path, final)
            if on_robot:
                os.replace(depth_clips[e][1], final.replace(".mp4", "_depth.mp4"))
            appear, vanish, backward, touching = people_events[e]
            print(f"[DEMO] {os.path.basename(final)} ({clip_t[e]:.1f} s, {clip_frames[e]} frames; people in view: {appear} appeared, "
                  f"{vanish} vanished, {backward} character-frames moving backwards; robot inside a structure in "
                  f"{touching} frames)", flush=True)
            if touching:
                print("[DEMO]   robot inside: " + ", ".join(f"{mat} ({n})" for mat, n in touches[e].most_common()), flush=True)

        import carb
        import omni.kit.app

        settings = carb.settings.get_settings()
        settings.set("/rtx/post/dlss/execMode", 2)  # DLSS "quality": render closer to native resolution

        def on_render(_):
            crowd.update(frame_dt)
            pos = base.scene["robot"].data.root_pos_w.torch.cpu().numpy()
            quat = base.scene["robot"].data.root_quat_w.torch.cpu().numpy()[:, [3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)
            yaw = scenario.robot_yaw().cpu().numpy()
            heights = action.base_height.cpu().numpy()
            velocity = action.velocity_b.cpu().numpy()
            centers = scenario.tile_center.cpu().numpy()
            for e, cam in enumerate(cameras):
                ground = float(pos[e][2] - heights[e])
                models[e].update(pos[e], quat[e], float(heights[e]), velocity[e], frame_dt)
                if on_robot:
                    cam.update(pos[e], quat[e], frame_dt)
                else:
                    cam.update(pos[e], float(yaw[e]), centers[e], frame_dt, camera_bounds[e], ground)
            play = settings.get("/app/player/playSimulations")
            settings.set("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()
            settings.set("/app/player/playSimulations", bool(play))
            origins = terrain.env_origins.cpu().numpy()
            for e, cam in enumerate(cameras):
                ground = float(pos[e][2] - heights[e])
                local = footprint_points(pos[e], float(yaw[e]), ground) - np.array([centers[e][0], centers[e][1], origins[e][2]])
                inside = robot_in_structure(generator.layouts[tiles[e]], local, radius=0.0) if clip_t[e] > 2 * frame_dt else None
                if inside is not None:
                    people_events[e][3] += 1
                    touches[e][inside[1]] += 1
                visible, points, backward = crowd.visible[e], crowd.points[e], crowd.backward[e]
                view = cam.in_view(points)
                if clip_t[e] > 2 * frame_dt and prev_people[e] is not None:
                    was_visible, was_in_view = prev_people[e]
                    people_events[e][0] += int((visible & ~was_visible & view).sum())
                    people_events[e][1] += int((~visible & was_visible & was_in_view).sum())
                    people_events[e][2] += int((backward & visible & view).sum())
                prev_people[e] = (visible, view)
                frame = cam.frame()
                if frame is not None and clip_t[e] > 2 * frame_dt:  # skip the first frames after a teleport
                    name = args.scenarios[e]
                    where = "Indoors" if name in INDOOR else "New York"
                    sub = "Digital twin of the deployment lab" if name == "makerlab" else f"{where} · difficulty {levels[e] + 1}/10"
                    shown = frame if args.no_overlay else overlay.draw(frame, TITLES[name], sub, clip_t[e])
                    clips[e][0].append_data(shown)
                    if on_robot:
                        depth = cam.depth_frame()
                        depth_clips[e][0].append_data(depth if depth is not None else np.zeros_like(frame))
                    clip_frames[e] += 1
                    if args.snapshot_every and clip_frames[e] % max(1, int(args.snapshot_every * fps)) == 1:
                        imageio.imwrite(os.path.join(out_dir, f"still_{name}_{args.camera}_ep{clip_count[e]:02d}_{clip_t[e]:05.1f}s.png"), frame)
                clip_t[e] += frame_dt

        obs, _ = env.reset()
        obs = env.get_observations()
        for e in range(len(args.scenarios)):
            open_clip(e)
        for _ in range(30):  # let textures and materials stream in before recording
            on_render(None)
        for e in range(len(args.scenarios)):
            clip_t[e] = 0.0
        base.sim.add_render_callback("b2w_demo", on_render, order=100)
        tm = base.termination_manager
        outcomes = [n for n in OUTCOMES if n in tm.active_terms]
        steps = int(args.seconds / base.step_dt)
        with torch.inference_mode():
            for step in range(steps):
                actions = policy(obs)
                if shield is not None:
                    actions, _ = shield(actions, obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
                if shield is not None:
                    shield.reset(dones)
                for e in torch.nonzero(dones[: len(args.scenarios)]).flatten().tolist():
                    outcome = next((n for n in outcomes if bool(tm.get_term(n)[e])), "done")
                    close_clip(e, outcome)
                    open_clip(e)
                if step % 100 == 0:
                    print(f"[DEMO] t = {step * base.step_dt:.0f} s", flush=True)
        base.sim.remove_render_callback("b2w_demo")
        for e in range(len(args.scenarios)):
            close_clip(e, "unfinished")


if __name__ == "__main__":
    main(sys.argv[1:])
