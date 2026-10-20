"""Record the New York demo: the G1 navigating NYC street scenarios among animated people, with city looks.

One robot per chosen scenario tile runs the navigation policy (teacher or student). The tiles are dressed with
textured streets, buildings and street furniture (``nyc_look.py``), the pedestrians are replaced by animated
characters (``people.py``) and a smoothed follow camera films each robot. Every episode is saved as its own clip,
named after its scenario and how it ended, so the best takes can be cut together afterwards.

The robot, the pedestrians' motion and everything the policy sees are exactly those of the evaluation environment;
only the rendering differs.

Usage:
    python scripts/demo/nyc_demo.py --checkpoint <model.pt> --scenarios scaffold crosswalk --level 7 --seconds 60
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
TITLES = {
    "sidewalk": "Busy sidewalk",
    "scaffold": "Sidewalk shed (scaffolding)",
    "crosswalk": "Crossing the street",
    "construction": "Construction detour",
    "bottleneck": "Doorway rush",
    "plaza": "Times Square plaza",
}
OUTCOMES = ("goal_reached", "pedestrian_collision", "obstacle_collision", "fall", "time_out")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="RL agent config entry point.")
    parser.add_argument("--scenarios", nargs="+", default=list(TITLES), choices=list(TITLES))
    parser.add_argument("--level", type=int, default=7, help="Difficulty level (terrain row) of the filmed tiles.")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Per-scenario levels (overrides --level).")
    parser.add_argument("--camera", default="chase", choices=["chase", "drone", "side"],
                        help="chase: behind the robot; drone: high, sweeping around it; side: tracking from the side.")
    parser.add_argument("--no_overlay", action="store_true", help="Record clean frames without titles or labels.")
    parser.add_argument("--variant", type=int, default=0, help="Which of the scenario's terrain columns to use.")
    parser.add_argument("--seconds", type=float, default=45.0, help="Simulated seconds to record.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--render_interval", type=int, default=6, help="Physics steps per frame (200 Hz physics).")
    parser.add_argument("--out", default=None, help="Output directory (default: <checkpoint dir>/demo).")
    parser.add_argument("--label", default="Unitree G1 · simulation · RL navigation policy")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--snapshot_every", type=float, default=0.0, help="Also save a PNG still every N seconds.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    args.video = True  # turns on headless rendering; frames are captured by this script, not the env's recorder
    enable_cameras_for_video(args)
    set_hydra_args(remaining)
    return args


class FollowCamera:
    """A smoothed chase camera for one robot, kept on the open side of the street and under scaffold decks."""

    def __init__(self, stage, path: str, kind: str, width: int, height: int, mode: str = "chase"):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.kind, self.mode, self.t = kind, mode, 0.0
        self.camera = UsdGeom.Camera.Define(stage, path)
        focal = {"chase": 16.0 if kind != "plaza" else 14.0, "drone": 18.0, "side": 24.0}[mode]
        self.camera.CreateFocalLengthAttr(focal)
        self.camera.CreateHorizontalApertureAttr(20.955)
        self.camera.CreateVerticalApertureAttr(20.955 * height / width)
        self.focal, self.aperture = focal, (20.955, 20.955 * height / width)
        self.camera.CreateClippingRangeAttr().Set((0.05, 500.0))
        xf = UsdGeom.Xformable(self.camera.GetPrim())
        self.transform = xf.AddTransformOp()
        self.render_product = rep.create.render_product(path, (width, height))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach(self.render_product)
        self.eye = None
        self.target = None

    def update(self, robot_pos: np.ndarray, robot_yaw: float, tile_center: np.ndarray, dt: float, bounds=None):
        self.t += dt
        heading = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
        if self.mode != "chase":
            return self._update_cinematic(robot_pos, heading, tile_center, dt, bounds)
        side = np.array([0.0, -1.0])  # along a sidewalk: stay on the street side, away from the buildings at +y
        # (back, lateral, height): higher where fences, barriers and partitions would block a low camera
        back, lateral, up = {
            "sidewalk": (3.2, 1.3, 2.0),
            "scaffold": (3.2, 0.3, 1.7),  # inside the shed, under its deck (3 m)
            "crosswalk": (3.2, 1.4, 2.7),
            "construction": (3.4, 1.6, 3.3),
            "bottleneck": (3.4, 1.6, 3.4),
            "plaza": (3.8, 1.8, 3.4),
        }[self.kind]
        if self.kind == "crosswalk":
            side = np.array([1.0, 0.0]) if robot_pos[0] < tile_center[0] else np.array([-1.0, 0.0])
        elif self.kind == "plaza":  # towards the middle of the plaza, away from the buildings
            to_center = tile_center[:2] - robot_pos[:2]
            side = np.array([-heading[1], heading[0]])
            side = side if side @ to_center >= 0 else -side
        eye = np.array([*(robot_pos[:2] - back * heading + lateral * side), robot_pos[2] + up - 0.75])
        if bounds is not None:  # stay out of walls and sheds: clamp, and look down more steeply instead
            x0, x1, y0, y1, z1 = bounds
            clamped = np.array([np.clip(eye[0], x0, x1), np.clip(eye[1], y0, y1)])
            eye[2] = min(eye[2] + 0.5 * np.linalg.norm(eye[:2] - clamped), z1)
            eye[:2] = clamped
        target = np.array([*(robot_pos[:2] + 1.8 * heading), robot_pos[2] - 0.1])
        if self.eye is None:
            self.eye, self.target = eye, target
        a = min(1.0, dt / 0.7)
        self.eye = self.eye + (eye - self.eye) * a
        self.target = self.target + (target - self.target) * min(1.0, dt / 0.35)
        self._look_at(self.eye, self.target)

    def _update_cinematic(self, robot_pos, heading, tile_center, dt, bounds):
        """Drone: high and slowly sweeping across the street side; side: a tracking shot from beside the route."""
        open_tile = self.kind in ("crosswalk", "plaza")
        if open_tile:  # look across the robot's path from the side nearer the tile center
            base = np.array([-heading[1], heading[0]])
            base = base if base @ (tile_center[:2] - robot_pos[:2]) >= 0 else -base
        else:  # the street side of a sidewalk
            base = np.array([0.0, -1.0])
        if self.mode == "drone":
            sweep = 0.75 * math.sin(0.12 * self.t)
            c, s = math.cos(sweep), math.sin(sweep)
            direction = np.array([c * base[0] - s * base[1], s * base[0] + c * base[1]])
            eye = np.array([*(robot_pos[:2] + 6.5 * direction - 2.0 * heading), robot_pos[2] + 5.5])
            target = np.array([*(robot_pos[:2] + 1.0 * heading), robot_pos[2] - 0.4])
        else:
            eye = np.array([*(robot_pos[:2] + 4.2 * base - 0.6 * heading), robot_pos[2] + 0.55])
            target = np.array([*(robot_pos[:2] + 0.9 * heading), robot_pos[2] - 0.05])
        if bounds is not None:
            x0, x1, y0, y1, z1 = bounds
            eye[0], eye[1] = np.clip(eye[0], x0, x1), np.clip(eye[1], y0, y1)
            eye[2] = min(eye[2], max(z1, 6.0) if self.mode == "drone" else eye[2])
        if self.eye is None:
            self.eye, self.target = eye, target
        self.eye = self.eye + (eye - self.eye) * min(1.0, dt / 1.2)
        self.target = self.target + (target - self.target) * min(1.0, dt / 0.5)
        self._look_at(self.eye, self.target)

    def _look_at(self, eye: np.ndarray, target: np.ndarray):
        from pxr import Gf

        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # USD cameras look down -z with +y up; rows are the camera axes in world coordinates
        m = Gf.Matrix4d(
            *right, 0.0,
            *up, 0.0,
            *(-forward), 0.0,
            *eye, 1.0,
        )  # fmt: skip
        self.transform.Set(m)

    def in_view(self, points: np.ndarray, max_distance: float = 60.0) -> np.ndarray:
        """Whether world points (m, 3) are in the image, in front of the camera and within ``max_distance``."""
        if self.eye is None:
            return np.zeros(len(points), dtype=bool)
        forward = (self.target - self.eye) / np.linalg.norm(self.target - self.eye)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        d = points - self.eye
        z, x, y = d @ forward, d @ right, d @ up
        tan_x, tan_y = self.aperture[0] / (2 * self.focal), self.aperture[1] / (2 * self.focal)
        return (z > 0.3) & (z < max_distance) & (np.abs(x) < z * tan_x) & (np.abs(y) < z * tan_y)

    def frame(self) -> np.ndarray | None:
        data = self.rgb.get_data()
        if data is None or len(data) == 0:
            return None
        return np.asarray(data)[..., :3]


class Overlay:
    """Scenario title (first seconds of a clip) and a small persistent label."""

    def __init__(self, width: int, height: int, label: str):
        from PIL import ImageFont

        fonts = "/usr/share/fonts/truetype/dejavu"
        self.title_font = ImageFont.truetype(f"{fonts}/DejaVuSans-Bold.ttf", int(height * 0.045))
        self.sub_font = ImageFont.truetype(f"{fonts}/DejaVuSans.ttf", int(height * 0.024))
        self.label = label
        self.width, self.height = width, height

    def draw(self, frame: np.ndarray, title: str, subtitle: str, t: float, caption: str | None = None) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        m = int(self.height * 0.04)
        alpha = int(255 * max(0.0, min(1.0, 3.5 - t)))  # fade the title out after ~3 s
        if alpha > 0:
            draw.text((m, m), title, font=self.title_font, fill=(255, 255, 255, alpha), stroke_width=2,
                      stroke_fill=(0, 0, 0, alpha // 2))
            draw.text((m, m + int(self.height * 0.06)), subtitle, font=self.sub_font, fill=(235, 235, 235, alpha),
                      stroke_width=1, stroke_fill=(0, 0, 0, alpha // 2))
        draw.text((m, self.height - m - int(self.height * 0.03)), self.label, font=self.sub_font,
                  fill=(255, 255, 255, 200), stroke_width=1, stroke_fill=(0, 0, 0, 120))
        if caption:  # what the robot is doing, bottom center
            w = draw.textlength(caption, font=self.title_font)
            x, y = (self.width - w) / 2, self.height - m - int(self.height * 0.11)
            pad = int(self.height * 0.012)
            draw.rounded_rectangle((x - 2 * pad, y - pad, x + w + 2 * pad, y + int(self.height * 0.05) + pad), radius=pad,
                                   fill=(0, 0, 0, 140))
            draw.text((x, y), caption, font=self.title_font, fill=(255, 255, 255, 235))
        return np.asarray(img)


def _camera_bounds(lay, center, inside_shed: bool = True) -> tuple:
    """Where a tile's camera may go (world x0, x1, y0, y1, z_max): in front of the building fronts, inside sheds, on the
    tile and over the street beside it."""
    from g1_patrol.tasks.navigation.nyc.tiles import HALF_L, HALF_W

    cx, cy = center
    y0, y1, z1 = -HALF_W - 6.0, HALF_W + 6.0, 6.0
    for y, facing in lay.facades:  # facing -1: a front at +y looking towards -y
        if facing < 0:
            y1 = min(y1, y - 0.35)
        else:
            y0 = max(y0, y + 0.35)
    posts = [c[1] for c, m in zip(lay.circles, lay.circle_mats) if m == "scaffold_steel"]
    if posts and inside_shed:  # inside the sidewalk shed, under its deck
        y0, z1 = min(posts) + 0.55, 2.5
    return cx - HALF_L - 8.0, cx + HALF_L + 8.0, cy + y0, cy + y1, z1


def robot_in_structure(lay, links: np.ndarray, radius: float = 0.04) -> tuple[int, str] | None:
    """The first robot link (tile-local positions, n x 3) inside one of the tile's structures (poles, posts, walls,
    barriers, fences), each extended by ``radius`` for the thickness of the limbs, as (link index, structure material);
    None if there is none."""
    x, y, z = links[:, 0], links[:, 1], links[:, 2]
    for (cx, cy, r, z0, z1), mat in zip(lay.circles, lay.circle_mats):
        base = lay.ground(cx, cy)
        hit = (np.hypot(x - cx, y - cy) < r + radius) & (z > base + z0 - radius) & (z < base + z1 + radius)
        if np.any(hit):
            return int(np.argmax(hit)), mat
    for (cx, cy, yaw, hx, hy, z0, z1), mat in zip(lay.boxes, lay.box_mats):
        base = lay.ground(cx, cy)
        c, s_ = math.cos(yaw), math.sin(yaw)
        lx, ly = c * (x - cx) + s_ * (y - cy), -s_ * (x - cx) + c * (y - cy)
        hit = (np.abs(lx) < hx + radius) & (np.abs(ly) < hy + radius) & (z > base + z0 - radius) & (z < base + z1 + radius)
        if np.any(hit):
            return int(np.argmax(hit)), mat
    return None


def main(argv):
    args = parse_args(argv)
    installed_version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=True)
    env_cfg.scene.num_envs = len(args.scenarios)
    env_cfg.curriculum.levels = None
    env_cfg.sim.render_interval = args.render_interval
    env_cfg.commands.nyc.debug_vis = False
    if getattr(env_cfg.commands.nyc, "social", False):
        env_cfg.commands.nyc.cyclist_ratio = (0.0, 0.0)  # no rider model: no cyclists in the films
    env_cfg.actions.walker.debug_vis = False  # velocity-command arrows
    pre_launch_video_config(env_cfg, args)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        import imageio.v2 as imageio
        import omni.usd

        sys.path.insert(0, HERE)
        import nyc_look
        from people import AnimatedCrowd

        from g1_patrol.tasks.navigation.nyc.terrain import NycTerrainGenerator
        from g1_patrol.tasks.navigation.nyc.tiles import KINDS

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
        terrain = base.scene.terrain
        stage = omni.usd.get_context().get_stage()

        # one robot per scenario, on that scenario's tile at the requested difficulty
        kinds = scenario.layout.kind
        levels = args.levels or [args.level] * len(args.scenarios)
        assert len(levels) == len(args.scenarios), "--levels needs one level per scenario"
        tiles = []
        for e, (name, level) in enumerate(zip(args.scenarios, levels)):
            cols = torch.nonzero(kinds[level] == KINDS.index(name)).flatten()
            col = int(cols[min(args.variant, len(cols) - 1)])
            terrain.terrain_levels[e], terrain.terrain_types[e] = level, col
            tiles.append((level, col))
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

        # looks: dress the filmed tiles and, along the street, copies of them (pedestrians enter and leave the scene
        # beyond the tile's ends, and walk past these copies); people; cameras
        generator = NycTerrainGenerator.latest
        placements = {(r + k, c): (r, c) for r, c in tiles for k in (-2, -1, 1, 2)}
        placements.update({t: t for t in tiles})  # filmed tiles always show their own layout
        nyc_look.dress(stage, generator, placements)
        stairs = [[item[1:] for item in generator.layouts[t].decor if item[0] == "stairs"] for t in tiles]
        civic = getattr(scenario, "social", False)
        clips_dir = os.path.join(PROJECT, "src/g1_patrol/assets/data/people")
        filmed = list(range(len(args.scenarios)))
        if civic:  # people (and the parents pushing strollers); wheelchair users, strollers and dogs by civic_look
            import civic_look

            _, stroller_slots = civic_look.wide_slots(scenario)
            people_slots = [k for k in range(scenario.num_peds) if int(scenario.slot_kind[k]) == 0] + stroller_slots
            pushers = {k: civic_look.PUSHER_BEHIND for k in stroller_slots}
            crowd = AnimatedCrowd(scenario, filmed, clips_dir, stairs=stairs, slots=people_slots, offsets=pushers)
            centers = [nyc_look.tile_center(generator, *t) for t in tiles]
            signals = civic_look.SignalHeads(stage, scenario, tiles, generator, centers)
            fallen = civic_look.FallenPerson(stage, scenario, filmed, clips_dir)
            dogs = civic_look.Dogs(stage, scenario, filmed)
            wheelchairs = civic_look.Wheelchairs(stage, scenario, filmed, clips_dir)
            strollers = civic_look.Strollers(stage, scenario, filmed)
            hands = base.action_manager.get_term("hands")
        else:
            crowd = AnimatedCrowd(scenario, filmed, clips_dir, stairs=stairs)
        cameras = [FollowCamera(stage, f"/World/Demo/Cameras/cam_{e}", name, args.width, args.height, args.camera)
                   for e, name in enumerate(args.scenarios)]  # fmt: skip
        camera_bounds = [
            _camera_bounds(generator.layouts[t], nyc_look.tile_center(generator, *t), inside_shed=args.camera == "chase")
            for t in tiles
        ]
        overlay = Overlay(args.width, args.height, args.label)
        frame_dt = base.physics_dt * args.render_interval
        fps = 1.0 / frame_dt

        clips = [None] * len(args.scenarios)
        # check per clip: characters appearing or vanishing in view, frames of characters moving backwards, frames in
        # which a part of the robot is inside a structure
        people_events = [[0, 0, 0, 0] for _ in args.scenarios]
        touches = [collections.Counter() for _ in args.scenarios]  # (link, structure) of the frames counted above
        body_names = base.scene["robot"].body_names
        prev_people = [None] * len(args.scenarios)
        clip_t = [0.0] * len(args.scenarios)
        clip_frames = [0] * len(args.scenarios)
        clip_count = [0] * len(args.scenarios)

        def open_clip(e):
            clip_count[e] += 1
            path = os.path.join(out_dir, f"{args.scenarios[e]}_L{levels[e]}_{args.camera}_ep{clip_count[e]:02d}.mp4")
            clips[e] = (imageio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=8), path)
            clip_t[e], clip_frames[e] = 0.0, 0
            people_events[e], prev_people[e] = [0, 0, 0, 0], None
            touches[e].clear()

        def close_clip(e, outcome):
            writer, path = clips[e]
            writer.close()
            if clip_frames[e] == 0:
                print(f"[DEMO] {args.scenarios[e]} episode {clip_count[e]}: {outcome} after {clip_t[e]:.1f} s, no frames", flush=True)
                return
            final = path.replace(".mp4", f"_{outcome}.mp4")
            os.replace(path, final)
            appear, vanish, backward, touching = people_events[e]
            print(f"[DEMO] {os.path.basename(final)} ({clip_t[e]:.1f} s, {clip_frames[e]} frames; people in view: {appear} appeared, "
                  f"{vanish} vanished, {backward} character-frames moving backwards; robot inside a structure in "
                  f"{touching} frames)", flush=True)
            if touching:
                print(f"[DEMO]   robot inside: " + ", ".join(f"{link} in {mat} ({n})" for (link, mat), n in touches[e].most_common()),
                      flush=True)

        import carb
        import omni.kit.app

        settings = carb.settings.get_settings()
        settings.set("/rtx/post/dlss/execMode", 2)  # DLSS "quality": render closer to native resolution

        def on_render(_):
            # pose people and cameras for this frame, render it (headless Kit only renders on request), read it
            crowd.update(frame_dt)
            if civic:
                signals.update(clip_t[0])
                fallen.update()
                dogs.update(frame_dt)
                wheelchairs.update(frame_dt)
                strollers.update(crowd)
            pos = base.scene["robot"].data.root_pos_w.torch.cpu().numpy()
            yaw = scenario.robot_yaw().cpu().numpy()
            centers = scenario.tile_center.cpu().numpy()
            for e, cam in enumerate(cameras):
                cam.update(pos[e], float(yaw[e]), centers[e], frame_dt, camera_bounds[e])
            play = settings.get("/app/player/playSimulations")
            settings.set("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()
            settings.set("/app/player/playSimulations", bool(play))
            links = base.scene["robot"].data.body_pos_w.torch.cpu().numpy()
            origins = terrain.env_origins.cpu().numpy()
            for e, cam in enumerate(cameras):
                local = links[e] - np.array([centers[e][0], centers[e][1], origins[e][2]])
                inside = robot_in_structure(generator.layouts[tiles[e]], local) if clip_t[e] > 2 * frame_dt else None
                if inside is not None:
                    people_events[e][3] += 1
                    touches[e][(body_names[inside[0]], inside[1])] += 1
                visible, points, backward = crowd.visible[e], crowd.points[e], crowd.backward[e]
                if civic:
                    visible = visible | wheelchairs.visible[e]
                view = cam.in_view(points)
                if clip_t[e] > 2 * frame_dt and prev_people[e] is not None:
                    was_visible, was_in_view = prev_people[e]
                    people_events[e][0] += int((visible & ~was_visible & view).sum())
                    people_events[e][1] += int((~visible & was_visible & was_in_view).sum())
                    people_events[e][2] += int((backward & visible & view).sum())
                prev_people[e] = (visible, view)
                frame = cam.frame()
                if frame is not None and clip_t[e] > 2 * frame_dt:  # skip the first frames after a teleport
                    title = TITLES[args.scenarios[e]]
                    caption = civic_look.caption(scenario, hands, e) if civic else None
                    shown = frame if args.no_overlay else overlay.draw(frame, title, f"New York · difficulty {levels[e] + 1}/10", clip_t[e], caption)
                    clips[e][0].append_data(shown)
                    clip_frames[e] += 1
                    if args.snapshot_every and clip_frames[e] % max(1, int(args.snapshot_every * fps)) == 1:
                        imageio.imwrite(os.path.join(out_dir, f"still_{args.scenarios[e]}_{args.camera}_ep{clip_count[e]:02d}_{clip_t[e]:05.1f}s.png"), frame)
                clip_t[e] += frame_dt

        obs, _ = env.reset()
        obs = env.get_observations()
        for e in range(len(args.scenarios)):
            open_clip(e)
        for _ in range(30):  # let textures and materials stream in before recording
            on_render(None)
        for e in range(len(args.scenarios)):
            clip_t[e] = 0.0
        base.sim.add_render_callback("nyc_demo", on_render, order=100)
        tm = base.termination_manager
        steps = int(args.seconds / base.step_dt)
        with torch.inference_mode():
            for step in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                for e in torch.nonzero(dones[: len(args.scenarios)]).flatten().tolist():
                    outcome = next((n for n in OUTCOMES if bool(tm.get_term(n)[e])), "done")
                    close_clip(e, outcome)
                    open_clip(e)
                if step % 100 == 0:
                    print(f"[DEMO] t = {step * base.step_dt:.0f} s", flush=True)
        base.sim.remove_render_callback("nyc_demo")
        for e in range(len(args.scenarios)):
            close_clip(e, "unfinished")


if __name__ == "__main__":
    main(sys.argv[1:])
