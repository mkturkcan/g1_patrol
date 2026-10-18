"""How do navigation episodes fail? The robot's state at every episode end, by cause.

Runs a navigation policy as ``evaluate_nav.py`` does (curriculum disabled, robots spread uniformly over all tiles,
play mode) and records, inside each termination term (before the reset), the tile type and level, the robot's
position on the tile, its speed and heading-frame velocity, the walker command, the nearest structure (material,
bearing in the robot frame, speed towards it), the gap to the nearest pedestrian and to the roadway, the time since the
episode start, and the robot's last 2 s. Prints per-cause summaries by tile type and difficulty and saves the records.

Usage:
    python scripts/analyze_terminations.py --checkpoint <teacher model.pt> --out summary.json physics=isaacsim_physx
"""

import argparse
import collections
import functools
import json
import math
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401

HISTORY = 20
"""Navigation steps of history kept per robot (2 s)."""
CAUSES = ("obstacle_collision", "pedestrian_collision", "roadway", "fall", "time_out")
LEVELS = {"easy": (0, 3), "medium": (4, 6), "hard": (7, 9)}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="G1Patrol-Navigate-NYC")
    parser.add_argument("--num_envs", type=int, default=2400)
    parser.add_argument("--num_steps", type=int, default=600)
    parser.add_argument("--records", default="terminations.npz", help="Path of the per-episode records.")
    parser.add_argument("--out", default=None, help="Path of the summary JSON.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.levels = None
    records: list[dict] = []
    history: dict[str, torch.Tensor] = {}

    def snapshot(env, done: torch.Tensor, cause: str):
        from g1_patrol.tasks.navigation.mdp.nyc_scenario import OVERHEAD_Z, box_distance
        from g1_patrol.tasks.navigation.nyc.terrain import NycTerrainGenerator

        ids = done.nonzero().flatten()
        if len(ids) == 0:
            return
        s = env.command_manager.get_term("nyc")
        walker = env.action_manager.get_term("walker")
        p, yaw = s.robot_pos()[ids], s.robot_yaw()[ids]
        v = s.robot.data.root_lin_vel_w.torch[ids, :2]
        c, sn = torch.cos(yaw), torch.sin(yaw)
        v_b = torch.stack([c * v[:, 0] + sn * v[:, 1], -sn * v[:, 0] + c * v[:, 1]], -1)
        # nearest structure surface point
        circles, boxes = s.circles[ids], s.boxes[ids]
        d_c = torch.linalg.norm(p[:, None] - circles[..., :2], dim=-1) - circles[..., 2]
        d_c = torch.where(s.circle_valid[ids] & (circles[..., 3] < OVERHEAD_Z), d_c, torch.full_like(d_c, 1e3))
        d_b, closest_b = box_distance(p[:, None], boxes)
        d_b, closest_b = d_b[:, 0], closest_b[:, 0]
        d_b = torch.where(s.box_valid[ids] & (boxes[..., 5] < OVERHEAD_Z), d_b, torch.full_like(d_b, 1e3))
        ic, ib = d_c.argmin(1), d_b.argmin(1)
        rows = torch.arange(len(ids), device=p.device)
        use_box = d_b[rows, ib] < d_c[rows, ic]
        center = circles[rows, ic, :2]
        towards = (p - center) / torch.linalg.norm(p - center, dim=-1, keepdim=True).clamp(min=1e-6)
        point = torch.where(use_box[:, None], closest_b[rows, ib], center + towards * circles[rows, ic, 2:3])
        rel = point - p
        bearing = torch.remainder(torch.atan2(rel[:, 1], rel[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
        approach = (v * rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)).sum(-1)
        layouts = NycTerrainGenerator.latest.layouts
        past = history["state"][ids].cpu().numpy() if "state" in history else None
        tile_row, tile_col = s.tile_row[ids].tolist(), s.tile_col[ids].tolist()
        for j, e in enumerate(ids.tolist()):
            lay = layouts[(tile_row[j], tile_col[j])]
            mat = lay.box_mats[int(ib[j])] if bool(use_box[j]) else lay.circle_mats[int(ic[j])]
            records.append({
                "cause": cause, "kind": lay.kind, "level": tile_row[j], "t": float(env.episode_length_buf[e] * env.step_dt),
                "x": float(p[j, 0]), "y": float(p[j, 1]), "yaw": float(yaw[j]), "vx": float(v_b[j, 0]), "vy": float(v_b[j, 1]),
                "command": walker.command[e].tolist(), "nearest": mat, "bearing": float(bearing[j]),
                "approach": float(approach[j]), "pedestrian_gap": float(s.pedestrian_clearance()[e]),
                "roadway_gap": float(s.street_clearance()[e]), "goal": s.goal[e].tolist(), "start": s.start[e].tolist(),
                "history": past[j] if past is not None else None,
            })  # fmt: skip

    def recording(func, cause):
        @functools.wraps(func)
        def term(env, **kwargs):
            done = func(env, **kwargs)
            snapshot(env, done.bool(), cause)
            return done

        return term

    for cause in CAUSES:
        term_cfg = getattr(env_cfg.terminations, cause)
        term_cfg.func = recording(term_cfg.func, cause)

    with launch_simulation(env_cfg, args):
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        s = base.command_manager.get_term("nyc")
        walker = base.action_manager.get_term("walker")
        # per step: x, y, yaw, heading-frame vx, vy, walker command (3), structure gap, pedestrian gap
        history["state"] = torch.zeros(base.num_envs, HISTORY, 10, device=base.device)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.num_steps):
                obs, _, dones, _ = env.step(policy(obs))
                p, yaw = s.robot_pos(), s.robot_yaw()
                v = s.robot.data.root_lin_vel_w.torch[:, :2]
                c, sn = torch.cos(yaw), torch.sin(yaw)
                row = torch.cat([p, yaw[:, None], (c * v[:, 0] + sn * v[:, 1])[:, None], (-sn * v[:, 0] + c * v[:, 1])[:, None],
                                 walker.command, s.obstacle_clearance()[:, None], s.pedestrian_clearance().clamp(max=10.0)[:, None]], 1)
                state = torch.roll(history["state"], -1, dims=1)
                state[:, -1] = row
                fresh = dones.bool()
                state[fresh] = row[fresh][:, None].expand(-1, HISTORY, -1)
                history["state"] = state
        np.savez(args.records, records=np.array(records, dtype=object))
        # (inside the simulation context: closing the app ends the process)
        summary = {"checkpoint": args.checkpoint, "records": len(records), "causes": dict(collections.Counter(r["cause"] for r in records))}
        groups = collections.defaultdict(list)
        for r in records:
            level = next(name for name, (lo, hi) in LEVELS.items() if lo <= r["level"] <= hi)
            groups[(r["cause"], r["kind"], level)].append(r)
        table = {}
        for (cause, kind, level), rs in sorted(groups.items()):
            bearing = np.abs([r["bearing"] for r in rs])
            speed = np.hypot([r["vx"] for r in rs], [r["vy"] for r in rs])
            table[f"{cause}/{kind}/{level}"] = {
                "count": len(rs),
                "structure_beside": float(np.mean((bearing >= math.pi / 4) & (bearing < 3 * math.pi / 4))),
                "structure_ahead": float(np.mean(bearing < math.pi / 4)),
                "median_speed_mps": float(np.median(speed)),
                "median_approach_mps": float(np.median([r["approach"] for r in rs])),
                "pedestrian_within_0.3m": float(np.mean([r["pedestrian_gap"] < 0.3 for r in rs])),
                "within_2s_of_start": float(np.mean([r["t"] < 2.0 for r in rs])),
                "nearest": collections.Counter(r["nearest"] for r in rs).most_common(3),
            }
        summary["groups"] = table
        print("\n[RESULT] Termination analysis:")
        print(json.dumps(summary, indent=1))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
