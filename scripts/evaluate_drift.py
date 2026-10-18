"""Drift of the locomotion policy under constant straight commands.

Every robot gets a fixed forward command (0.3, 0.6 or 0.9 m/s, no lateral velocity, no yaw rate) on the city terrain,
with the training randomization and sensor noise and without shoves. After 1 s of settling the script reports, per
speed, the mean absolute 1 s average of the heading-frame lateral velocity (sustained sideways drift; the sway of the
gait averages out), the mean absolute 1 s average of the yaw rate, and the lateral offset and heading change after
3 s, measured from the line and heading the robot had at the start.

Usage:
    python scripts/evaluate_drift.py --checkpoint <model.pt> physics=isaacsim_physx
"""

import argparse
import json
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401

SPEEDS = (0.3, 0.6, 0.9)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="G1Patrol-Walk-City")
    parser.add_argument("--num_envs", type=int, default=768, help="Robots (split evenly over the speeds).")
    parser.add_argument("--out", default=None, help="Path of the metrics JSON.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = args.num_envs
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e6, 1.0e6)
    command.rel_standing_envs = 0.0
    command.heading_command = False
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.events.push_robot = None
    with launch_simulation(env_cfg, args):
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        robot = base.scene["robot"]
        term = base.command_manager.get_term("base_velocity")
        n, dev = base.num_envs, base.device
        speeds = torch.tensor(SPEEDS, device=dev)[torch.arange(n, device=dev) % len(SPEEDS)]
        per_second = int(round(1.0 / base.step_dt))
        steps = 6 * per_second
        lateral = torch.zeros(n, steps, device=dev)
        yaw_rate = torch.zeros_like(lateral)
        position = torch.zeros(n, steps, 2, device=dev)
        heading = torch.zeros_like(lateral)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        obs = env.get_observations()
        with torch.inference_mode():
            for t in range(steps):
                term.vel_command_b[:, 0] = speeds
                term.vel_command_b[:, 1:] = 0.0
                obs, _, dones, _ = env.step(policy(obs))
                alive &= ~dones.bool()
                q = robot.data.root_quat_w.torch  # (x, y, z, w)
                yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
                v = robot.data.root_lin_vel_w.torch[:, :2]
                lateral[:, t] = -torch.sin(yaw) * v[:, 0] + torch.cos(yaw) * v[:, 1]
                yaw_rate[:, t] = robot.data.root_ang_vel_w.torch[:, 2]
                position[:, t] = robot.data.root_pos_w.torch[:, :2]
                heading[:, t] = yaw
        t0, t3 = per_second, 4 * per_second  # after 1 s of settling; 3 s later
        results = {"checkpoint": args.checkpoint, "robots": n, "falls": int((~alive).sum()), "speeds": {}}
        for speed in SPEEDS:
            m = alive & (speeds == speed)
            windows = lambda x: x[m, t0:].unfold(1, per_second, per_second).mean(-1)  # noqa: E731
            d = position[m, t3] - position[m, t0]
            h0 = heading[m, t0]
            offset = -torch.sin(h0) * d[:, 0] + torch.cos(h0) * d[:, 1]
            turn = torch.remainder(heading[m, t3] - h0 + torch.pi, 2 * torch.pi) - torch.pi
            results["speeds"][f"{speed:.1f}"] = {
                "robots": int(m.sum()),
                "lateral_drift_mps": float(windows(lateral).abs().mean()),
                "yaw_rate_drift_radps": float(windows(yaw_rate).abs().mean()),
                "lateral_offset_3s_m": float(offset.abs().mean()),
                "lateral_offset_3s_p90_m": float(offset.abs().quantile(0.9)),
                "heading_change_3s_rad": float(turn.abs().mean()),
            }
        print("\n[RESULT] Drift metrics:")
        print(json.dumps(results, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
