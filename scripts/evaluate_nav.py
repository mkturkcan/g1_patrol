"""Evaluate a navigation policy headlessly: how often episodes end at the goal, in a collision or a fall.

Runs the policy in many parallel environments (play-mode config: no random shoves) for ``--num_steps`` navigation
steps (10 per second) and counts how every episode ended (one outcome per episode, in the order of ``OUTCOMES``).
Pass ``--video`` to also record an RTX clip of robot 0.

For the New York task (``G1Patrol-Navigate-NYC*``) the curriculum is frozen and robots are spread uniformly over all
difficulty levels, so the results break down by scenario type x difficulty (easy: levels 0-3, medium: 4-6,
hard: 7-9). Pass ``--keep_curriculum`` to evaluate with the training curriculum instead.

Outputs are written next to the checkpoint, in ``<run_dir>/eval/nav_metrics_<physics>.json``, or to ``--out``.

Usage:
    python scripts/evaluate_nav.py --checkpoint latest physics=isaacsim_physx
    python scripts/evaluate_nav.py --task G1Patrol-Navigate-NYC --checkpoint <model.pt> --num_envs 2400
"""

import contextlib
import json
import os
import sys

import torch

from isaaclab.app import launch_simulation

from isaaclab_rl.entrypoints.common import (
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    pre_launch_video_config,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

from evaluate import cli_args, parse_args, resolve_checkpoint

OUTCOMES = ("goal_reached", "pedestrian_collision", "obstacle_collision", "roadway", "fall", "time_out")
"""Episode outcomes in order of precedence; tasks without a term (e.g. no roadway) skip it."""
LEVEL_BUCKETS = {"easy": (0, 3), "medium": (4, 6), "hard": (7, 9)}
MOTION_TERMS = (
    "forward_command_mps",  # mean forward velocity command
    "abs_lateral_command_mps",
    "abs_yaw_rate_command_radps",
    "fraction_lateral_above_0.25mps",
    "fraction_backward",  # forward command below -0.05 m/s
    "fraction_yaw_rate_above_0.8radps",
    "fraction_saturated",  # requested command within 5% of its range limit (mean over the 3 components)
    "forward_command_change_mps2",  # mean |d command / dt|
    "lateral_command_change_mps2",
    "yaw_rate_command_change_radps2",
    "planar_speed_mps",
    "torso_tilt_deg",
    "knee_flexion_deg",
)


def main(argv: list[str]) -> None:
    if not any(a == "--task" or a.startswith("--task=") for a in argv):
        argv = ["--task", "G1Patrol-Navigate-Crowd", *argv]
    keep_curriculum = "--keep_curriculum" in argv
    argv = [a for a in argv if a != "--keep_curriculum"]
    args_cli = parse_args(argv)
    installed_version = check_rsl_rl_version()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    nyc = hasattr(env_cfg.commands, "nyc")
    if nyc and not keep_curriculum:
        env_cfg.scene.terrain.max_init_terrain_level = None  # uniform over all levels
        env_cfg.curriculum.levels = None
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = resolve_checkpoint(args_cli, agent_cfg, log_root_path)
        out_dir = os.path.join(os.path.dirname(resume_path), "eval")
        os.makedirs(out_dir, exist_ok=True)
        env_cfg.log_dir = os.path.dirname(resume_path)
        apply_video_recording(env_cfg, env_cfg.log_dir, args_cli, subdir="eval", checkpoint_path=resume_path)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base_env = env.unwrapped
        dev = base_env.device
        term_manager = base_env.termination_manager
        scenario = base_env.command_manager.get_term("nyc" if nyc else "crowd")
        crowd_contacts = hasattr(scenario, "contact_ped_aware")
        contacts = {"aware": [], "robot_approach": [], "ped_approach": []}
        steps_in_episode = torch.zeros(base_env.num_envs, device=dev)
        success_steps = []

        # per-cell statistics; a cell is (scenario type, difficulty bucket) for NYC, a single cell otherwise
        if nyc:
            from g1_patrol.tasks.navigation.nyc.tiles import KINDS

            cells = [(k, b) for k in KINDS for b in LEVEL_BUCKETS]
            bucket_of_level = torch.zeros(scenario.layout.kind.shape[0], dtype=torch.long, device=dev)
            for i, (lo, hi) in enumerate(LEVEL_BUCKETS.values()):
                bucket_of_level[lo : hi + 1] = i
        else:
            cells = [("all", "all")]
        num_cells = len(cells)
        outcomes = [name for name in OUTCOMES if name in term_manager.active_terms]
        counts = {name: torch.zeros(num_cells, device=dev) for name in outcomes}
        cell_steps = torch.zeros(num_cells, device=dev)
        cell_delay = torch.zeros(num_cells, device=dev)
        cell_street = torch.zeros(num_cells, device=dev)
        cell_close = torch.zeros(num_cells, device=dev)
        cell_goal_time = torch.zeros(num_cells, device=dev)

        # motion quality: the velocity commands sent to the walker and the walker's posture
        walker = base_env.action_manager.get_term("walker") if nyc else None
        robot = base_env.scene["robot"]
        knees = [i for i, n in enumerate(robot.joint_names) if "knee" in n]
        motion = torch.zeros(len(MOTION_TERMS), device=dev)
        motion_steps = torch.zeros((), device=dev)

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.num_steps):
                # the episode's cell, read before the step resets finished episodes onto new tiles
                if nyc:
                    cell = scenario.layout.kind[scenario.tile_row, scenario.tile_col] * len(LEVEL_BUCKETS)
                    cell = cell + bucket_of_level[scenario.tile_row]
                else:
                    cell = torch.zeros(base_env.num_envs, dtype=torch.long, device=dev)
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                steps_in_episode += 1
                if walker is not None:
                    live = base_env.episode_length_buf > 1  # skip the first step after a reset
                    cmd, step = walker.command, (walker.command - walker.prev_command).abs() / base_env.step_dt
                    span = walker._high - walker._low
                    saturated = torch.minimum(walker.target - walker._low, walker._high - walker.target) < 0.05 * span
                    gravity = robot.data.projected_gravity_b.torch
                    values = torch.stack([
                        cmd[:, 0], cmd[:, 1].abs(), cmd[:, 2].abs(),
                        (cmd[:, 1].abs() > 0.25).float(), (cmd[:, 0] < -0.05).float(), (cmd[:, 2].abs() > 0.8).float(),
                        saturated.float().mean(dim=1), step[:, 0], step[:, 1], step[:, 2],
                        torch.linalg.norm(robot.data.root_lin_vel_b.torch[:, :2], dim=1),
                        torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))),
                        torch.rad2deg(robot.data.joint_pos.torch[:, knees].abs().mean(dim=1)),
                    ], dim=1)
                    motion += (values * live[:, None]).sum(dim=0)
                    motion_steps += live.sum()

                cell_steps += torch.bincount(cell, minlength=num_cells)
                if nyc:
                    cell_delay += torch.bincount(cell, weights=scenario.ped_delay.sum(dim=1), minlength=num_cells)
                    cell_street += torch.bincount(cell, weights=scenario.in_street().float(), minlength=num_cells)
                    close = (scenario.pedestrian_clearance() < 0.5).float()
                    cell_close += torch.bincount(cell, weights=close, minlength=num_cells)
                assigned = torch.zeros(base_env.num_envs, dtype=torch.bool, device=dev)
                for name in outcomes:
                    ended = term_manager.get_term(name) & ~assigned
                    assigned |= ended
                    counts[name] += torch.bincount(cell[ended], minlength=num_cells)
                reached = term_manager.get_term("goal_reached")
                cell_goal_time += torch.bincount(cell[reached], weights=steps_in_episode[reached], minlength=num_cells)
                success_steps.extend(steps_in_episode[reached].tolist())
                hit = term_manager.get_term("pedestrian_collision")
                if crowd_contacts and hit.any():
                    contacts["aware"].extend(scenario.contact_ped_aware[hit].tolist())
                    contacts["robot_approach"].extend(scenario.contact_robot_approach_speed[hit].tolist())
                    contacts["ped_approach"].extend(scenario.contact_ped_approach_speed[hit].tolist())
                steps_in_episode[dones.bool()] = 0

        totals = {name: int(counts[name].sum()) for name in outcomes}
        episodes = sum(totals.values())
        metrics = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "physics": type(base_env.cfg.sim.physics).__name__,
            "num_envs": base_env.num_envs,
            "episodes": episodes,
            **{f"{name}_rate": totals[name] / max(episodes, 1) for name in outcomes},
            "mean_time_to_goal_s": sum(success_steps) / max(len(success_steps), 1) * base_env.step_dt,
        }
        if crowd_contacts:
            n = max(len(contacts["aware"]), 1)
            robot_faster = sum(r > p for r, p in zip(contacts["robot_approach"], contacts["ped_approach"]))
            metrics["pedestrian_collisions"] = {
                "count": len(contacts["aware"]),
                "with_robot_aware_pedestrian": sum(contacts["aware"]) / n,
                "robot_closing_faster_than_pedestrian": robot_faster / n,
                "mean_robot_approach_speed": sum(contacts["robot_approach"]) / n,
                "mean_pedestrian_approach_speed": sum(contacts["ped_approach"]) / n,
            }
        if nyc:
            metrics["curriculum"] = "training" if keep_curriculum else "frozen, uniform levels"
            metrics["mean_pedestrian_delay_mps"] = float(cell_delay.sum() / cell_steps.sum().clamp(min=1))
            metrics["time_within_0.5m_of_pedestrian"] = float(cell_close.sum() / cell_steps.sum().clamp(min=1))
            metrics["time_in_street"] = float(cell_street.sum() / cell_steps.sum().clamp(min=1))
            per_cell = {}
            for i, (kind, bucket) in enumerate(cells):
                eps = sum(float(counts[name][i]) for name in outcomes)
                per_cell[f"{kind}/{bucket}"] = {
                    "episodes": int(eps),
                    **{f"{name}_rate": float(counts[name][i]) / max(eps, 1) for name in outcomes},
                    "mean_time_to_goal_s": float(cell_goal_time[i] / counts["goal_reached"][i].clamp(min=1))
                    * base_env.step_dt,
                    "mean_pedestrian_delay_mps": float(cell_delay[i] / cell_steps[i].clamp(min=1)),
                    "time_within_0.5m_of_pedestrian": float(cell_close[i] / cell_steps[i].clamp(min=1)),
                    "time_in_street": float(cell_street[i] / cell_steps[i].clamp(min=1)),
                }
            metrics["per_scenario"] = per_cell
            metrics["motion"] = {k: float(v) for k, v in zip(MOTION_TERMS, motion / motion_steps.clamp(min=1))}

        print("\n[RESULT] Navigation metrics:")
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_scenario"}, indent=2))
        if nyc:
            print(f"\n{'scenario/difficulty':24s} {'eps':>5s} {'goal':>6s} {'ped':>6s} {'obst':>6s} {'road':>6s} {'fall':>6s} "
                  f"{'tout':>6s} {'t_goal':>7s} {'delay':>6s}")
            for name, m in metrics["per_scenario"].items():
                print(f"{name:24s} {m['episodes']:5d} {m['goal_reached_rate']:6.2f} {m['pedestrian_collision_rate']:6.2f} "
                      f"{m['obstacle_collision_rate']:6.2f} {m.get('roadway_rate', 0.0):6.2f} {m['fall_rate']:6.2f} {m['time_out_rate']:6.2f} "
                      f"{m['mean_time_to_goal_s']:7.1f} {m['mean_pedestrian_delay_mps']:6.3f}")
        tag = metrics["physics"].removesuffix("Cfg").lower() + ("_train_cfg" if args_cli.train_env_cfg else "")
        out_path = args_cli.out or os.path.join(out_dir, f"nav_metrics_{tag}.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
