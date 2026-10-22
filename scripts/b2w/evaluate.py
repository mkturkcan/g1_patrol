"""Evaluate a B2-W navigation policy (teacher or lidar student) headlessly.

Counts how episodes end (goal, pedestrian or structure contact, roadway or drop, fall, time-out) per tile kind and
difficulty (easy, medium, hard thirds of the levels), with the curriculum frozen and robots spread over all levels.
Stress tests of the sim-to-real assumptions:

* ``--controller``: a built-in controller worse than in training -- ``laggy`` (0.16-0.25 s latency), ``sluggish``
  (twice the response time, half the acceleration), ``weak`` (moves at 70-80 % of the command), ``drifty`` (twice the
  yaw drift) -- or ``sysid:<file.json>`` with the values fitted on the robot (``scripts/b2w/sysid.py fit --json``);
* ``--lidar_dropout`` / ``--odometry_drift``: a degraded lidar map (student); ``--camera_failure``: the fused
  student's depth cameras deliver nothing;
* ``--twin <place.yaml>``: only the digital twin of the real place (:mod:`g1_patrol.tasks.navigation.nyc.twin`);
* ``--map_stats``: statistics of the student's lidar maps, to compare with those of a recorded drive
  (``scripts/b2w/replay.py``).

Usage:
    python scripts/b2w/evaluate.py --task G1Patrol-B2W-Navigate-Lidar --checkpoint <model.pt> --num_envs 1024 \\
        --num_steps 2000 --controller laggy physics=isaacsim_physx
"""

import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import launch_simulation

from isaaclab_rl.entrypoints.common import close_env, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

from evaluate import cli_args, parse_args, resolve_checkpoint  # noqa: E402

OUTCOMES = ("goal_reached", "pedestrian_collision", "obstacle_collision", "roadway", "fall", "time_out")
BUCKETS = ("easy", "medium", "hard")
CONTROLLERS = {
    "laggy": {"delay_range": (0.16, 0.25)},
    "sluggish": {"tau_scale": 2.0, "accel_scale": 0.5},
    "weak": {"gain_range": ((0.7, 0.8), (0.5, 0.6), (0.7, 0.8))},
    "drifty": {"drift_scale": 2.0},
}


def _extra(argv: list[str], name: str, default=None, flag: bool = False):
    """Pop ``--name value`` (or ``--name`` for flags) from argv."""
    for i, a in enumerate(argv):
        if a == f"--{name}":
            if flag:
                argv.pop(i)
                return True
            argv.pop(i)
            return argv.pop(i)
    return False if flag else default


def _apply_controller(base_cfg, spec: str):
    if spec.startswith("sysid:"):  # the robot's fitted values, as narrow ranges around them
        fit = json.load(open(spec[len("sysid:") :]))
        axes = ("forward", "lateral", "yaw rate")
        base_cfg.delay_range = (fit["delay_s"], fit["delay_s"])
        base_cfg.gain_range = tuple((fit[a]["gain"] * 0.97, fit[a]["gain"] * 1.03) for a in axes)
        base_cfg.tau_range = tuple((fit[a]["tau"] * 0.9, fit[a]["tau"] * 1.1) for a in axes)
        base_cfg.accel_range = tuple((fit[a]["accel"] * 0.9, fit[a]["accel"] * 1.1) for a in axes)
        base_cfg.deadband_max = tuple(max(fit[a]["deadband"], 1e-3) for a in axes)
        base_cfg.drift_max = abs(fit["drift"])
        base_cfg.noise_max = tuple(fit[a]["noise"] for a in axes)
        return
    c = CONTROLLERS[spec]
    for key in ("delay_range", "gain_range"):
        if key in c:
            setattr(base_cfg, key, c[key])
    if "tau_scale" in c:
        base_cfg.tau_range = tuple((lo * c["tau_scale"], hi * c["tau_scale"]) for lo, hi in base_cfg.tau_range)
    if "accel_scale" in c:
        base_cfg.accel_range = tuple((lo * c["accel_scale"], hi * c["accel_scale"]) for lo, hi in base_cfg.accel_range)
    if "drift_scale" in c:
        base_cfg.drift_max *= c["drift_scale"]


def main(argv: list[str]) -> None:
    argv = list(argv)
    controller = _extra(argv, "controller", "nominal")
    twin = _extra(argv, "twin")
    lidar_dropout = _extra(argv, "lidar_dropout")
    odometry_drift = _extra(argv, "odometry_drift")
    want_map_stats = _extra(argv, "map_stats", flag=True)
    camera_failure = _extra(argv, "camera_failure", flag=True)
    if not any(a == "--task" or a.startswith("--task=") for a in argv):
        argv = ["--task", "G1Patrol-B2W-Navigate-Fused", *argv]
    args_cli = parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    env_cfg.scene.terrain.max_init_terrain_level = None  # uniform over all levels
    env_cfg.curriculum.levels = None
    if twin:
        from g1_patrol.b2w import FOOTPRINT
        from g1_patrol.tasks.navigation.nyc.terrain_cfg import nyc_terrain_generator_cfg

        env_cfg.scene.terrain.terrain_generator = nyc_terrain_generator_cfg(
            num_rows=6, variants_per_kind=4, kinds=["twin"], route_half_width=FOOTPRINT[1], twin=os.path.abspath(twin))
    if controller != "nominal":
        _apply_controller(env_cfg.actions.base, controller)
    group = next((g for g in ("lidar_map", "height_map") if hasattr(env_cfg.observations, g)), None)
    if group is not None:
        params = getattr(getattr(env_cfg.observations, group), group).params
        if camera_failure:
            params["extra_sensors"] = tuple({**e, "episode_dropout": 1.0} for e in params.get("extra_sensors", ()))
        if lidar_dropout is not None:
            params["dropout"] = float(lidar_dropout)
        if odometry_drift is not None:
            params["drift_pos"], params["drift_yaw"] = float(odometry_drift), float(odometry_drift) * 0.4
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        from g1_patrol.tasks.navigation.nyc.indoor import ALL_KINDS

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        env_cfg.seed = agent_cfg.seed
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = resolve_checkpoint(args_cli, agent_cfg, log_root_path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base_env = env.unwrapped
        dev = base_env.device
        tm = base_env.termination_manager
        scenario = base_env.command_manager.get_term("nyc")
        base = base_env.action_manager.get_term("base")
        rows = scenario.layout.num_rows
        kinds = sorted(set(scenario.layout.kind.flatten().tolist()))
        cells = [(ALL_KINDS[k], b) for k in kinds for b in BUCKETS]
        cell_index = torch.full((len(ALL_KINDS), 3), -1, dtype=torch.long, device=dev)
        for i, (name, b) in enumerate(cells):
            cell_index[ALL_KINDS.index(name), BUCKETS.index(b)] = i
        bucket_of_row = torch.tensor([min(3 * r // rows, 2) for r in range(rows)], device=dev)
        outcomes = [o for o in OUTCOMES if o in tm.active_terms]
        counts = {o: torch.zeros(len(cells), device=dev) for o in outcomes}
        goal_time = torch.zeros(len(cells), device=dev)
        steps_in_episode = torch.zeros(base_env.num_envs, device=dev)
        motion = {"speed": 0.0, "backward": 0.0, "lateral": 0.0, "command_change": 0.0, "near_people": 0.0, "steps": 0}
        maps = []
        obs = env.get_observations()
        with torch.inference_mode():
            for t in range(args_cli.num_steps):
                cell = cell_index[scenario.layout.kind[scenario.tile_row, scenario.tile_col], bucket_of_row[scenario.tile_row]]
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                steps_in_episode += 1
                assigned = torch.zeros(base_env.num_envs, dtype=torch.bool, device=dev)
                for name in outcomes:
                    ended = tm.get_term(name) & ~assigned
                    assigned |= ended
                    counts[name] += torch.bincount(cell[ended], minlength=len(cells))
                reached = tm.get_term("goal_reached")
                goal_time += torch.bincount(cell[reached], weights=steps_in_episode[reached], minlength=len(cells))
                steps_in_episode[dones.bool()] = 0
                live = base_env.episode_length_buf > 1
                motion["speed"] += float((torch.linalg.norm(base.velocity_b[:, :2], dim=1) * live).sum())
                motion["backward"] += float(((base.command[:, 0] < -0.05) & live).sum())
                motion["lateral"] += float((base.command[:, 1].abs() * live).sum())
                motion["command_change"] += float(((base.command - base.prev_command).abs().sum(dim=1) * live).sum())
                motion["near_people"] += float(((scenario.pedestrian_clearance() < 0.5) & live).sum())
                motion["steps"] += int(live.sum())
                if want_map_stats and group in obs.keys() and t % 10 == 0:
                    maps.append(obs[group][:64].cpu())
        episodes = int(sum(float(c.sum()) for c in counts.values()))
        metrics = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "controller": controller,
            "twin": twin,
            "lidar_dropout": lidar_dropout,
            "camera_failure": camera_failure,
            "odometry_drift": odometry_drift,
            "episodes": episodes,
            **{f"{o}_rate": float(counts[o].sum()) / max(episodes, 1) for o in outcomes},
            "mean_time_to_goal_s": float(goal_time.sum() / counts["goal_reached"].sum().clamp(min=1)) * base_env.step_dt,
            "motion": {
                "mean_speed_mps": motion["speed"] / max(motion["steps"], 1),
                "fraction_backward": motion["backward"] / max(motion["steps"], 1),
                "mean_abs_lateral_command_mps": motion["lateral"] / max(motion["steps"], 1),
                "mean_command_change": motion["command_change"] / max(motion["steps"], 1),
                "time_within_0.5m_of_pedestrian": motion["near_people"] / max(motion["steps"], 1),
            },
            "per_scenario": {
                f"{k}/{b}": {
                    "episodes": int(sum(float(counts[o][i]) for o in outcomes)),
                    **{f"{o}_rate": float(counts[o][i]) / max(sum(float(counts[x][i]) for x in outcomes), 1) for o in outcomes},
                }
                for i, (k, b) in enumerate(cells)
            },
        }
        if maps:
            from g1_patrol.b2w import MAP
            from g1_patrol.heightmap import map_statistics

            metrics["map_stats"] = map_statistics(torch.cat(maps).numpy(), MAP["unknown_value"], MAP["resolution"])
        out = args_cli.out or os.path.join(os.path.dirname(resume_path), "eval",
                                           f"b2w_{controller.replace(':', '_').replace('/', '_')}{'_twin' if twin else ''}"
                                           f"{'_no_cameras' if camera_failure else ''}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] {episodes} episodes: " + ", ".join(f"{o} {metrics[o + '_rate']:.3f}" for o in outcomes))
        print(f"[INFO] metrics -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
