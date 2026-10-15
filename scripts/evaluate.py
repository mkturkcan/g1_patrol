"""Evaluate a trained RSL-RL locomotion policy headlessly.

Runs the policy in many parallel environments (play-mode config: no observation noise or pushes) and reports how
well the robots track their velocity commands and how often they fall. It also saves the link trajectory of one
robot so it can be rendered offline with ``scripts/render_trajectory.py``. Pass ``--video`` to also record an RTX
video of robot 0 (requires a Kit-based physics backend such as ``physics=isaacsim_physx``).

Outputs are written next to the checkpoint, in ``<run_dir>/eval/``.

Usage:
    python scripts/evaluate.py --task G1Patrol-Walk-Flat --checkpoint latest
    python scripts/evaluate.py --task G1Patrol-Walk-Flat --checkpoint logs/rsl_rl/g1_walk_flat/<run> physics=isaacsim_physx
    python scripts/evaluate.py --task G1Patrol-Walk-Street --checkpoint latest --push_velocity 1.0   # shove test
"""

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.managers import EventTermCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    set_hydra_args,
)
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an RSL-RL locomotion policy.")
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="RL agent config entry point.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of policy steps to run (50 steps = 1 s).")
    parser.add_argument("--traj_env", type=int, default=0, help="Environment whose link trajectory is saved.")
    parser.add_argument(
        "--push_velocity",
        type=float,
        default=0.0,
        help="Shove test: kick robots with random planar velocities of up to this many m/s (0 = no shoves).",
    )
    parser.add_argument(
        "--push_interval", type=float, nargs=2, default=(4.0, 8.0), help="Seconds between shoves (min max)."
    )
    parser.add_argument(
        "--full_terrain",
        action="store_true",
        help="Evaluate on as many random terrain tiles as in training (play mode keeps only 5 x 5).",
    )
    parser.add_argument(
        "--nav_commands",
        action="store_true",
        help="Draw commands from the navigation policy's ranges (forward -0.4 to 1.0 m/s, lateral +-0.15 m/s, "
        "yaw rate +-1.0 rad/s) instead of the walker's training ranges.",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    if args_cli.num_envs is None:
        args_cli.num_envs = 256
    enable_cameras_for_video(args_cli)
    set_hydra_args(remaining_args)
    return args_cli


def resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg, log_root_path: str) -> str:
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        # a run directory: take its latest checkpoint
        run_dir = os.path.abspath(args_cli.checkpoint)
        return get_checkpoint_path(os.path.dirname(run_dir), os.path.basename(run_dir), r"model_.*\.pt")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def link_parent_indices(scene, robot) -> list[int]:
    """Parent link index of every body of ``robot`` (-1 for the root), read from the joints in the USD stage."""
    from pxr import Usd, UsdPhysics

    root_path = robot.cfg.prim_path.replace(scene.env_regex_ns, scene.env_prim_paths[0])
    body_index = {name: i for i, name in enumerate(robot.body_names)}
    parents = [-1] * len(robot.body_names)
    for prim in Usd.PrimRange(scene.stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if body0 and body1 and body0[0].name in body_index and body1[0].name in body_index:
            parents[body_index[body1[0].name]] = body_index[body0[0].name]
    return parents


def main(argv: list[str]) -> None:
    args_cli = parse_args(argv)
    installed_version = check_rsl_rl_version()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    generator = env_cfg.scene.terrain.terrain_generator
    if args_cli.full_terrain and generator is not None:
        train_generator = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)[0].scene.terrain.terrain_generator
        generator.num_rows, generator.num_cols = train_generator.num_rows, train_generator.num_cols
    if args_cli.nav_commands:
        ranges = env_cfg.commands.base_velocity.ranges
        ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (-0.4, 1.0), (-0.15, 0.15), (-1.0, 1.0)
    if args_cli.push_velocity > 0:
        v = args_cli.push_velocity
        env_cfg.events.push_robot = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(args_cli.push_interval),
            params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
        )
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
        robot = base_env.scene["robot"]
        term_manager = base_env.termination_manager
        link_names = list(robot.body_names)
        parent_indices = link_parent_indices(base_env.scene, robot)
        k = args_cli.traj_env
        device = base_env.device

        # posture probes
        torso_id = link_names.index("torso_link")
        knee_ids, _ = robot.find_joints(".*_knee_joint")
        hip_roll_ids, _ = robot.find_joints(".*_hip_roll_joint")
        scanner = base_env.scene.sensors.get("height_scanner")
        gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(base_env.num_envs, 3)
        posture = {"torso_tilt_deg": 0.0, "pelvis_height_m": 0.0, "knee_bend_rad": 0.0, "hip_roll_abs_rad": 0.0}
        # smoothness probes: leg power, change of joint targets, torso roll/pitch rate, peak foot force / body weight
        leg_ids, _ = robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        contact = base_env.scene.sensors.get("contact_forces")
        feet_ids = contact.find_bodies(".*_ankle_roll_link")[0] if contact is not None else []
        weight = float(robot.data.default_mass.torch[0].sum()) * 9.81
        smooth = {"leg_power_w": 0.0, "joint_target_change_rad": 0.0, "torso_roll_pitch_rate_radps": 0.0, "peak_foot_force_bw": 0.0}
        action_scale = float(base_env.cfg.actions.joint_pos.scale)
        prev_action = None

        # accumulators
        lin_err_sum = torch.zeros(2, device=device)
        ang_err_sum = torch.zeros((), device=device)
        cmd_speed_sum = torch.zeros((), device=device)
        act_speed_sum = torch.zeros((), device=device)
        num_samples = 0
        num_falls = 0
        num_timeouts = 0
        traj = {"body_pos": [], "root_pos": [], "root_yaw": [], "command": [], "lin_vel_yaw": [], "ang_vel_z": []}
        traj_falls = []

        obs = env.get_observations()
        with torch.inference_mode():
            for step in range(args_cli.num_steps):
                # command that the action is computed from
                command = base_env.command_manager.get_command("base_velocity").clone()
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
                policy.reset(dones)

                # tracking error in the heading (yaw) frame, skipping envs that were just reset
                alive = ~dones.bool()
                root_quat_w = robot.data.root_quat_w.torch  # (x, y, z, w)
                lin_vel_yaw = math_utils.quat_apply_inverse(
                    math_utils.yaw_quat(root_quat_w), robot.data.root_lin_vel_w.torch
                )[:, :2]
                ang_vel_z = robot.data.root_ang_vel_w.torch[:, 2]
                lin_err_sum += (lin_vel_yaw - command[:, :2]).abs()[alive].sum(dim=0)
                ang_err_sum += (ang_vel_z - command[:, 2]).abs()[alive].sum()
                cmd_speed_sum += command[:, :2].norm(dim=1)[alive].sum()
                act_speed_sum += lin_vel_yaw.norm(dim=1)[alive].sum()
                num_samples += int(alive.sum())

                g_torso = math_utils.quat_apply_inverse(robot.data.body_quat_w.torch[:, torso_id], gravity)
                tilt = torch.rad2deg(torch.acos((-g_torso[:, 2]).clamp(-1.0, 1.0)))
                ground = 0.0
                if scanner is not None:
                    hz = scanner.data.ray_hits_w.torch[..., 2]
                    hz = torch.where(torch.isfinite(hz), hz, torch.nan)
                    ground = torch.nanmean(hz, dim=1)
                height = robot.data.root_pos_w.torch[:, 2] - ground
                joint_pos = robot.data.joint_pos.torch
                posture["torso_tilt_deg"] += float(tilt[alive].sum())
                posture["pelvis_height_m"] += float(height[alive].sum())
                posture["knee_bend_rad"] += float(joint_pos[:, knee_ids].mean(dim=1)[alive].sum())
                posture["hip_roll_abs_rad"] += float(joint_pos[:, hip_roll_ids].abs().mean(dim=1)[alive].sum())
                power = (robot.data.applied_torque.torch[:, leg_ids] * robot.data.joint_vel.torch[:, leg_ids]).abs().sum(dim=1)
                smooth["leg_power_w"] += float(power[alive].sum())
                if prev_action is not None:
                    rate = (action - prev_action).abs().mean(dim=1) * action_scale  # joint targets, rad per step
                    smooth["joint_target_change_rad"] += float(rate[alive].sum())
                prev_action = action.clone()
                w_torso = math_utils.quat_apply_inverse(robot.data.body_quat_w.torch[:, torso_id], robot.data.body_ang_vel_w.torch[:, torso_id])
                smooth["torso_roll_pitch_rate_radps"] += float(w_torso[:, :2].norm(dim=1)[alive].sum())
                if contact is not None:
                    foot_force = contact.data.net_forces_w.torch[:, feet_ids, 2].abs().amax(dim=1)
                    smooth["peak_foot_force_bw"] += float((foot_force / weight)[alive].sum())

                falls = term_manager.get_term("base_contact")
                num_falls += int(falls.sum())
                num_timeouts += int(term_manager.time_outs.sum())

                # trajectory of one robot (positions relative to its env origin)
                origin = base_env.scene.env_origins[k]
                _, _, yaw = math_utils.euler_xyz_from_quat(root_quat_w[k : k + 1])
                traj["body_pos"].append((robot.data.body_pos_w.torch[k] - origin).cpu().numpy())
                traj["root_pos"].append((robot.data.root_pos_w.torch[k] - origin).cpu().numpy())
                traj["root_yaw"].append(float(yaw[0]))
                traj["command"].append(command[k].cpu().numpy())
                traj["lin_vel_yaw"].append(lin_vel_yaw[k].cpu().numpy())
                traj["ang_vel_z"].append(float(ang_vel_z[k]))
                if bool(falls[k]):
                    traj_falls.append(step)

        sim_seconds = args_cli.num_steps * base_env.step_dt
        robot_minutes = args_cli.num_envs * sim_seconds / 60.0
        metrics = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "physics": type(base_env.cfg.sim.physics).__name__,
            "num_envs": args_cli.num_envs,
            "sim_seconds_per_env": sim_seconds,
            "push_velocity": args_cli.push_velocity,
            "full_terrain": args_cli.full_terrain,
            "nav_commands": args_cli.nav_commands,
            "mean_abs_err_lin_vel_x": float(lin_err_sum[0] / num_samples),
            "mean_abs_err_lin_vel_y": float(lin_err_sum[1] / num_samples),
            "mean_abs_err_ang_vel_z": float(ang_err_sum / num_samples),
            "mean_commanded_speed": float(cmd_speed_sum / num_samples),
            "mean_actual_speed": float(act_speed_sum / num_samples),
            "falls": num_falls,
            "timeouts": num_timeouts,
            "falls_per_robot_minute": num_falls / robot_minutes,
            **{f"mean_{k}": v / num_samples for k, v in posture.items()},
            **{f"mean_{k}": v / num_samples for k, v in smooth.items()},
        }
        print("\n[RESULT] Evaluation metrics:")
        print(json.dumps(metrics, indent=2))
        suffix = f"_push{args_cli.push_velocity:g}" if args_cli.push_velocity > 0 else ""
        suffix += ("_fullterrain" if args_cli.full_terrain else "") + ("_navcmd" if args_cli.nav_commands else "")
        with open(os.path.join(out_dir, f"metrics{suffix}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        np.savez_compressed(
            os.path.join(out_dir, f"trajectory_env{k}.npz"),
            **{key: np.stack(val) for key, val in traj.items()},
            link_names=np.array(link_names),
            parent_indices=np.array(parent_indices),
            falls=np.array(traj_falls, dtype=np.int64),
            dt=base_env.step_dt,
        )
        print(f"[INFO] Saved metrics and trajectory to: {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
