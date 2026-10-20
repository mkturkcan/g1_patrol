"""Hands controller of the civic tasks with scripted triggers: walking up to litter or a trash bag and picking it up,
then putting it in a litter basket or at a curbside bag pile.

Every robot stands on a civic tile with nobody around. An item (a piece of litter, or with ``--bags`` a trash bag) is
placed on clear, level sidewalk 0.3-1.0 m in front of it, within 0.45 m to either side, and the trigger is pressed
once. The robots that picked up their item are then placed 1.0-1.6 m from a litter basket, facing it to within
35 degrees (with a bag: 0.5-1.0 m from the zone beside a bag pile, facing it to within 30 degrees), and the trigger
is pressed once more. The script reports the share of items picked up (also by distance and lateral offset), of
held items put where they belong, the times taken and the falls. The robot runs the reaching walker; no navigation
policy is involved.

Usage:
    python scripts/evaluate_hands.py --num_envs 512 [--bags] physics=isaacsim_physx
"""

import argparse
import json
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401

TASK = "G1Patrol-Navigate-Civic"
SECONDS = 12.0
"""Time (s) after each trigger."""


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--bags", action="store_true", help="Trash bags instead of litter.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Path of the metrics JSON.")
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]
    env_cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    nyc = env_cfg.commands.nyc
    nyc.attack_ratio = nyc.block_ratio = nyc.fallen_ratio = (0.0, 0.0)
    nyc.trash_count = (0, 0) if args.bags else (3, 3)
    nyc.bag_count = (1, 1) if args.bags else (0, 0)
    env_cfg.events.push_robot = None
    for name in ("pedestrian_collision", "obstacle_collision", "roadway", "goal_reached"):
        setattr(env_cfg.terminations, name, None)
    for name in ("pedestrian_contact", "obstacle_contact", "roadway", "success"):
        setattr(env_cfg.rewards, name, None)
    env_cfg.episode_length_s = 120.0
    env_cfg.curriculum.levels = None
    with launch_simulation(env_cfg, args):
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

        from g1_patrol.tasks.navigation.mdp.nyc_manipulation import CROUCH
        from g1_patrol.tasks.navigation.mdp.nyc_trash import ON_GROUND

        env = gym.make(TASK, cfg=env_cfg).unwrapped
        s = env.command_manager.get_term("nyc")
        hands = env.action_manager.get_term("hands")
        walker = env.action_manager.get_term("walker")
        s.layout.crowd.zero_()  # nobody around
        env.reset()
        robot, n, dev = s.robot, env.num_envs, env.device
        envs = torch.arange(n, device=dev)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        rand = lambda *shape: torch.rand(*shape, device=dev, generator=gen)  # noqa: E731
        # navigation actions: a zero velocity command (the forward action is tanh-squashed into the walker's range),
        # trigger and brace off
        act = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
        act[:, 0] = math.atanh(float(-walker._mid[0] / walker._half[0]))
        act[:, 3:] = -1.0
        steps = int(SECONDS / env.step_dt)

        def step(trigger: torch.Tensor | None = None):
            a = act.clone()
            if trigger is not None:
                a[trigger, 3] = 1.0
            env.step(a)

        def ground(p: torch.Tensor) -> torch.Tensor:  # (n, 2) tile-local points
            return s.layout.ground_height(s.tile_row, s.tile_col, p[:, None])[:, 0]

        def fallen() -> torch.Tensor:
            return robot.data.root_pos_w.torch[:, 2] - ground(s.robot_pos()) < 0.45

        def heading_vectors():
            yaw = s.robot_yaw()
            return torch.stack([torch.cos(yaw), torch.sin(yaw)], 1), torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)

        # 1. an item in front of each robot
        for _ in range(20):
            step()
        kind = s.is_bag[None] if args.bags else ~s.is_bag[None]
        on = (s.trash_state == ON_GROUND) & kind
        item = torch.where(on, rand(n, s.num_trash), -torch.ones(n, s.num_trash, device=dev)).argmax(dim=1)
        ahead, side = 0.3 + 0.7 * rand(n), (2 * rand(n) - 1) * 0.45
        fwd, lat = heading_vectors()
        spot = s.robot_pos() + ahead[:, None] * fwd + side[:, None] * lat
        ok = on.any(dim=1) & ((ground(spot) - ground(s.robot_pos())).abs() < 0.02) & ~fallen()
        for f in torch.linspace(0.0, 1.0, 6, device=dev):  # clear way, off the street
            mid = s.robot_pos() + f * (spot - s.robot_pos())
            ok &= (s.obstacle_clearance(mid[:, None], radius=0.3)[:, 0] > 0.0) & ~s._street_mask(mid[:, None])[:, 0]
        ids = envs[ok]
        pose = torch.zeros(len(ids), 1, 7, device=dev)
        pose[:, 0, :2] = spot[ids] + s.tile_center[ids]
        pose[:, 0, 2] = ground(spot)[ids] + s.trash_rest_height[item[ids]] + 0.005
        zero = torch.zeros(len(ids), device=dev)
        pose[:, 0, 3:] = quat_mul(quat_from_euler_xyz(zero, zero, 2 * math.pi * rand(len(ids))), s.trash_rest_quat[item[ids]])
        for k in item[ids].unique().tolist():
            sel = item[ids] == k
            s.trash.write_body_pose_to_sim_index(body_poses=pose[sel], body_ids=[k], env_ids=ids[sel])
            s.trash.write_body_velocity_to_sim_index(
                body_velocities=torch.zeros(int(sel.sum()), 1, 6, device=dev), body_ids=[k], env_ids=ids[sel]
            )
        for _ in range(2):
            step()
        placed = torch.zeros(n, dtype=torch.bool, device=dev)
        placed[ids] = True
        t_crouch = torch.full((n,), math.nan, device=dev)
        t_pick = torch.full((n,), math.nan, device=dev)
        for t in range(steps):
            step(placed if t == 0 else None)
            crouch = torch.isnan(t_crouch) & (hands.phase == CROUCH)
            t_crouch[crouch] = (t + 1) * env.step_dt
            got = torch.isnan(t_pick) & (s.held_item >= 0)
            t_pick[got] = (t + 1) * env.step_dt
        held = placed & (s.held_item >= 0)
        reached = placed & ~torch.isnan(t_crouch)
        picking = {
            "items": int(placed.sum()),
            "picked": float(held.sum() / placed.sum().clamp(min=1)),
            "reached_the_item": float(reached.sum() / placed.sum().clamp(min=1)),
            "grasped_when_reached": float((held & reached).sum() / reached.sum().clamp(min=1)),
            "median_time_to_reach_s": float(t_crouch[reached].median()) if reached.any() else None,
            "median_time_to_hold_s": float(t_pick[held].median()) if held.any() else None,
            "falls": int((placed & fallen()).sum()),
            "picked_by_distance": {},
            "picked_by_lateral_offset": {},
        }
        for lo, hi in ((0.3, 0.5), (0.5, 0.75), (0.75, 1.0)):
            m = placed & (ahead >= lo) & (ahead < hi)
            picking["picked_by_distance"][f"{lo:.2f}-{hi:.2f} m"] = float(held[m].sum() / m.sum().clamp(min=1))
        for lo, hi in ((0.0, 0.15), (0.15, 0.3), (0.3, 0.45)):
            m = placed & (side.abs() >= lo) & (side.abs() < hi)
            picking["picked_by_lateral_offset"][f"{lo:.2f}-{hi:.2f} m"] = float(held[m].sum() / m.sum().clamp(min=1))

        # 2. the held item to a basket or a bag pile
        ids = envs[held]
        k = len(ids)
        if args.bags:  # beside the bag zone, on the sidewalk side, facing it
            rect = s.piles[ids, 0]
            cx, cy = (rect[:, 0] + rect[:, 1]) / 2, (rect[:, 2] + rect[:, 3]) / 2
            edge = torch.where(cy > 0, rect[:, 2], rect[:, 3])  # the zone's edge away from the curb
            target = torch.stack([cx + (rand(k) - 0.5) * 0.6, edge - torch.sign(cy) * (0.5 + 0.5 * rand(k))], 1)
            yaw = torch.atan2(cy - target[:, 1], cx - target[:, 0]) + (rand(k) - 0.5) * 1.0
        else:  # around a basket, facing it
            b = s.baskets[ids, 0]
            ang = 2 * math.pi * rand(k)
            target = b[:, :2] + (1.0 + 0.6 * rand(k))[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
            yaw = ang + math.pi + (rand(k) - 0.5) * 1.2
        spot = s.robot_pos().clone()
        spot[ids] = target
        clear = (s.obstacle_clearance(spot[:, None], radius=0.3)[:, 0] > 0.0) & ~s._street_mask(spot[:, None])[:, 0]
        ids, target, yaw = ids[clear[ids]], target[clear[ids]], yaw[clear[ids]]
        pose = torch.zeros(len(ids), 7, device=dev)
        pose[:, :2] = target + s.tile_center[ids]
        pose[:, 2] = s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], target[:, None])[:, 0] + 0.78
        zero = torch.zeros(len(ids), device=dev)
        pose[:, 3:] = quat_from_euler_xyz(zero, zero, yaw)
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=dev), env_ids=ids)
        for _ in range(15):
            step()
        carrying = torch.zeros(n, dtype=torch.bool, device=dev)
        carrying[ids] = True
        carrying &= s.held_item >= 0
        done = torch.zeros(n, device=dev)
        t_done = torch.full((n,), math.nan, device=dev)
        for t in range(steps):
            step(carrying if t == 0 else None)
            done += s.bags_delivered if args.bags else s.trash_deposited
            first = torch.isnan(t_done) & (done > 0)
            t_done[first] = (t + 1) * env.step_dt
        delivered = carrying & (done > 0)
        putting = {
            "robots": int(carrying.sum()),
            "put_where_it_belongs": float(delivered.sum() / carrying.sum().clamp(min=1)),
            "let_go_elsewhere": float((carrying & (s.held_item < 0) & ~delivered).sum() / carrying.sum().clamp(min=1)),
            "median_time_s": float(t_done[delivered].median()) if delivered.any() else None,
            "falls": int((carrying & fallen()).sum()),
        }
        metrics = {"items": "bags" if args.bags else "litter", "picking_up": picking, "putting_down": putting}
        print("[RESULT] Hands controller:")
        print(json.dumps(metrics, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(metrics, f, indent=2)
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
