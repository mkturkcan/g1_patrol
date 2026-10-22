"""Tests for the B2-W robot-side code: the deployed observation matches training, the guard stops, the controller
model behaves (no simulator needed)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from b2w.navigator import Navigator, Pose  # noqa: E402

from g1_patrol.b2w import BASE_HEIGHT, CROP_BOX, LIDAR_MOUNT, MAP, deploy_config  # noqa: E402
from g1_patrol.controller_model import simulate  # noqa: E402
from g1_patrol.heightmap import HeightMap, drop_returns, rotate_inverse  # noqa: E402


def _quat_xyzw(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def _scene(rng):
    """World returns of a few walls, boxes and the floor, plus some on the robot's own body, around a pose."""
    pose = Pose(3.2, -1.4, BASE_HEIGHT + 0.05, 0.03, -0.05, 0.7)
    floor = np.c_[rng.uniform(-1, 7, 3000), rng.uniform(-5, 3, 3000), np.full(3000, 0.05)]
    wall = np.c_[np.full(800, 5.5), rng.uniform(-4, 1, 800), rng.uniform(0.05, 2.5, 800)]
    box = np.c_[rng.uniform(1.5, 2.0, 400), rng.uniform(-2.5, -2.0, 400), rng.uniform(0.05, 0.4, 400)]
    world = np.concatenate([floor, wall, box])
    body_local = np.c_[rng.uniform(-0.4, 0.4, 200), rng.uniform(-0.14, 0.14, 200), np.full(200, 0.07)]
    body = body_local @ pose.rotation().T + np.array([pose.x, pose.y, pose.z])
    return pose, np.concatenate([world, body])


def test_deployed_map_matches_training():
    """The navigator, fed the scan in the lidar frame and the odometry pose, builds the map the simulation builds
    from world-frame returns and the robot's state (no noise, no drift)."""
    rng = np.random.default_rng(0)
    pose, world = _scene(rng)
    cfg = deploy_config()
    nav = Navigator(cfg, policy=None)
    # the scan as the lidar reports it
    rot = pose.rotation()
    base = (world - np.array([pose.x, pose.y, pose.z])) @ rot
    lidar = base - np.array(LIDAR_MOUNT)
    deployed = nav.update_map(lidar, pose, 0.1)
    # the simulation's pipeline (perception.accumulated_height_map)
    q = torch.tensor(_quat_xyzw(pose.roll, pose.pitch, pose.yaw), dtype=torch.float32)[None]
    root = torch.tensor([[pose.x, pose.y, pose.z]], dtype=torch.float32)
    pts = torch.tensor(world, dtype=torch.float32)[None]
    local = rotate_inverse(q[:, None].expand(-1, pts.shape[1], -1), pts - root[:, None])
    ground = root[:, 2] - BASE_HEIGHT
    pts = drop_returns(pts, local, ground, CROP_BOX, MAP["drop_above"])
    sim_map = HeightMap(1, MAP["size"], MAP["resolution"])
    trained = sim_map.update(pts, root, torch.tensor([pose.yaw]), ground, 0.1, MAP["memory_s"], MAP["z_range"],
                             MAP["unknown_value"])[0].numpy()
    assert deployed.shape == (3, 80, 80)
    assert np.abs(deployed - trained).max() < 1e-4
    # the robot's own body is cropped, the box and the wall are there
    known = deployed[1] > MAP["unknown_value"] + 1e-6
    assert known.mean() > 0.2
    assert deployed[1].max() > 0.9  # the wall (clipped at 1 m)
    assert deployed[1][39:41, 39:41].max() <= MAP["unknown_value"] + 1e-6 or deployed[1][39:41, 39:41].max() < 0.1


def test_guard_stops_before_an_obstacle():
    cfg = deploy_config()
    nav = Navigator(cfg, policy=lambda obs, m: np.array([3.0, 0.0, 0.0]))  # full speed ahead
    nav.set_goal(10.0, 0.0)
    pose = Pose(0.0, 0.0, BASE_HEIGHT)
    rng = np.random.default_rng(1)
    obstacle = np.c_[rng.uniform(0.75, 0.85, 50), rng.uniform(-0.2, 0.2, 50), rng.uniform(0.05, 0.5, 50)]  # 0.2 m ahead
    floor = np.c_[rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500), np.zeros(500)]
    scan = np.concatenate([obstacle, floor]) - np.array([0, 0, BASE_HEIGHT]) - np.array(LIDAR_MOUNT)
    command, info = nav.step(scan, pose, np.zeros(3), 0.0)
    assert info["state"] == "guard stop" and command[0] == 0.0
    command, info = nav.step(floor - np.array([0, 0, BASE_HEIGHT]) - np.array(LIDAR_MOUNT), pose, np.zeros(3), 0.1)
    assert info["state"] == "driving" and command[0] > 1.0


def test_controller_model_settles_at_gain():
    cmd = torch.zeros(1, 200, 3)
    cmd[:, 10:, 0] = 1.0
    out = simulate(cmd, 0.02, torch.tensor([3]), torch.tensor([[0.9, 1.0, 1.0]]), torch.zeros(1, 3),
                   torch.tensor([[0.2, 0.2, 0.2]]), torch.tensor([[2.0, 2.0, 2.0]]), torch.zeros(1))
    assert out[0, 12, 0] == 0.0  # still within the latency
    assert abs(float(out[0, -1, 0]) - 0.9) < 1e-3  # settles at gain x command
    assert float(out[0, 14:, 0].diff().max()) <= 2.0 * 0.02 + 1e-6  # bounded acceleration
