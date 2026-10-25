"""Tests for the B2-W robot-side code: the deployed observation matches training, the guard stops, the controller
model behaves (no simulator needed)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from b2w.navigator import Navigator, Pose, depth_image_points  # noqa: E402

from g1_patrol.b2w import BASE_HEIGHT, CROP_BOX, D435I, DEPTH_CAMERAS, LIDAR_MOUNT, MAP, deploy_config, rpy_matrix  # noqa: E402
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


def test_shield_stops_before_an_obstacle():
    """The policy asks for full speed at a box 0.2 m ahead: the shield stops the robot; with the way clear it drives."""
    cfg = deploy_config()
    nav = Navigator(cfg, policy=lambda obs, m: np.array([3.0, 0.0, 0.0]))  # full speed ahead
    nav.set_goal(10.0, 0.0)
    pose = Pose(0.0, 0.0, BASE_HEIGHT)
    rng = np.random.default_rng(1)
    obstacle = np.c_[rng.uniform(0.75, 0.85, 50), rng.uniform(-0.2, 0.2, 50), rng.uniform(0.05, 0.5, 50)]  # 0.2 m ahead
    floor = np.c_[rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500), np.zeros(500)]
    scan = np.concatenate([obstacle, floor]) - np.array([0, 0, BASE_HEIGHT]) - np.array(LIDAR_MOUNT)
    command, info = nav.step(scan, pose, np.zeros(3), 0.0)
    assert info["state"] == "shield" and command[0] < 0.2  # a creep at most
    nav = Navigator(cfg, policy=lambda obs, m: np.array([3.0, 0.0, 0.0]))
    nav.set_goal(10.0, 0.0)
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


def test_deployed_fused_map_matches_training():
    """With depth cameras: returns in each camera's optical frame, placed with the camera's mount, give the map the
    simulation builds from the same world-frame returns (within the cameras' range)."""
    rng = np.random.default_rng(2)
    pose, world = _scene(rng)
    nav = Navigator(deploy_config(depth=True), policy=None)
    rot, origin = pose.rotation(), np.array([pose.x, pose.y, pose.z])
    base = (world - origin) @ rot
    lidar = base - np.array(LIDAR_MOUNT)
    depth, sim_parts = {}, [world]
    for name, (pos, rpy) in DEPTH_CAMERAS.items():
        r_cam = np.array(rpy_matrix(rpy))
        optical = (base - np.array(pos)) @ r_cam  # every return, seen from the camera
        dist = np.linalg.norm(optical, axis=1)
        keep = (optical[:, 2] > 0) & (dist >= D435I["min_range"]) & (dist < D435I["max_range"])
        depth[name] = optical[keep]
        sim_parts.append(world[keep])
    deployed = nav.update_map(lidar, pose, 0.1, depth)
    q = torch.tensor(_quat_xyzw(pose.roll, pose.pitch, pose.yaw), dtype=torch.float32)[None]
    root = torch.tensor([[pose.x, pose.y, pose.z]], dtype=torch.float32)
    pts = torch.tensor(np.concatenate(sim_parts), dtype=torch.float32)[None]
    local = rotate_inverse(q[:, None].expand(-1, pts.shape[1], -1), pts - root[:, None])
    ground = root[:, 2] - BASE_HEIGHT
    pts = drop_returns(pts, local, ground, CROP_BOX, MAP["drop_above"])
    trained = HeightMap(1, MAP["size"], MAP["resolution"]).update(
        pts, root, torch.tensor([pose.yaw]), ground, 0.1, MAP["memory_s"], MAP["z_range"], MAP["unknown_value"])[0].numpy()
    assert np.abs(deployed - trained).max() < 1e-4


def test_depth_image_back_projection():
    """A flat floor 0.6 m below a camera looking 45 degrees down comes back at the floor's height."""
    fx = fy = 400.0
    cx, cy, w, h = 424.0, 240.0, 848, 480
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    # optical ray directions, rotated 45 deg down about the camera's x axis: the floor is at y_world = 0.6 below
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    down = d[..., 1] * c + d[..., 2] * s  # component along gravity (down)
    z = np.where(down > 0, 0.6 / np.maximum(down, 1e-9), 0.0)  # depth (z) at which the ray meets the floor
    depth = np.round(np.where(z < 10, z, 0) * 1000).astype(np.uint16)
    pts = depth_image_points(depth, fx, fy, cx, cy, D435I["width"], D435I["height"])
    assert len(pts) > 0.8 * D435I["width"] * D435I["height"]
    height = pts[:, 1] * c + pts[:, 2] * s
    assert np.abs(height - 0.6).max() < 0.01


def test_floor_is_measured_not_assumed():
    """The robot stands 0.58 m high, not the nominal height: the navigator measures the floor, and its map reads the
    floor at 0 and a 0.3 m box at 0.3 m, despite low platforms and a drop in view."""
    from g1_patrol.floor import FloorEstimator

    rng = np.random.default_rng(3)
    height = 0.58
    pose = Pose(0.0, 0.0, 10.0 + height, 0.02, -0.03, 0.4)
    floor = np.c_[rng.uniform(-6, 6, 6000), rng.uniform(-6, 6, 6000), np.full(6000, 10.0)]
    pallet = np.c_[rng.uniform(1.5, 3.0, 1500), rng.uniform(-1.0, 1.0, 1500), np.full(1500, 10.15)]  # low platform
    dock = np.c_[rng.uniform(-3.0, -2.0, 300), rng.uniform(-3, 3, 300), np.full(300, 8.8)]  # a drop of 1.2 m
    box = np.c_[rng.uniform(1.0, 1.3, 300), rng.uniform(1.5, 1.8, 300), rng.uniform(10.05, 10.3, 300)]
    world = np.concatenate([floor, pallet, dock, box])
    est = FloorEstimator(prior=-BASE_HEIGHT)
    assert abs(est.measure(world - np.array([pose.x, pose.y, pose.z])) + height) < 0.01
    nav = Navigator(deploy_config(), policy=None)
    lidar = (world - np.array([pose.x, pose.y, pose.z])) @ pose.rotation() - np.array(LIDAR_MOUNT)
    for k in range(20):  # the running estimate converges from the prior
        m = nav.update_map(lidar, pose, 0.1)
    assert abs(nav.floor_z + height) < 0.01
    c = MAP["size"] / MAP["resolution"] / 2
    yy, xx = np.meshgrid((np.arange(int(2 * c)) + 0.5 - c) * MAP["resolution"], (np.arange(int(2 * c)) + 0.5 - c) * MAP["resolution"], indexing="ij")
    # cells around (-2, -2) in the heading frame are floor (no platform, no box there)
    ring = (np.hypot(xx, yy) > 1.0) & (np.hypot(xx, yy) < 1.6)
    known = m[0] > MAP["unknown_value"] + 1e-6
    assert np.median(m[0][ring & known]) == pytest.approx(0.0, abs=0.02)
    assert m[0].max() == pytest.approx(0.3, abs=0.02)  # the box
