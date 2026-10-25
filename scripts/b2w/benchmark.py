"""Check and time the B2-W navigation stack on the robot's computer (Jetson Orin NX).

``--check`` runs the exported policy on the inputs saved by ``scripts/b2w/export.py`` (``samples.npz``) and compares
its actions with the ones computed in simulation: a mismatch means a broken inference stack (provider, precision).
The timing runs the full step -- height map from a synthetic scan of the lidar's size, observation, policy -- and
reports latency; one step per scan must fit well within the 100 ms scan period.

Usage (on the robot):
    PYTHONPATH=src:deploy python3 scripts/b2w/benchmark.py --config deploy.json --policy policy.onnx \\
        --check samples.npz --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from b2w.navigator import Navigator, OnnxPolicy, Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="deploy.json")
    parser.add_argument("--policy", default="policy.onnx")
    parser.add_argument("--check", default=None, help="samples.npz from export.py")
    parser.add_argument("--device", default="cpu", help="torch device of the height map (cpu or cuda)")
    parser.add_argument("--providers", default=None, help="comma-separated onnxruntime providers (default: all available)")
    parser.add_argument("--points", type=int, default=57600, help="returns per scan (Helios 32 lines x 1800 at 10 Hz)")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    providers = args.providers.split(",") if args.providers else None
    policy = OnnxPolicy(args.policy, providers)
    print("onnxruntime providers:", policy.session.get_providers())
    if args.check:
        s = np.load(args.check)
        out = np.stack([policy(o, m) for o, m in zip(s["obs"], s["lidar_map"])])
        diff = np.abs(out - s["actions"]).max()
        print(f"check: {len(out)} samples, max |action difference| {diff:.2e} -> {'ok' if diff < 1e-2 else 'MISMATCH'}")
    nav = Navigator(args.config, policy, device=args.device)
    nav.set_goal(50.0, 0.0)
    rng = np.random.default_rng(0)
    times = {"map": [], "policy": [], "shield": [], "step": []}
    for k in range(args.steps):
        # a synthetic scan: floor around the robot and a few walls, in the lidar frame
        az = rng.uniform(-np.pi, np.pi, args.points)
        el = rng.uniform(np.radians(-55), np.radians(15), args.points)
        r = np.where(el < 0, np.minimum(0.73 / np.tan(-el + 1e-3), 12.0), rng.uniform(2.0, 12.0, args.points))
        scan = np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]
        pose = Pose(0.1 * k, 0.0, 0.55, 0.0, 0.0, 0.0)
        depth = {name: np.c_[rng.uniform(-1, 1, (2560, 2)), rng.uniform(0.4, 4.0, 2560)] for name in nav.cameras}
        t0 = time.perf_counter()
        height_map = nav.update_map(scan, pose, 0.1, depth)
        t1 = time.perf_counter()
        action = policy(nav.proprio(pose, np.zeros(3)), height_map)
        t2 = time.perf_counter()
        nav.shielded(nav.to_command(action), np.zeros(3), height_map)
        t3 = time.perf_counter()
        times["map"].append(t1 - t0), times["policy"].append(t2 - t1), times["shield"].append(t3 - t2)
        times["step"].append(t3 - t0)
    for k, v in times.items():
        v = np.array(v[10:]) * 1000
        print(f"{k:6s} mean {v.mean():6.1f} ms  p99 {np.percentile(v, 99):6.1f} ms  max {v.max():6.1f} ms")
    step = np.percentile(np.array(times["step"][10:]) * 1000, 99)
    print("fits a 10 Hz lidar" if step < 50 else "TOO SLOW for one step per scan at 10 Hz (p99 >= 50 ms)")


if __name__ == "__main__":
    main()
