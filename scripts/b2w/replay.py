"""Replay a recorded drive of the B2-W through the navigation stack, offline.

The deployment node records every lidar scan with the odometry pose, the velocity, the goal and the command it gave
(``record:=drive.npz``). This script feeds the recording to the same :class:`b2w.navigator.Navigator` and

* checks that it reproduces the recorded commands (the node and the offline code agree);
* saves the height maps the policy saw (``maps.png``: accumulated / current / previous scan, every ``--every`` scans);
* reports statistics to compare with simulation (``scripts/b2w/evaluate.py --map_stats``): the share of known cells,
  near and far, and the height of the floor around the robot -- a floor reading away from 0 means that
  ``BASE_HEIGHT`` or the lidar mount in ``deploy.json`` differ from the robot.

Usage:
    PYTHONPATH=src:deploy python scripts/b2w/replay.py --record drive.npz --config deploy.json --policy policy.onnx --out replay/
"""

import argparse
from pathlib import Path

import numpy as np

from b2w.navigator import Navigator, OnnxPolicy, Pose

from g1_patrol.heightmap import map_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", required=True)
    parser.add_argument("--config", default="deploy.json")
    parser.add_argument("--policy", default="policy.onnx")
    parser.add_argument("--out", default="replay")
    parser.add_argument("--every", type=int, default=20)
    args = parser.parse_args()
    rec = np.load(args.record, allow_pickle=True)
    nav = Navigator(args.config, OnnxPolicy(args.policy), guard=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps, diffs, commands = [], [], []
    for k in range(len(rec["stamp"])):
        goal = rec["goal"][k]
        if np.isfinite(goal).all():
            nav.set_goal(*goal)
        pose = Pose(*rec["pose"][k])
        command, info = nav.step(rec["points"][k], pose, rec["velocity"][k], float(rec["stamp"][k]))
        maps.append(info["map"])
        commands.append(command)
        diffs.append(np.abs(command - rec["command"][k]).max())
    maps = np.stack(maps)
    np.savetxt(out / "commands.csv", np.c_[rec["stamp"], np.stack(commands)], delimiter=",", header="stamp,vx,vy,wz", comments="")
    print(f"{len(maps)} scans; max |command - recorded| {max(diffs):.2e}")
    stats = map_statistics(maps, nav.cfg["map"]["unknown_value"], nav.cfg["map"]["resolution"])
    for k, v in stats.items():
        print(f"{k:28s} {v:.3f}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = list(range(0, len(maps), args.every))[:12]
        fig, axes = plt.subplots(len(frames), 3, figsize=(9, 3 * len(frames)), squeeze=False)
        for row, f in enumerate(frames):
            for c, title in enumerate(("accumulated", "current scan", "previous scan")):
                img = np.where(maps[f, c] <= nav.cfg["map"]["unknown_value"] + 1e-6, np.nan, maps[f, c])
                axes[row, c].imshow(img[:, ::-1], origin="lower", cmap="viridis", vmin=-0.6, vmax=1.0, extent=(4, -4, -4, 4))
                axes[row, c].set_title(f"scan {f}: {title}", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "maps.png", dpi=70)
        print("maps ->", out / "maps.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
