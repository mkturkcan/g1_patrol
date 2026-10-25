"""Write a simulated tile (structures, routes) as a JSON world for ``fake_b2w.py`` (run where the training code
imports, e.g. the Isaac Lab environment; the fake robot itself needs only numpy and ROS 2).

Usage:
    python deploy/b2w/sim/export_world.py --kind makerlab --difficulty 0.6 --seed 3 --out deploy/b2w/sim/lab_world.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", default="makerlab")
    parser.add_argument("--difficulty", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from g1_patrol.b2w import FOOTPRINT
    from g1_patrol.tasks.navigation.nyc.kinds import ALL_BUILDERS
    from g1_patrol.tasks.navigation.nyc.routing import plan_routes

    lay = ALL_BUILDERS[args.kind](args.difficulty, np.random.default_rng(args.seed), None)
    keep = plan_routes(lay, FOOTPRINT[1])
    world = {
        "source": f"{args.kind} d={args.difficulty} seed={args.seed}",
        "size": [16.0, 10.0],
        "boxes": [[float(v) for v in b] for b in lay.boxes],
        "circles": [[float(v) for v in c] for c in lay.circles],
        "routes": [[float(v) for v in lay.routes[k]] for k in keep],
        "people_paths": [[[float(x), float(y)] for x, y in pts] for pts, *_ in lay.paths],
    }
    with open(args.out, "w") as f:
        json.dump(world, f, indent=1)
    print(f"{args.out}: {len(world['boxes'])} boxes, {len(world['circles'])} circles, {len(world['routes'])} routes")


if __name__ == "__main__":
    main()
