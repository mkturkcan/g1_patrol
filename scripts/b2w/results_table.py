"""Markdown tables of the B2-W release evaluations (``scripts/reproduce/b2w_3_evaluate.sh``), for the READMEs.

Prints the tables; with ``--write``, replaces them between ``<!-- table:b2w -->`` / ``<!-- table:b2w_stress -->``
markers in ``README.md`` and ``deploy/b2w/README.md``.

Usage:
    python scripts/b2w/results_table.py [--logs logs/rsl_rl] [--write]
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    ("Lab teacher (privileged)", "teacher", "b2w_nominal.json"),
    ("Fused student (lidar + 2 depth cameras)", "fused", "b2w_nominal.json"),
    ("Lidar-only student", "lidar", "b2w_nominal.json"),
    ("Fused, shield", "fused", "b2w_nominal_shield.json"),
    ("Fused, shield, cap 0.5 m/s", "fused", "b2w_nominal_shield_limit0.5.json"),
    ("Fused, shield, cap 1.0 m/s", "fused", "b2w_nominal_shield_limit1.0.json"),
]
STRESS = [
    ("Nominal controller", "b2w_nominal.json"),
    ("Laggy controller (0.16-0.25 s latency)", "b2w_laggy.json"),
    ("Sluggish controller (2x response time, 1/2 acceleration)", "b2w_sluggish.json"),
    ("Weak controller (70-80 % of the command)", "b2w_weak.json"),
    ("Drifting controller (2x yaw drift)", "b2w_drifty.json"),
    ("Depth cameras failed", "b2w_nominal_no_cameras.json"),
    ("Degraded lidar map (20 % dropout, odometry drift)", "b2w_degraded_map.json"),
    ("Laggy controller, shield", "b2w_laggy_shield.json"),
]


def _lab(d: dict, key: str) -> float:
    cells = [d["per_scenario"][f"makerlab/{b}"] for b in ("easy", "medium", "hard")]
    n = sum(c["episodes"] for c in cells)
    return sum(c[key] * c["episodes"] for c in cells) / max(n, 1)


def _row(name: str, d: dict) -> str:
    pc = d["pedestrian_contacts"]
    shield = d.get("shield_stats")
    stops = f"{100 * shield['emergency_rate']:.1f}" if shield else "–"
    return (f"| {name} | {d['episodes']:,} | {100 * d['goal_reached_rate']:.1f} | {100 * _lab(d, 'goal_reached_rate'):.1f} | "
            f"{100 * d['obstacle_collision_rate']:.1f} | {100 * pc['robot_moving_rate']:.1f} | "
            f"{100 * pc['pedestrian_walked_in_rate']:.1f} | {100 * d['time_out_rate']:.1f} | {d['motion']['mean_speed_mps']:.2f} | {stops} |")


HEADER = ("| Policy | Episodes | Goal (%) | Goal, lab hall (%) | Structure contact (%) | Pedestrian contact, robot "
          "moving (%) | Pedestrian walked into stopped robot (%) | Time-out (%) | Speed (m/s) | Shield stops (% of steps) |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default="logs/rsl_rl")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.logs)
    dirs = {"teacher": root / "b2w_nav_lab_teacher/release/eval", "fused": root / "b2w_nav_lab_fused_student/release/eval",
            "lidar": root / "b2w_nav_lab_lidar_student/release/eval"}
    main_rows = [_row(name, json.loads((dirs[who] / f).read_text())) for name, who, f in ROWS if (dirs[who] / f).exists()]
    stress_rows = [_row(name, json.loads((dirs["fused"] / f).read_text())) for name, f in STRESS
                   if (dirs["fused"] / f).exists()]
    tables = {"b2w": "\n".join([HEADER, *main_rows]),
              "b2w_stress": "\n".join([HEADER.replace("| Policy |", "| Fused student |"), *stress_rows])}
    print(tables["b2w"] + "\n\n" + tables["b2w_stress"])
    if args.write:
        for path in (ROOT / "README.md", ROOT / "deploy" / "b2w" / "README.md"):
            text = path.read_text()
            for name, table in tables.items():
                text = re.sub(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)",
                              lambda m: m.group(1) + table + m.group(2), text, flags=re.S)
            path.write_text(text)
            print("wrote", path)


if __name__ == "__main__":
    main()
