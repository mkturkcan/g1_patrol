"""Tables of the B2-W release evaluations (``scripts/reproduce/b2w_3_evaluate.sh``), for the READMEs and the project page.

Prints the tables; with ``--write``, replaces them between ``<!-- table:NAME -->`` / ``<!-- /table:NAME -->``
markers in ``README.md`` and ``deploy/b2w/README.md`` and in the ``<table id="NAMETable">`` elements of
``docs/b2w/index.html`` (``b2w``, ``b2w_stress``, ``b2w_shield``, ``b2w_civil``, ``b2w_camera``, ``b2w_people``).

Usage:
    python scripts/b2w/results_table.py [--logs logs/rsl_rl] [--tag release] [--write]
"""

import argparse
import html
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
SHIELD = [
    ("No shield", "b2w_nominal.json"),
    ("Shield alone: commands scaled down", "b2w_nominal_shield_plain.json"),
    ("+ alternatives from rest, backing off (crawl:=false)", "b2w_nominal_shield_nocrawl.json"),
    ("+ crawling near what moved (released default)", "b2w_nominal_shield.json"),
]
COLUMNS = ["Episodes", "Goal (%)", "Goal, lab hall (%)", "Structure contact (%)", "Pedestrian contact, robot moving (%)",
           "Pedestrian walked into stopped robot (%)", "Time-out (%)", "Speed (m/s)", "Shield stops (% of steps)"]
CIVIL = [  # (name, run, policy, evaluation file): the mixture in which the civil engineering lab takes half the lab tiles
    ("Released fused student", "release", "fused", "b2w_civil_mix.json"),
    ("Released fused student, shield", "release", "fused", "b2w_civil_mix_shield.json"),
    ("Lab teacher trained with the civil lab (privileged)", "civil", "teacher", "b2w_nominal.json"),
    ("Fused student trained with the civil lab", "civil", "fused", "b2w_nominal.json"),
    ("Fused student trained with the civil lab, shield", "civil", "fused", "b2w_nominal_shield.json"),
]
CAMERA = [  # (name, policy, evaluation file): 256 robots for 150 s (scripts/reproduce/b2w_4_camera.sh)
    ("Fused student (released)", "fused", "b2w_nominal_256.json"),
    ("Camera student (+ front color camera)", "camera", "b2w_nominal.json"),
    ("Fused student, shield", "fused", "b2w_nominal_shield_256.json"),
    ("Camera student, shield", "camera", "b2w_nominal_shield.json"),
]
PEOPLE = [  # (name, run, policy, shield): pedestrians seen as capsules or posed characters (b2w_6_people.sh)
    ("Fused, released", "release", "fused", False),
    ("Fused, fine-tuned", "people", "fused", False),
    ("Lidar-only, released", "release", "lidar", False),
    ("Lidar-only, fine-tuned", "people", "lidar", False),
    ("Fused + shield, released", "release", "fused", True),
    ("Fused + shield, fine-tuned", "people", "fused", True),
]
PEOPLE_SEEN = (("capsules", ""), ("people", "_people"), ("other poses", "_people40"))
PEOPLE_COLUMNS = ([f"Goal, {seen} (%)" for seen, _ in PEOPLE_SEEN]
                  + [f"Pedestrian contact, {seen} (%)" for seen, _ in PEOPLE_SEEN] + ["Structure contact, people (%)"])
CIVIL_COLUMNS = ["Episodes", "Goal (%)", "Goal, lab hall (%)", "Goal, civil eng. lab (%)",
                 "Structure contact, civil eng. lab (%)", "Pedestrian contact, robot moving (%)", "Time-out (%)"]


def _lab(d: dict, key: str, kind: str = "makerlab") -> float | None:
    """The rate ``key`` over the tiles of ``kind`` (all difficulties); None if the evaluation had none."""
    cells = [d["per_scenario"][f"{kind}/{b}"] for b in ("easy", "medium", "hard") if f"{kind}/{b}" in d["per_scenario"]]
    n = sum(c["episodes"] for c in cells)
    return sum(c[key] * c["episodes"] for c in cells) / n if n else None


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _cells(d: dict) -> list[str]:
    pc = d["pedestrian_contacts"]
    shield = d.get("shield_stats")
    return [f"{d['episodes']:,}", f"{100 * d['goal_reached_rate']:.1f}", _pct(_lab(d, "goal_reached_rate")),
            f"{100 * d['obstacle_collision_rate']:.1f}", f"{100 * pc['robot_moving_rate']:.1f}",
            f"{100 * pc['pedestrian_walked_in_rate']:.1f}", f"{100 * d['time_out_rate']:.1f}",
            f"{d['motion']['mean_speed_mps']:.2f}", f"{100 * shield['emergency_rate']:.1f}" if shield else "–"]


def _civil_cells(d: dict) -> list[str]:
    return [f"{d['episodes']:,}", f"{100 * d['goal_reached_rate']:.1f}", _pct(_lab(d, "goal_reached_rate")),
            _pct(_lab(d, "goal_reached_rate", "civil_lab")), _pct(_lab(d, "obstacle_collision_rate", "civil_lab")),
            f"{100 * d['pedestrian_contacts']['robot_moving_rate']:.1f}", f"{100 * d['time_out_rate']:.1f}"]


def _markdown(first: str, rows: list, columns: list = COLUMNS) -> str:
    lines = ["| " + " | ".join([first, *columns]) + " |", "|" + "---|" * (len(columns) + 1)]
    return "\n".join(lines + ["| " + " | ".join([name, *cells]) + " |" for name, cells in rows])


def _html(first: str, rows: list, caption: str, columns: list = COLUMNS) -> str:
    head = "".join(f'<th class="num">{html.escape(c)}</th>' for c in columns)
    body = "".join("<tr><td>" + html.escape(name) + "</td>" + "".join(f'<td class="num">{c}</td>' for c in cells) + "</tr>"
                   for name, cells in rows)
    return f"<caption>{caption}</caption><thead><tr><th>{first}</th>{head}</tr></thead><tbody>{body}</tbody>"


CAPTIONS = {
    "b2w": "<b>Table 5.</b> Policies on the lab mixture (half lab-hall tiles), all difficulty levels, 1,024 robots for "
           "300 s. Speed caps are drawn per episode from 0.4–1.5 m/s unless fixed. A pedestrian contact counts as the "
           "robot's if it was moving towards the person at more than 0.1 m/s. Shield stops: steps at which the shield "
           "set the command to zero because the robot's own momentum already breached.",
    "b2w_stress": "<b>Table 6.</b> The fused student with a built-in controller worse than in training and with "
                  "degraded sensing (setup of Table 5).",
    "b2w_shield": "<b>Table 2.</b> What the shield and the guard around it do (fused student, setup of Table 5).",
    "b2w_civil": "<b>Table 7.</b> The civil engineering lab: policies on a mixture in which it takes half the lab tiles "
                 "(setup of Table 5 otherwise); the released policy was trained without it.",
    "b2w_people": "<b>Table 9.</b> Pedestrians as the lidar and the depth cameras see them: capsules, as in "
                  "training, or Isaac Sim characters in walking poses (people: the 20 of 80 posed meshes used in the "
                  "fine-tuning; other poses: 20 others). Released students and students fine-tuned with people for 500 "
                  "iterations; pedestrian contacts with the robot moving towards the person, all contacts with the "
                  "capsules (setup of Table 5).",
    "b2w_camera": "<b>Table 8.</b> The camera student and the fused student it started from, on the lab mixture (the "
                  "camera student with its camera seeing the environment drawn as in the demo), 256 robots for 150 s.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default="logs/rsl_rl")
    parser.add_argument("--tag", default="release", help="The run (TAG of scripts/reproduce) to show.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.logs)
    dirs = {who: root / f"b2w_nav_lab_{run}/{args.tag}/eval"
            for who, run in (("teacher", "teacher"), ("fused", "fused_student"), ("lidar", "lidar_student"),
                             ("camera", "camera_student"))}
    load = lambda who, f: json.loads((dirs[who] / f).read_text()) if (dirs[who] / f).exists() else None  # noqa: E731
    rows = {
        "b2w": ("Policy", [(name, _cells(d)) for name, who, f in ROWS if (d := load(who, f))]),
        "b2w_stress": ("Fused student", [(name, _cells(d)) for name, f in STRESS if (d := load("fused", f))]),
        "b2w_shield": ("Fused student", [(name, _cells(d)) for name, f in SHIELD if (d := load("fused", f))]),
        "b2w_camera": ("Policy", [(name, _cells(d)) for name, who, f in CAMERA if (d := load(who, f))]),
    }
    runs = {"teacher": "teacher", "fused": "fused_student", "lidar": "lidar_student"}
    civil_rows = []
    for name, run, who, f in CIVIL:
        path = root / f"b2w_nav_lab_{runs[who]}/{run}/eval/{f}"
        if path.exists():
            civil_rows.append((name, _civil_cells(json.loads(path.read_text()))))
    people_rows = []
    for name, run, who, shield in PEOPLE:
        runs_eval = root / f"b2w_nav_lab_{runs[who]}/{run}/eval"
        d = [json.loads(f.read_text()) if (f := runs_eval / f"b2w_nominal{suffix}{'_shield' if shield else ''}.json")
             .exists() else None for _, suffix in PEOPLE_SEEN]
        if d[0] or d[1]:
            people_rows.append((name, [_pct(x and x["goal_reached_rate"]) for x in d]
                                + [_pct(x and x["pedestrian_contacts"]["robot_moving_rate"]) for x in d]
                                + [_pct(d[1] and d[1]["obstacle_collision_rate"])]))
    rows["b2w_people"] = ("Policy", people_rows)
    columns = {name: COLUMNS for name in rows}
    columns["b2w_people"] = PEOPLE_COLUMNS
    rows["b2w_civil"], columns["b2w_civil"] = ("Policy", civil_rows), CIVIL_COLUMNS
    for name, (first, r) in rows.items():
        print(_markdown(first, r, columns[name]) + "\n")
    if args.write:
        for path in (ROOT / "README.md", ROOT / "deploy" / "b2w" / "README.md"):
            text = path.read_text()
            for name, (first, r) in rows.items():
                text = re.sub(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)",
                              lambda m: m.group(1) + _markdown(first, r, columns[name]) + m.group(2), text, flags=re.S)
            path.write_text(text)
            print("wrote", path)
        page = ROOT / "docs" / "b2w" / "index.html"
        if page.exists():
            text = page.read_text()
            for name, (first, r) in rows.items():
                text = re.sub(rf'<table id="{name}Table">.*?</table>',
                              lambda m: f'<table id="{name}Table">{_html(first, r, CAPTIONS[name], columns[name])}</table>',
                              text, flags=re.S)
            page.write_text(text)
            print("wrote", page)


if __name__ == "__main__":
    main()
