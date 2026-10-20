"""Write the evaluation results of ``scripts/reproduce`` into the README and the project page.

Reads the JSON files of ``scripts/reproduce/4_evaluate.sh`` (and ``5_ablations.sh``, if present) and the teacher's
training log, and regenerates:

* in ``README.md``, the tables between ``<!-- table:NAME -->`` and ``<!-- /table:NAME -->``;
* in ``docs/index.html``, the bodies of ``<table id="NAME">`` and the data of the per-scenario chart and the training
  curves;
* ``docs/data/results.json`` (used by ``make_video.py``).

Usage:
    python scripts/figures/build_results.py --eval logs/reproduce/eval --teacher_log logs/reproduce/2_teacher.log
"""

import argparse
import json
import math
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
KINDS = ["sidewalk", "scaffold", "crosswalk", "construction", "bottleneck", "plaza"]
TITLES = {"sidewalk": "Sidewalk", "scaffold": "Sidewalk shed", "crosswalk": "Crosswalk", "construction": "Construction zone",
          "bottleneck": "Doorway", "plaza": "Plaza"}  # fmt: skip
LEVELS = {"easy": "Easy (levels 1–4)", "medium": "Medium (5–7)", "hard": "Hard (8–10)"}
POLICIES = {"teacher": "Teacher", "lidar": "Lidar student", "camera": "Camera student"}

pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731


def load(eval_dir: str, name: str) -> dict | None:
    path = os.path.join(eval_dir, f"{name}.json")
    return json.load(open(path)) if os.path.isfile(path) else None


class Table:
    """A table rendered both as Markdown (README) and as HTML rows (project page)."""

    def __init__(self, head: list[str], rows: list[list[str]], caption: str = ""):
        self.head, self.rows, self.caption = head, rows, caption

    def markdown(self) -> str:
        esc = lambda s: s.replace("|", "\\|")  # noqa: E731
        lines = ["| " + " | ".join(esc(h) for h in self.head) + " |", "|---|" + "---:|" * (len(self.head) - 1)]
        lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in self.rows]
        return "\n".join(lines)

    def html(self) -> str:
        head = "".join(f'<th{" class=\"num\"" if k else ""}>{h}</th>' for k, h in enumerate(self.head))
        body = "".join("<tr>" + "".join(f'<td{" class=\"num\"" if k else ""}>{c}</td>' for k, c in enumerate(row)) + "</tr>"
                       for row in self.rows)
        caption = f"<caption>{self.caption}</caption>" if self.caption else ""
        return f"{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"


def outcomes(nav: dict) -> Table:
    rows = [
        ("Goal reached", lambda e: pct(e["goal_reached_rate"])),
        ("Pedestrian contact", lambda e: pct(e["pedestrian_collision_rate"])),
        ("Obstacle contact", lambda e: pct(e["obstacle_collision_rate"])),
        ("Roadway (left the sidewalk or crosswalk)", lambda e: pct(e.get("roadway_rate", 0.0))),
        ("Fall", lambda e: f'{100 * e["fall_rate"]:.2f}%'),
        ("Time-out (40 s)", lambda e: pct(e["time_out_rate"])),
        ("Mean time to goal (successful episodes)", lambda e: f'{e["mean_time_to_goal_s"]:.1f} s'),
        ("Time within 0.5 m of a pedestrian", lambda e: pct(e["time_within_0.5m_of_pedestrian"])),
        ("Completed episodes", lambda e: f'{e["episodes"]:,}'),
    ]
    cols = [k for k in POLICIES if nav.get(k)]
    return Table(["Outcome"] + [POLICIES[k] for k in cols], [[name] + [f(nav[k]) for k in cols] for name, f in rows],
                 "<b>Table 3.</b> Episode outcomes over all tiles (2,400 robots, 60 s, curriculum disabled).")


def scenarios(nav: dict) -> Table:
    cols = [k for k in POLICIES if nav.get(k)]
    rows = []
    for kind in KINDS:
        cells = [" / ".join(f'{100 * nav[k]["per_scenario"][f"{kind}/{lv}"]["goal_reached_rate"]:.0f}' for k in cols) + "%"
                 for lv in LEVELS]
        rows.append([TITLES[kind]] + cells)
    return Table(["Scenario"] + list(LEVELS.values()), rows)


def motion(nav: dict) -> Table:
    rows = [
        ("Forward command (m/s)", "forward_command_mps", "{:.2f}"),
        ("|Lateral command| (m/s)", "abs_lateral_command_mps", "{:.2f}"),
        ("|Yaw-rate command| (rad/s)", "abs_yaw_rate_command_radps", "{:.2f}"),
        ("Time with |yaw-rate command| > 0.8 rad/s", "fraction_yaw_rate_above_0.8radps", "pct"),
        ("Requested command at a range limit", "fraction_saturated", "pct"),
        ("Backward command", "fraction_backward", "pct"),
        ("Yaw-rate command change (rad/s²)", "yaw_rate_command_change_radps2", "{:.2f}"),
        ("Planar speed (m/s)", "planar_speed_mps", "{:.2f}"),
        ("Torso tilt", "torso_tilt_deg", "{:.1f}°"),
        ("Knee flexion", "knee_flexion_deg", "{:.1f}°"),
    ]
    cols = [k for k in POLICIES if nav.get(k)]
    fmt = lambda v, f: pct(v) if f == "pct" else f.format(v)  # noqa: E731
    return Table(["Quantity"] + [POLICIES[k] for k in cols], [[name] + [fmt(nav[k]["motion"][key], f) for k in cols] for name, key, f in rows],
                 "<b>Table 4.</b> Velocity commands and posture during the evaluation (time averages). A requested command "
                 "is at a range limit when it lies within 5% of the range of a command component (average over the three "
                 "components).")


def walker(runs: dict) -> Table:
    rows = [
        ("Forward velocity error (m/s)", lambda m: f'{m["mean_abs_err_lin_vel_x"]:.3f}'),
        ("Lateral velocity error (m/s)", lambda m: f'{m["mean_abs_err_lin_vel_y"]:.3f}'),
        ("Yaw-rate error (rad/s)", lambda m: f'{m["mean_abs_err_ang_vel_z"]:.3f}'),
        ("Torso tilt", lambda m: f'{m["mean_torso_tilt_deg"]:.1f}°'),
        ("Knee flexion", lambda m: f'{math.degrees(m["mean_knee_bend_rad"]):.1f}°'),
        ("Pelvis height (m)", lambda m: f'{m["mean_pelvis_height_m"]:.3f}'),
        ("Leg mechanical power (W)", lambda m: f'{m["mean_leg_power_w"]:.0f}'),
        ("Change of joint targets per step (rad)", lambda m: f'{m["mean_joint_target_change_rad"]:.3f}'),
        ("Torso roll and pitch rate (rad/s)", lambda m: f'{m["mean_torso_roll_pitch_rate_radps"]:.2f}'),
        ("Falls in 341 robot-minutes", lambda m: f'{m["falls"]}'),
    ]
    cols = [(name, m) for name, m in runs.items() if m]
    return Table(["Metric"] + [c for c, _ in cols], [[name] + [f(m) for _, m in cols] for name, f in rows],
                 "<b>Table 5.</b> Locomotion policy on city terrain with the training randomization (payload, torque "
                 "limits, latencies, friction, sensor noise and biases): 1,024 robots for 20 s each on 200 random terrain "
                 "tiles, random velocity commands from the training ranges or, in the second column, from the navigation "
                 "policy's ranges. Tracking errors are mean absolute errors; the other quantities are time averages. The "
                 "walker is trained on Newton; PhysX is a second simulator it was not trained on.")


def drift(runs: dict) -> Table:
    cols = [(name, d) for name, d in runs.items() if d]
    rows = []
    for speed in ("0.3", "0.6", "0.9"):
        for label, key, f in (("lateral drift (m/s)", "lateral_drift_mps", "{:.3f}"),
                              ("lateral offset after 3 s (m)", "lateral_offset_3s_m", "{:.2f}"),
                              ("heading change after 3 s (rad)", "heading_change_3s_rad", "{:.2f}")):
            rows.append([f"{speed} m/s, {label}"] + [f.format(d["speeds"][speed][key]) for _, d in cols])
    ablations = (" The ablations are trained with the same recipe and one change each "
                 "(<code>scripts/reproduce/5_ablations.sh</code>).") if len(cols) > 1 else ""
    return Table(["Straight command"] + [c for c, _ in cols], rows,
                 "<b>Table 6.</b> Drift under a constant straight command (PhysX, 256 robots per speed, after 1 s of "
                 "settling). Lateral drift is the mean absolute 1 s average of the heading-frame lateral velocity; the "
                 "offset is measured from the line the robot started on." + ablations)


STRESS = [("nominal", "Training randomization"), ("payload", "Torso payload +5 kg"), ("weak_motors", "Torque limits 80% of rated"),
          ("latency", "Actuation latency 15–25 ms"), ("slippery", "Friction 0.3 (static) / 0.25 (dynamic)"),
          ("sensor_noise", "Twice the sensor noise and biases"), ("combined", "All of the above")]  # fmt: skip


def walker_stress(eval_dir: str) -> Table:
    variants = [("walker", "Walker"), ("ablation_no_drift", "No drift penalty"), ("ablation_ground_1.0", "Ground friction 1.0")]
    cols = []
    for prefix, label in variants:
        for sim in ("newton", "physx"):
            stem = "walker_stress" if prefix == "walker" else f"{prefix}_stress"
            if load(eval_dir, f"{stem}_nominal_{sim}"):
                cols.append((f"{label}, {'Newton' if sim == 'newton' else 'PhysX'}", stem, sim))
    rows = []
    for condition, label in STRESS:
        cells = []
        for _, stem, sim in cols:
            m = load(eval_dir, f"{stem}_{condition}_{sim}")
            cells.append(f'{m["falls"]:,}' if m else "–")
        rows.append([label] + cells)
    return Table(["Condition (every robot)"] + [c for c, _, _ in cols], rows,
                 "<b>Table 7.</b> Falls in 341 robot-minutes (setup of Table 5) with every robot at, or beyond, the edge "
                 "of the training randomization. Counts from separate runs of the same condition differ by about their "
                 "square root (Table 5 and the first row here are separate runs).")


def nav_stress(eval_dir: str, nav: dict) -> Table:
    cell = lambda e: (f'{100 * e["goal_reached_rate"]:.1f} / {100 * e["pedestrian_collision_rate"]:.1f} / '  # noqa: E731
                      f'{100 * e["obstacle_collision_rate"]:.1f} / {100 * e["fall_rate"]:.1f}') if e else "–"
    conditions = [("nominal", "Nominal (Table 3)"),
                  ("sensors", "Sensors: twice the noise and odometry drift, 0–200 ms height-map latency"),
                  ("pedestrians", "Pedestrians: 20% faster, 30% fewer notice the robot"),
                  ("robot", "Robot: +5 kg, torque 80%, 15–25 ms actuation latency, friction 0.3")]
    rows = []
    for condition, label in conditions:
        row = [label]
        for k in POLICIES:
            e = nav.get(k) if condition == "nominal" else load(eval_dir, f"nav_{k}_stress_{condition}")
            row.append(cell(e))
        rows.append(row)
    return Table(["Condition"] + list(POLICIES.values()), rows,
                 "<b>Table 8.</b> Navigation under degraded conditions applied to every robot: goal reached / pedestrian "
                 "contact / obstacle contact / fall (%). The teacher observes privileged state, so the sensor condition "
                 "does not apply to it.")


CIVIC = {"teacher": "civic_teacher", "lidar": "civic_lidar", "camera": "civic_camera"}


def civic(eval_dir: str) -> Table:
    nav = {k: load(eval_dir, name) for k, name in CIVIC.items()}
    per100 = lambda v: f"{100 * v:.1f}"  # noqa: E731
    rows = [
        ("Goal reached", lambda e, c: pct(e["goal_reached_rate"])),
        ("Pedestrian contact (including passing between a dog and its owner)", lambda e, c: pct(e["pedestrian_collision_rate"])),
        ("Obstacle contact", lambda e, c: pct(e["obstacle_collision_rate"])),
        ("Roadway (left the sidewalk, or crossed against the signal)", lambda e, c: pct(e.get("roadway_rate", 0.0))),
        ("Fall", lambda e, c: f'{100 * e["fall_rate"]:.2f}%'),
        ("Time-out (60 s)", lambda e, c: pct(e["time_out_rate"])),
        ("Litter picked up, per item", lambda e, c: pct(c["litter_picked_per_item"])),
        ("Litter put in a basket, per item", lambda e, c: pct(c["litter_in_basket_per_item"])),
        ("Bags set down at a curbside pile, per bag", lambda e, c: pct(c["bags_delivered_per_bag"])),
        ("Fallen person: help called", lambda e, c: pct(c["fallen_person_helped"])),
        ("Attack: fall", lambda e, c: pct(c["attack_fall_rate"])),
        ("Attack: contact with a bystander", lambda e, c: pct(c["attack_pedestrian_collision_rate"])),
        ("Attack: mean distance to the attacker (m)", lambda e, c: f'{c["attack_mean_gap_m"]:.2f}'),
        ("Attack: braced while the attacker is within 1 m", lambda e, c: pct(c["attack_braced_when_close"])),
        ("Shoves and kicks received per episode", lambda e, c: f'{c["shoves_per_episode"]:.2f}'),
        ("Braced without a threat (time)", lambda e, c: pct(c["braced_without_threat"])),
        ("Time within 1.2 m of a person who needs room", lambda e, c: pct(c["time_within_1.2m_of_person_needing_room"])),
        ("Groups or lines split, per 100 episodes", lambda e, c: per100(c["group_or_line_splits_per_episode"])),
        ("Signal violations, per 100 episodes", lambda e, c: per100(c["signal_violations_per_episode"])),
        ("Completed episodes", lambda e, c: f'{e["episodes"]:,}'),
    ]
    cols = [k for k in POLICIES if nav.get(k)]
    return Table(["Outcome"] + [POLICIES[k] for k in cols],
                 [[name] + [f(nav[k], nav[k]["civic"]) for k in cols] for name, f in rows],
                 "<b>Table 9.</b> Civic task: episode outcomes and civic metrics over all tiles (2,400 robots, 120 s, "
                 "curriculum disabled). Attack rows are over the episodes in which an attack took place.")


REACH_STRESS = [("nominal", "Training randomization"), ("attacks", "Shoves and kicks half of the time"),
                ("brace_retreat", "Braced, walking at 0.8–1.3 m/s and turning while attacked"),
                ("deep_squat", "Deep squats only"), ("hand_load", "3 kg in one hand all the time"),
                ("combined", "Payload, weak motors, latency, friction 0.3 and sensor noise (Table 7)")]  # fmt: skip


def reach_walker_stress(eval_dir: str) -> Table:
    cols = [(f"Reaching walker, {'Newton' if sim == 'newton' else 'PhysX'}", sim) for sim in ("newton", "physx")
            if load(eval_dir, f"reach_walker_stress_nominal_{sim}")]
    rows = []
    for condition, label in REACH_STRESS:
        cells = []
        for _, sim in cols:
            m = load(eval_dir, f"reach_walker_stress_{condition}_{sim}")
            cells.append(f'{m["falls"]:,}' if m else "–")
        rows.append([label] + cells)
    return Table(["Condition (every robot)"] + [c for c, _ in cols], rows,
                 "<b>Table 10.</b> Falls of the reaching walker in 341 robot-minutes (setup of Table 7) under its "
                 "training randomization and with every robot at the edge of one of its tasks.")


def civic_stress(eval_dir: str) -> Table:
    cell = lambda e: (f'{100 * e["goal_reached_rate"]:.1f} / {100 * e["pedestrian_collision_rate"]:.1f} / '  # noqa: E731
                      f'{100 * e["obstacle_collision_rate"]:.1f} / {100 * e["fall_rate"]:.1f}') if e else "–"
    conditions = [("nominal", "Nominal (Table 9)"), ("attacks", "An attack in every episode, at full strength"),
                  ("incidents", "Attacks, people blocking the way and fallen people at twice the training rates"),
                  ("sensors", "Sensors: twice the noise and odometry drift, 0–200 ms latency"),
                  ("robot", "Robot: +5 kg, torque 80%, 15–25 ms actuation latency, friction 0.3")]
    rows = []
    for condition, label in conditions:
        row = [label]
        for k, name in CIVIC.items():
            row.append(cell(load(eval_dir, name if condition == "nominal" else f"{name}_stress_{condition}")))
        rows.append(row)
    return Table(["Condition"] + list(POLICIES.values()), rows,
                 "<b>Table 11.</b> Civic task under degraded conditions applied to every robot: goal reached / pedestrian "
                 "contact / obstacle contact / fall (%).")


def curves(log: str, block: int = 25) -> dict:
    points, cur = {}, None
    for line in open(log, errors="ignore"):
        m = re.search(r"Learning iteration (\d+)/", line)
        if m:
            cur = {"it": int(m.group(1))}
        elif cur is not None and "Episode_Termination/goal_reached" in line:
            cur["goal"] = float(line.split(":")[-1])
        elif cur is not None and "Curriculum/levels/mean_level" in line:
            cur["lvl"] = float(line.split(":")[-1])
        if cur is not None and "goal" in cur and "lvl" in cur:
            points[cur["it"]] = (cur["goal"], cur["lvl"])
            cur = None
    it = np.array(sorted(points))
    goal, lvl = np.array([points[k][0] for k in it]), np.array([points[k][1] for k in it])
    n = len(it) // block
    return {"iteration": [int(it[i * block + block // 2]) for i in range(n)],
            "goal": [round(float(goal[i * block:(i + 1) * block].mean()), 4) for i in range(n)],
            "level": [round(float(lvl[i * block:(i + 1) * block].mean()) + 1, 3) for i in range(n)]}  # levels as 1-10


def per_scenario(nav: dict) -> dict:
    return {k: {kind: {lv: {"goal": round(e["per_scenario"][f"{kind}/{lv}"]["goal_reached_rate"], 4),
                            "ped": round(e["per_scenario"][f"{kind}/{lv}"]["pedestrian_collision_rate"], 4),
                            "obst": round(e["per_scenario"][f"{kind}/{lv}"]["obstacle_collision_rate"], 4),
                            "fall": round(e["per_scenario"][f"{kind}/{lv}"]["fall_rate"], 4),
                            "eps": e["per_scenario"][f"{kind}/{lv}"]["episodes"]} for lv in LEVELS} for kind in KINDS}
            for k, e in nav.items() if e}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", default=os.path.join(ROOT, "logs", "reproduce", "eval"))
    parser.add_argument("--teacher_log", default=os.path.join(ROOT, "logs", "reproduce", "2_teacher.log"))
    parser.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    parser.add_argument("--html", default=os.path.join(ROOT, "docs", "index.html"))
    args = parser.parse_args()
    nav = {k: load(args.eval, f"nav_{k}") for k in POLICIES}
    tables = {
        "outcomes": outcomes(nav),
        "scenarios": scenarios(nav),
        "motion": motion(nav),
        "walker": walker({"Walker, Newton": load(args.eval, "walker_newton"),
                          "Walker, navigation commands": load(args.eval, "walker_navcmd"),
                          "Walker, PhysX": load(args.eval, "walker_physx")}),
        "drift": drift({"Walker": load(args.eval, "walker_drift"),
                        "No drift penalty": load(args.eval, "ablation_no_drift_drift"),
                        "Ground friction 1.0": load(args.eval, "ablation_ground_1.0_drift")}),
        "walker_stress": walker_stress(args.eval),
        "nav_stress": nav_stress(args.eval, nav),
    }
    if load(args.eval, "civic_teacher"):
        tables.update({"civic": civic(args.eval), "reach_walker_stress": reach_walker_stress(args.eval),
                       "civic_stress": civic_stress(args.eval)})
    readme = open(args.readme).read()
    for name, table in tables.items():
        readme, n = re.subn(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)",
                            lambda m, t=table: m.group(1) + t.markdown() + m.group(2), readme, flags=re.S)
        if not n:
            print(f"README: no markers for table {name}")
    open(args.readme, "w").write(readme)
    html = open(args.html).read()
    for name, table in tables.items():
        if name == "scenarios":
            continue
        html, n = re.subn(rf'<table id="{name}Table">.*?</table>', lambda m, t=table, k=name: f'<table id="{k}Table">{t.html()}</table>',
                          html, flags=re.S)
        if not n:
            print(f"page: no table {name}Table")
    results, curve = per_scenario(nav), curves(args.teacher_log)
    html = re.sub(r"const RESULTS = \{.*?\};\n", lambda _: "const RESULTS = " + json.dumps(results, separators=(",", ":")) + ";\n", html, count=1)
    html = re.sub(r"const CURVES = \{.*?\};\n", lambda _: "const CURVES = " + json.dumps(curve, separators=(",", ":")) + ";\n", html, count=1)
    open(args.html, "w").write(html)
    summary = {k: {"goal": e["goal_reached_rate"], "ped": e["pedestrian_collision_rate"], "obst": e["obstacle_collision_rate"],
                   "fall": e["fall_rate"], "episodes": e["episodes"], "motion": e["motion"]} for k, e in nav.items() if e}
    data = os.path.join(os.path.dirname(args.html), "data", "results.json")
    json.dump({"policies": results, "summary": summary}, open(data, "w"), indent=1)
    print("wrote", args.readme, args.html, data)


if __name__ == "__main__":
    main()
