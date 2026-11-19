"""Builds docs/sidestep/index.html, the results page of the Sidestep benchmark, from the paper's generated tables
(paper/sidestep_bench/tables/*.tex, written by scripts/bench/cvpr_numbers.py) and the page template next to this script
(sidestep_page.html). The page's style sheet is the one of docs/b2w/index.html, so the subpages look alike.

    python scripts/figures/sidestep_page.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES = os.path.join(ROOT, "paper", "sidestep_bench", "tables")
OUT = os.path.join(ROOT, "docs", "sidestep", "index.html")
FAMILIES = {"References": "ref", "Game agent": "game", "Navigation models": "nav", "Embodied reasoning models": "er",
            "General vision-language models": "vlm"}


def num(cell):
    """'53{\\tiny\\,[46,\\,60]}' -> (53.0, 46.0, 60.0); '31' -> (31.0, None, None); '--' -> None."""
    cell = cell.replace("$-$", "-").replace("$", "").replace("\\,", "").replace("\\scriptsize", "").replace("^\\ast", "")
    cell = cell.replace("{\\tiny", " ").replace("}", " ").strip()
    if cell in ("--", ""):
        return None
    m = re.match(r"^([+-]?[\d.]+)\s*(?:\[\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\s*\])?$", cell)
    if not m:
        raise ValueError(cell)
    return tuple(None if g is None else float(g) for g in m.groups())


def rows(name):
    for line in open(os.path.join(TABLES, name)):
        line = line.strip()
        if line.endswith("\\\\"):
            yield [c.strip() for c in line[:-2].split("&")]


def leaderboard():
    out, family = [], None
    for cells in rows("leaderboard.tex"):
        if len(cells) == 1:
            family = FAMILIES[re.search(r"\\emph\{(.*?)\}", cells[0]).group(1)]
            continue
        v = [num(c) for c in cells[2:]]
        out.append({"name": cells[0], "iface": cells[1], "family": family,
                    "sidewalk": v[0] and v[0][0], "crossing": v[1] and v[1][0], "succ": v[2], "clear": v[3], "ped": v[4],
                    "lang_sidewalk": v[5] and v[5][0], "lang_crossing": v[6] and v[6][0], "lang_ped": v[7]})
    return out


def factorial():
    return [{"name": cells[0].replace("\\,", " "), "succ": num(cells[1]), "ped": num(cells[2]), "structure": num(cells[3]),
             "roadway": num(cells[4]), "timeout": num(cells[5]), "sig": ["\\ast" in c for c in cells[1:6]]}
            for cells in rows("factorial.tex")]


def realism():
    out = []
    for cells in rows("realism_cl.tex"):
        if cells[1] != "both setups":
            continue
        name = cells[0].replace("\\PAI", "PAI").replace("~", " ").replace("--", "–")
        out.append({"name": name, "n": int(cells[2]), "render": float(cells[3]), "transfer": float(cells[4]),
                    "diff": num(cells[5]), "p": cells[6], "ped_render": float(cells[7]), "ped_transfer": float(cells[8]),
                    "ped_diff": num(cells[9])})
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    page = open(os.path.join(here, "sidestep_page.html")).read()
    style = re.search(r"<style>.*?</style>", open(os.path.join(ROOT, "docs", "b2w", "index.html")).read(), re.S).group(0)
    data = {"board": leaderboard(), "factorial": factorial(), "realism": realism()}
    page = page.replace("<!--STYLE-->", style).replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(page)
    print("wrote", OUT, len(page), "bytes;", len(data["board"]), "configurations")


if __name__ == "__main__":
    main()
