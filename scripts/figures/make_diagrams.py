"""Architecture and workflow diagrams for the project page and README (SVG with embedded images and icons).

Style follows https://github.com/mkturkcan/elemental-design-elements (transformer design): pastel boxes with 2.5 px
black outlines and small corner radii, light-gray group containers, thick black arrows.

Usage:
    python scripts/figures/make_diagrams.py --icons <lucide-static icons dir> --out docs/figures
Icons: Lucide (https://lucide.dev, ISC license).
"""

import argparse
import base64
import io
import os
import re
from xml.sax.saxutils import escape

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "..", "docs", "media")

FILL = {
    "pink": "#F2D8DA",
    "cyan": "#C2E9F2",
    "yellow": "#F1F2C2",
    "orange": "#F2D6B3",
    "green": "#CCE7CF",
    "lavender": "#DBDFEF",
    "group": "#F3F3F4",
    "white": "#FFFFFF",
}
FONT = "'Google Sans', 'Google Sans Text', Helvetica, Arial, sans-serif"
STROKE = 2.5


class Svg:
    def __init__(self, width: int, height: int, icons: str):
        self.w, self.h, self.icons = width, height, icons
        self.defs, self.body = [], []
        self.clip_id = 0

    # -- primitives -----------------------------------------------------------------------------------------------
    def rect(self, x, y, w, h, fill="white", r=4, dash=None, stroke="#000", sw=STROKE):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{FILL.get(fill, fill)}" '
                         f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=15, weight=400, anchor="middle", color="#000", italic=False, rotate=0):
        """Text with line breaks (\\n) and subscripts written as _{...}."""
        lines = s.split("\n")
        style = "font-style:italic;" if italic else ""
        y0 = y - (len(lines) - 1) * size * 0.62

        def markup(line):
            parts = re.split(r"_\{([^}]*)\}", line)
            return "".join(escape(p) if i % 2 == 0 else f'<tspan baseline-shift="sub" font-size="{size * 0.72:.1f}">{escape(p)}</tspan>'
                           for i, p in enumerate(parts))

        spans = "".join(f'<tspan x="{x}" y="{y0 + i * size * 1.24:.1f}">{markup(line)}</tspan>' for i, line in enumerate(lines))
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.body.append(f'<text font-family="{FONT}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
                         f'dominant-baseline="middle" fill="{color}" style="{style}"{tr}>{spans}</text>')

    def icon(self, name, x, y, size=22, color="#000"):
        src = open(os.path.join(self.icons, f"{name}.svg")).read()
        inner = re.sub(r"(?s)^.*?<svg[^>]*>", "", src)
        inner = re.sub(r"(?s)</svg>.*$", "", inner)
        s = size / 24.0
        self.body.append(f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{color}" stroke-width="2" '
                         f'stroke-linecap="round" stroke-linejoin="round">{inner}</g>')

    def image(self, path, x, y, w, h, crop=None, r=4, frame=True):
        im = Image.open(path).convert("RGB")
        if crop:
            im = im.crop(crop)
        # fit (cover) into w x h at 2x for sharpness
        tw, th = int(w * 2), int(h * 2)
        sw, sh = im.size
        scale = max(tw / sw, th / sh)
        im = im.resize((max(tw, round(sw * scale)), max(th, round(sh * scale))), Image.LANCZOS)
        ox, oy = (im.size[0] - tw) // 2, (im.size[1] - th) // 2
        im = im.crop((ox, oy, ox + tw, oy + th))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        data = base64.b64encode(buf.getvalue()).decode()
        self.clip_id += 1
        cid = f"c{self.clip_id}"
        self.defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath>')
        self.body.append(f'<image href="data:image/jpeg;base64,{data}" x="{x}" y="{y}" width="{w}" height="{h}" '
                         f'clip-path="url(#{cid})" preserveAspectRatio="none"/>')
        if frame:
            self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="#000" '
                             f'stroke-width="{STROKE}"/>')

    def arrow(self, points, dash=None, sw=STROKE, head=True):
        d = "M" + " L".join(f"{px},{py}" for px, py in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if head else ""
        self.body.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="{sw}" stroke-linejoin="round"{extra}{marker}/>')

    # -- composites -----------------------------------------------------------------------------------------------
    def group(self, x, y, w, h, title, r=10):
        self.rect(x, y, w, h, "group", r=r)
        self.text(x + w / 2, y + 26, title, size=17, weight=500)

    def item(self, x, y, w, h, label, fill, icon=None, size=15):
        self.rect(x, y, w, h, fill)
        if icon:
            self.icon(icon, x + 12, y + (h - 22) / 2)
            self.text(x + 44, y + h / 2, label, size=size, anchor="start")
        else:
            self.text(x + w / 2, y + h / 2, label, size=size)

    def save(self, path):
        marker = ('<marker id="arrow" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="4.2" markerHeight="4.2" '
                  'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker>')
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">'
               f'<defs>{marker}{"".join(self.defs)}</defs><rect width="100%" height="100%" fill="#fff"/>{"".join(self.body)}</svg>')
        with open(path, "w") as f:
            f.write(svg)


def media(name):
    return os.path.join(MEDIA, name)


def architecture(icons: str, out: str):
    s = Svg(1600, 760, icons)
    # simulation
    s.group(20, 40, 300, 610, "Simulation")
    s.image(media("stills/sidewalk.jpg"), 36, 80, 268, 151)
    s.text(170, 250, "New York street tile, 16 m × 10 m", size=14, color="#333")
    rows = [("users", "Pedestrians (social forces)", "cyan"), ("building-2", "Structures and furniture", "yellow"),
            ("map", "Height field: curbs, potholes", "orange"), ("route", "Route and goal per episode", "pink"),
            ("timer", "PhysX, 200 Hz", "lavender")]
    for k, (ic, lab, fill) in enumerate(rows):
        s.item(36, 280 + 66 * k, 268, 46, lab, fill, ic, size=14.5)

    # observations
    s.group(370, 40, 360, 330, "Privileged observations (teacher)")
    obs = [("flag-triangle-right", "Goal position  ·  2", "pink"), ("users", "10 nearest pedestrians: p, v  ·  50", "cyan"),
           ("scan-line", "Obstacle distance, 2 × 72 sectors  ·  144", "yellow"), ("map", "Height scan, 11 × 11  ·  121", "orange"),
           ("activity", "Velocity, gravity, roadway, action  ·  15", "lavender")]
    for k, (ic, lab, fill) in enumerate(obs):
        s.item(386, 80 + 56 * k, 328, 44, lab, fill, ic, size=14.5)
    s.group(370, 400, 360, 250, "Onboard observations (students)")
    s.image(media("sensor_lidar.png"), 386, 440, 120, 120)
    s.image(media("sensor_depth.png"), 524, 440, 190, 113)
    s.text(446, 578, "MID-360 elevation map\n40 × 40 × 2", size=13, color="#333")
    s.text(619, 578, "D435i depth\n64 × 38 × 3", size=13, color="#333")
    s.item(386, 600, 328, 36, "Goal, velocity, gravity, action  ·  14", "lavender", size=14)

    # navigation policy: teacher (top) and student (bottom) both produce the velocity command (middle)
    s.group(780, 40, 330, 610, "Navigation policy, 10 Hz")
    s.text(800, 88, "Teacher  π_{T}", size=15, weight=500, anchor="start")
    s.item(800, 104, 250, 52, "MLP  512 · 256 · 128", "green", "brain-circuit")
    s.item(800, 300, 250, 60, "Velocity command\nv_{x}, v_{y}, ω_{z}", "lavender")
    s.text(800, 394, "Student  π_{S}", size=15, weight=500, anchor="start")
    s.item(800, 412, 250, 52, "MLP  512 · 256 · 128", "green", "brain-circuit")
    s.item(800, 520, 250, 48, "CNN encoder  16 · 32 · 32", "orange", "layers")
    s.arrow([(925, 156), (925, 297)])
    s.arrow([(925, 520), (925, 467)])
    s.arrow([(925, 412), (925, 363)])
    # distillation: the student is trained to reproduce the teacher's command
    s.arrow([(1050, 130), (1078, 130), (1078, 438), (1053, 438)], dash="7 6", sw=2)
    s.text(1094, 214, "distillation (action MSE)", size=13, color="#333", rotate=90)

    # locomotion policy
    s.group(1160, 190, 250, 330, "Locomotion policy, 50 Hz")
    s.text(1285, 238, "frozen", size=13, color="#555", italic=True)
    s.item(1176, 258, 218, 62, "Command, joint states,\ngait phase, height scan", "yellow", size=14)
    s.item(1176, 348, 218, 50, "MLP  512 · 256 · 128", "green", "brain-circuit")
    s.item(1176, 428, 218, 44, "Joint targets  ·  37", "pink", "footprints")
    s.arrow([(1285, 320), (1285, 345)])
    s.arrow([(1285, 398), (1285, 425)])

    # robot
    s.image(media("robot_crop.jpg"), 1452, 150, 128, 300)
    s.text(1516, 474, "Unitree G1\n(PD control)", size=14, weight=500)

    # flow
    s.arrow([(320, 205), (367, 205)])
    s.arrow([(320, 525), (367, 525)])
    s.arrow([(730, 130), (797, 130)])
    s.arrow([(730, 544), (797, 544)])
    s.arrow([(1050, 330), (1157, 330)])
    s.arrow([(1410, 300), (1449, 300)])
    s.arrow([(1516, 500), (1516, 712), (170, 712), (170, 653)])
    s.text(843, 734, "next state: base pose and velocity, joint states, contacts, pedestrian positions", size=14, color="#333")
    s.save(out)


def workflow(icons: str, out: str):
    s = Svg(1600, 700, icons)
    xs = [20, 340, 660, 980, 1300]
    W, Y, H = 280, 90, 540
    titles = ["Scenario generation", "1  Locomotion policy", "2  Teacher policy", "3  Student policies", "Evaluation"]
    for x, t in zip(xs, titles):
        s.group(x, Y, W, H, t)

    # scenario generation
    thumbs = ["sidewalk_chase", "scaffold_chase", "crosswalk_chase", "construction_drone", "bottleneck_drone", "plaza_drone"]
    for k, name in enumerate(thumbs):
        cx, cy = xs[0] + 16 + (k % 2) * 128, Y + 46 + (k // 2) * 76
        s.image(media(f"{name}.jpg"), cx, cy, 120, 68, r=3, frame=True)
    s.item(xs[0] + 16, Y + 286, 248, 44, "6 types × 10 levels", "yellow", "construction")
    s.item(xs[0] + 16, Y + 342, 248, 44, "Pedestrian lanes, timing", "cyan", "users")
    s.item(xs[0] + 16, Y + 398, 248, 44, "Routes, structures, terrain", "orange", "route")
    s.item(xs[0] + 16, Y + 454, 248, 60, "Passability test\nfor every tile", "lavender", "shield-check", size=14)

    # locomotion
    x = xs[1]
    s.image(media("stills/sidewalk_side.jpg"), x + 16, Y + 46, 248, 140, crop=(100, 250, 1000, 758))
    s.item(x + 16, Y + 206, 248, 44, "PPO, Newton (MuJoCo-Warp)", "lavender", "cpu", size=14)
    s.item(x + 16, Y + 262, 248, 60, "7,000 iterations\nup to 16,384 robots", "yellow", "layers", size=14)
    s.item(x + 16, Y + 334, 248, 60, "Actuator limits, gait\nand posture rewards", "orange", "gauge", size=14)
    s.item(x + 16, Y + 406, 248, 44, "Pushes up to 1.2 m/s", "cyan", "move", size=14)
    s.item(x + 16, Y + 470, 248, 46, "Frozen walker", "green", "footprints")

    # teacher
    x = xs[2]
    s.image(media("stills/plaza.jpg"), x + 16, Y + 46, 248, 140)
    s.item(x + 16, Y + 206, 248, 44, "PPO, PhysX", "lavender", "cpu", size=14)
    s.item(x + 16, Y + 262, 248, 60, "3,350 iterations\n8,192 robots, 4 GPUs", "yellow", "layers", size=14)
    s.item(x + 16, Y + 334, 248, 60, "Privileged state:\npedestrians, obstacles", "cyan", "users", size=14)
    s.item(x + 16, Y + 406, 248, 44, "Curriculum, 10 levels", "orange", "target", size=14)
    s.item(x + 16, Y + 470, 248, 46, "Teacher  π_{T}", "green", "brain-circuit")

    # students
    x = xs[3]
    s.image(media("sensor_lidar.png"), x + 16, Y + 46, 140, 140)
    s.image(media("sensor_depth.png"), x + 164, Y + 46, 100, 140)
    s.item(x + 16, Y + 206, 248, 44, "Action matching to π_{T}", "lavender", "graduation-cap", size=14)
    s.item(x + 16, Y + 262, 248, 60, "MID-360 lidar\nelevation map", "yellow", "radar", size=14)
    s.item(x + 16, Y + 334, 248, 60, "D435i depth camera\nnoise, dropout, occlusion", "orange", "camera", size=14)
    s.item(x + 16, Y + 406, 248, 44, "CNN + MLP", "cyan", "layers", size=14)
    s.item(x + 16, Y + 470, 248, 46, "Students  π_{S}", "green", "brain-circuit")

    # evaluation
    x = xs[4]
    s.image(media("stills/bottleneck.jpg"), x + 16, Y + 46, 248, 140)
    s.item(x + 16, Y + 206, 248, 60, "120 tiles\ncurriculum disabled", "lavender", "map", size=14)
    s.item(x + 16, Y + 278, 248, 60, "Success, contact\nand fall rates", "yellow", "target", size=14)
    s.item(x + 16, Y + 350, 248, 60, "RTX rendering with\nanimated pedestrians", "orange", "camera", size=14)
    s.item(x + 16, Y + 422, 248, 44, "13,238 episodes", "cyan", "layers", size=14)

    # flow between stages
    mid = Y + 290
    for a, b in zip(xs[:-1], xs[1:]):
        s.arrow([(a + W, mid), (b - 3, mid)])
    # curriculum feedback: teacher -> scenario generation (over the top)
    s.arrow([(xs[2] + W / 2, Y), (xs[2] + W / 2, 44), (xs[0] + W / 2, 44), (xs[0] + W / 2, Y - 3)], dash="7 6", sw=2)
    s.text((xs[0] + xs[2] + W) / 2, 30, "curriculum: harder tile after success, easier tile after contact or fall", size=14, color="#333")
    # frozen walker feeds the teacher and students
    s.arrow([(xs[1] + W / 2, Y + H), (xs[1] + W / 2, 672), (xs[3] + W / 2, 672), (xs[3] + W / 2, Y + H + 3)], dash="7 6", sw=2)
    s.arrow([(xs[2] + W / 2, 672), (xs[2] + W / 2, Y + H + 3)], dash="7 6", sw=2)
    s.text(xs[2] + W / 2 + 120, 690, "frozen walker executes all navigation policies", size=13.5, color="#333")
    s.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--icons", required=True, help="Directory with lucide-static icon SVGs.")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs", "figures"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    architecture(args.icons, os.path.join(args.out, "architecture.svg"))
    workflow(args.icons, os.path.join(args.out, "workflow.svg"))
    print("wrote", args.out)
