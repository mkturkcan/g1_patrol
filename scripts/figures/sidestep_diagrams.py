"""Figures 1 and 2 of the Sidestep paper as vector graphics: the scenarios, interfaces and setup (teaser_scenarios) and
the block diagram of the closed loop (overview). They use the drawing primitives and the style of make_diagrams.py
(github.com/mkturkcan/elemental-design-elements, transformer design), at a size meant for a 6.9 in wide figure.

    python scripts/figures/sidestep_diagrams.py --icons <lucide-static icons dir>

writes <name>.svg, and with Playwright installed <name>.pdf (vector, fonts embedded) and <name>.png, into
paper/sidestep_bench/figures. Raster panels: the released stills (docs/media/stills), a dumped model view
(logs/bench_views) and the images of the paper's realism figure. Icons: Lucide (ISC license).
"""
import argparse
import math
import os

from make_diagrams import FILL, FONT, STROKE, Svg, media

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWS = os.path.join(ROOT, "logs", "bench_views", "sidewalk_s1")
BLUE, GRAY, RED = "#1F6FD0", "#8A8F98", "#D43D3D"


def realism_pair(tmp):
    """The render and its streamed transfer from the paper's realism figure (extracted from the PDF)."""
    import pymupdf

    doc = pymupdf.open(os.path.join(ROOT, "paper", "sidestep_bench", "figures", "realism_cl.pdf"))
    out = []
    for k, im in enumerate(doc[0].get_images(full=True)[:2]):
        pm = pymupdf.Pixmap(doc, im[0])
        if pm.n > 3:
            pm = pymupdf.Pixmap(pymupdf.csRGB, pm)
        path = os.path.join(tmp, f"sidestep_realism_{k}.png")
        pm.save(path)
        out.append(path)
    return out


def item2(s, x, y, w, h, title, lines, fill, icon=None, size=12.5, small=10.6):
    """A box with a title in medium weight and smaller lines below it."""
    s.rect(x, y, w, h, fill)
    lines = [lines] if isinstance(lines, str) else lines
    step = small * 1.28
    top = y + h / 2 - (size + step * len(lines)) / 2 + size / 2
    if icon:
        s.icon(icon, x + 11, y + (h - 20) / 2, size=20)
        tx, anchor = x + 40, "start"
    else:
        tx, anchor = x + w / 2, "middle"
    s.text(tx, top, title, size=size, weight=500, anchor=anchor)
    for k, line in enumerate(lines):
        s.text(tx, top + size / 2 + 2 + step * (k + 0.5), line, size=small, anchor=anchor, color="#202124")


def line(s, d, color="#000", sw=STROKE, dash=None, cap="butt"):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    s.body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="{cap}" stroke-linejoin="round"{extra}/>')


def colored_arrow(s, a, b, color, sw=2.2, dash=None):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    e = (b[0] - ux * 7, b[1] - uy * 7)
    line(s, f"M{a[0]:.1f},{a[1]:.1f} L{e[0]:.1f},{e[1]:.1f}", color, sw, dash)
    px, py = -uy * 3.6, ux * 3.6
    s.body.append(f'<path d="M{b[0]:.1f},{b[1]:.1f} L{e[0] + px:.1f},{e[1] + py:.1f} L{e[0] - px:.1f},{e[1] - py:.1f} z" fill="{color}"/>')


def topdown(s, x, y, w, h):
    """Eq. 2 from above: where the robot would be after 1 s and 2 s along each of its nine directions (rings), against
    where a crossing person will be (constant-velocity forecast). A direction is cut at the first time the two come
    within the gap g; directions that pass behind the person stay open."""
    s.clip_id += 1
    cid = f"c{s.clip_id}"
    s.defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/></clipPath>')
    s.rect(x, y, w, h, "white", sw=0)
    scale, v, g, H = 46.0, 0.8, 0.6, 2.0
    ox, oy = x + w * 0.44, y + h - 16
    P = lambda mx, my: (ox + mx * scale, oy - my * scale)
    p0, u = (-0.95, 1.86), (0.9, 0.0)
    ink2, faint, ring = "#3c4043", "#EEF0F3", "#B4BAC2"
    out = [f'<g clip-path="url(#{cid})">']
    add = out.append

    def halo(tx, ty, text, size=9.2, anchor="middle", color=ink2, weight=400):
        add(f'<text x="{tx:.1f}" y="{ty:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" fill="{color}" stroke="#fff" stroke-width="2.6" stroke-linejoin="round" paint-order="stroke">{text}</text>')

    def path(d, color, sw, dash=None, cap="round", opacity=1):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="{cap}" stroke-linejoin="round" opacity="{opacity}"{extra}/>')

    def arrowhead(tip, e, color, size=6.4):
        bx, by = tip[0] - e[0] * size, tip[1] + e[1] * size
        nx, ny = e[1] * size * 0.46, e[0] * size * 0.46
        add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{bx + nx:.1f},{by + ny:.1f} L{bx - nx:.1f},{by - ny:.1f} z" fill="{color}"/>')

    # half-metre grid
    k = -6
    while k <= 8:
        gx = ox + k * 0.5 * scale
        path(f"M{gx:.1f},{y} V{y + h}", faint, 1)
        k += 1
    for k in range(-1, 6):
        gy = oy - k * 0.5 * scale
        path(f"M{x},{gy:.1f} H{x + w}", faint, 1)
    # where the robot would be after 1 s and 2 s
    for t in (1.0, 2.0):
        r = v * t * scale
        a0, a1 = math.radians(-52), math.radians(52)
        path(f"M{ox + r * math.sin(a0):.1f},{oy - r * math.cos(a0):.1f} A{r:.1f},{r:.1f} 0 0 1 {ox + r * math.sin(a1):.1f},{oy - r * math.cos(a1):.1f}", ring, 1.1, "3 2.6", cap="butt")
        halo(ox + (r + 1) * math.sin(math.radians(-58)), oy - (r + 1) * math.cos(math.radians(-58)) + 1, f"{t:.0f} s", 8.6, color="#80868b")
    # the person's forecast: positions after 1 s and 2 s, and the gap that must stay free at 2 s
    a = P(*p0)
    end = P(p0[0] + u[0] * (H + 0.42), p0[1])
    c2 = P(p0[0] + u[0] * H, p0[1])
    add(f'<circle cx="{c2[0]:.1f}" cy="{c2[1]:.1f}" r="{g * scale:.1f}" fill="{RED}" fill-opacity="0.09" stroke="{RED}" stroke-opacity="0.55" stroke-width="1.1" stroke-dasharray="3 2.4"/>')
    path(f"M{a[0] + 9:.1f},{a[1]:.1f} H{end[0] - 6:.1f}", GRAY, 1.5, "4 3", cap="butt")
    arrowhead(end, (1.0, 0.0), GRAY, 6.5)
    # the nine directions
    marks = []
    for deg in range(-40, 41, 10):
        e = (math.sin(math.radians(deg)), math.cos(math.radians(deg)))
        tstar = next((i / 10 for i in range(1, int(H * 10) + 1)
                      if math.hypot(v * i / 10 * e[0] - p0[0] - u[0] * i / 10, v * i / 10 * e[1] - p0[1] - u[1] * i / 10) < g), None)
        start, full = P(e[0] * 0.36, e[1] * 0.36), P(e[0] * v * H, e[1] * v * H)
        if tstar is None:
            stop = P(e[0] * (v * H - 0.1), e[1] * (v * H - 0.1))
            path(f"M{start[0]:.1f},{start[1]:.1f} L{stop[0]:.1f},{stop[1]:.1f}", BLUE, 2.3, cap="butt")
            arrowhead(full, e, BLUE)
            continue
        c = P(e[0] * v * tstar, e[1] * v * tstar)
        q = P(e[0] * max(v * tstar - 0.2, 0.3), e[1] * max(v * tstar - 0.2, 0.3))
        path(f"M{q[0]:.1f},{q[1]:.1f} L{full[0]:.1f},{full[1]:.1f}", "#AEB4BC", 1.5, "0.1 3.1")
        path(f"M{start[0]:.1f},{start[1]:.1f} L{q[0]:.1f},{q[1]:.1f}", BLUE, 2.3, opacity=0.55)
        marks.append(c)
    for c in marks:
        path(f"M{c[0] - 2.9:.1f},{c[1] - 2.9:.1f}l5.8,5.8m0,-5.8l-5.8,5.8", "#fff", 4)
        path(f"M{c[0] - 2.9:.1f},{c[1] - 2.9:.1f}l5.8,5.8m0,-5.8l-5.8,5.8", RED, 1.8)
    for t in (1.0, 2.0):
        c = P(p0[0] + u[0] * t, p0[1])
        add(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="6.2" fill="#F7E6E8" stroke="#000" stroke-opacity="0.55" stroke-width="1.3" stroke-dasharray="2.4 1.9"/>')
        halo(c[0], c[1] - 12.5, f"{t:.0f} s", 8.6, color="#80868b")
    add(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="7.2" fill="{FILL["pink"]}" stroke="#000" stroke-width="1.9"/>')
    halo(a[0] - 11.5, a[1] + 0.5, "person", 9.6, anchor="end", color="#000", weight=500)
    add(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="8.2" fill="{FILL["orange"]}" stroke="#000" stroke-width="1.9"/>')
    add(f'<path d="M{ox - 3:.1f},{oy + 2:.1f} L{ox:.1f},{oy - 4:.1f} L{ox + 3:.1f},{oy + 2:.1f} z" fill="#000"/>')
    halo(ox - 12.5, oy + 0.5, "robot", 9.6, anchor="end", color="#000", weight=500)
    halo(c2[0] + g * scale * 0.72, c2[1] + g * scale * 0.72 + 6, "0.6 m gap", 8.6, anchor="start", color=RED)
    # legend
    lx, ly, lh = x + w - 88, y + h - 43, 12.6
    add(f'<rect x="{lx - 7}" y="{ly - 9}" width="{89}" height="{3 * lh + 6.5}" rx="3" fill="#fff" fill-opacity="0.92" stroke="#DADCE0" stroke-width="1"/>')
    path(f"M{lx},{ly}h12", BLUE, 2.3, cap="butt")
    path(f"M{lx},{ly + lh}h12", "#AEB4BC", 1.5, "0.1 3.1")
    path(f"M{lx + 3.2},{ly + 2 * lh - 2.8}l5.6,5.6m0,-5.6l-5.6,5.6", RED, 1.8)
    for k, lab in enumerate(("offered", "cut by forecast", "first conflict")):
        add(f'<text x="{lx + 17}" y="{ly + k * lh + 0.5:.1f}" font-family="{FONT}" font-size="8.8" dominant-baseline="middle" fill="{ink2}">{lab}</text>')
    out.append("</g>")
    s.body.append("".join(out))
    s.rect(x, y, w, h, "none")


def split_image(s, a, b, x, y, w, h):
    """Image `a` left of a slanted cut and `b` right of it, in one frame."""
    for path, shape in ((a, f"M{x},{y}h{w * 0.56}l{-w * 0.12},{h}h{-w * 0.44}z"), (b, f"M{x + w * 0.56},{y}h{w * 0.44}v{h}h{-w * 0.56}z")):
        s.image(path, x, y, w, h, frame=False)
        s.defs[-1] = s.defs[-1].split(">", 1)[0] + f'><path d="{shape}"/></clipPath>'
    line(s, f"M{x + w * 0.56},{y}l{-w * 0.12},{h}", "#fff", 3)
    s.rect(x, y, w, h, "none")
    for lx, lab, lw in ((x + 7, "render", 48), (x + w - 7 - 56, "transfer", 56)):
        s.rect(lx, y + h - 24, lw, 17, "white", sw=1.5)
        s.text(lx + lw / 2, y + h - 15.5, lab, size=10.5)


def teaser(icons, tmp):
    s = Svg(992, 470, icons)
    iw, ih, pad, gap = 146, 82, 12, 10
    cards = (("sidewalk", "Open sidewalk", "5.0 to 2.6 m wide"), ("scaffold", "Sidewalk shed", "passage 2.4 to 1.25 m"),
             ("construction", "Construction zone", "barriers and obstacles"), ("bottleneck", "Doorway", "stairs and entrances"),
             ("crosswalk", "Crosswalk", "pedestrians cross the route"), ("plaza", "Plaza", "open space, crossing flows"))
    groups = ((8, 4, "Sidewalk setup: goal about 14 m ahead"), (658, 2, "Crossing setup: goal 7 to 13 m away"))
    k = 0
    for gx, n, title in groups:
        s.group(gx, 8, 2 * pad + n * iw + (n - 1) * gap, 164, title, r=8)
        s.body[-1] = s.body[-1].replace('font-size="17"', 'font-size="14.5"')
        for j in range(n):
            key, name, sub = cards[k]
            x = gx + pad + j * (iw + gap)
            s.image(media(f"stills/{key}.jpg"), x, 46, iw, ih)
            s.text(x + iw / 2, 143, name, size=12.5, weight=500)
            s.text(x + iw / 2, 158, sub, size=10.6, color="#202124")
            k += 1
    # the model's view through the interfaces
    s.group(8, 186, 976, 214, "The model's view through the interfaces, and the realism track", r=8)
    s.body[-1] = s.body[-1].replace('font-size="17"', 'font-size="14.5"')
    pw, ph = 226, 127
    px = [20 + j * (pw + 16) for j in range(4)]
    s.image(os.path.join(VIEWS, "r0_t003.5_plain.jpg"), px[0], 224, pw, ph)
    s.image(os.path.join(VIEWS, "r0_t003.5.jpg"), px[1], 224, pw, ph)
    topdown(s, px[2], 224, pw, ph)
    render, transfer = realism_pair(tmp)
    split_image(s, render, transfer, px[3], 224, pw, ph)
    captions = (("VLMnav arrows", "directions clear in the depth image now"), ("Predictive affordance interface", "motions forecast to stay clear for 2 s"),
                ("Anticipation from above", "directions against a person's forecast"), ("Realistic video in the loop", "the model acts on the streamed transfer"))
    for x, (a, b) in zip(px, captions):
        s.text(x + pw / 2, 366, a, size=12.5, weight=500)
        s.text(x + pw / 2, 382, b, size=10.6, color="#202124")
    # setup
    chips = (("bot", "Unitree G1", "walking policy at 50 Hz", "orange"), ("camera", "Head camera", "640 × 360, 90° view", "lavender"),
             ("layout-grid", "6 tile types", "10 difficulty levels", "yellow"), ("users", "4 to 20 people", "10 to 40% inattentive", "pink"),
             ("cpu", "15 models", "33 configurations", "cyan"), ("chart-column", "192 episodes", "paired, per controller", "green"))
    cw = (976 - 5 * 10) / 6
    for j, (icon, a, b, fill) in enumerate(chips):
        item2(s, round(8 + j * (cw + 10), 1), 414, round(cw, 1), 46, a, b, fill, icon)
    return s


def overview(icons, tmp):
    s = Svg(1000, 600, icons)

    def group(x, y, w, h, title):
        s.group(x, y, w, h, title, r=8)
        s.body[-1] = s.body[-1].replace('font-size="17"', 'font-size="14.5"')

    def item(x, y, w, label, fill, icon=None, h=30, size=12.5):
        s.rect(x, y, w, h, fill)
        if icon:
            s.icon(icon, x + 9, y + (h - 18) / 2, size=18)
            s.text(x + 35, y + h / 2 + 0.5, label, size=size, anchor="start")
        else:
            s.text(x + w / 2, y + h / 2 + 0.5, label, size=size)

    # simulation
    group(8, 10, 220, 430, "Simulation (Isaac Lab)")
    s.image(media("stills/crosswalk.jpg"), 20, 46, 196, 110)
    s.text(118, 170, "New York street tile, 16 m × 10 m", size=10.6, color="#333")
    for k, (icon, label, fill) in enumerate((("layout-grid", "6 tile types, 10 levels", "yellow"), ("users", "Pedestrians (social forces)", "cyan"),
                                             ("dices", "Seeded episodes", "pink"), ("flag", "Goal per episode", "lavender"))):
        item(20, 186 + 38 * k, 196, label, fill, icon)
    item(20, 352, 196, "Unitree G1 humanoid", "orange", "bot")
    item(20, 398, 196, "Walking policy, 50 Hz", "green", "footprints")
    s.arrow([(118, 398), (118, 384)])
    # observations
    group(258, 10, 176, 284, "Observations")
    obs = (("camera", "Head camera"), ("layers", "Depth"), ("radar", "Lidar, 72 sectors"), ("users", "Person tracks"), ("map", "Roadway map"), ("flag", "Goal or instruction"))
    for k, (icon, label) in enumerate(obs):
        item(270, 46 + 40 * k, 152, label, "lavender" if k < 5 else "pink", icon, size=12)
    s.arrow([(228, 152), (256, 152)])
    # realism track
    s.rect(258, 314, 176, 126, "white", r=8, dash="7 4")
    s.text(346, 334, "Realism track", size=14.5, weight=500)
    render, transfer = realism_pair(tmp)
    s.image(render, 270, 350, 66, 37, frame=False)
    s.rect(270, 350, 66, 37, "none", sw=2)
    s.image(transfer, 356, 350, 66, 37, frame=False)
    s.rect(356, 350, 66, 37, "none", sw=2)
    s.arrow([(337, 368.5), (354.5, 368.5)], sw=2)
    s.text(346, 404, "causal video model, streamed", size=10.6, color="#202124")
    s.text(346, 419, "into the control loop", size=10.6, color="#202124")
    s.arrow([(346, 294), (346, 312.5)], dash="5 4")
    s.arrow([(434, 377), (462, 377)], dash="5 4")
    # interface
    group(464, 10, 250, 430, "Interface")
    item(476, 46, 226, "Native output or named actions", "yellow", "message-square", size=12)
    item(476, 84, 226, "VLMnav arrows: clear now", "yellow", "navigation", size=12)
    s.rect(476, 124, 226, 304, "white")
    s.text(589, 143, "Predictive affordance interface", size=12.5, weight=500)
    tx = [484 + k * 53.5 for k in range(4)]
    for x, name in zip(tx, ("depth", "lidar", "map", "forecast")):
        s.rect(x, 158, 49.5, 24, "green" if name != "forecast" else "pink", sw=2)
        s.text(x + 24.75, 170.5, name, size=10.8)
    cx = [x + 24.75 for x in tx]
    line(s, f"M{cx[0]},182 V196 H{cx[3]} V182", sw=2)
    for c in cx[1:3]:
        line(s, f"M{c},182 V196", sw=2)
    s.arrow([(589, 196), (589, 205.5)], sw=2)
    s.body.append('<circle cx="589" cy="217" r="11" fill="#fff" stroke="#000" stroke-width="2"/>')
    s.text(589, 217.5, "min", size=9.8)
    s.arrow([(589, 228), (589, 240.5)], sw=2)
    s.image(os.path.join(VIEWS, "r0_t003.5.jpg"), 489, 242, 200, 112.5)
    s.text(589, 370, "arrows end where the forecast", size=10.6, color="#202124")
    s.text(589, 384, "predicts a conflict within 2 s", size=10.6, color="#202124")
    s.rect(486, 396, 206, 24, "lavender", sw=2)
    s.text(589, 408.5, "prompt: clear distance per arrow", size=10.8)
    s.arrow([(434, 152), (462, 152)])
    # decision maker
    group(744, 10, 248, 430, "Decision maker")
    s.text(756, 58, "Foundation models (15)", size=12, weight=500, anchor="start")
    fam = (("Navigation and VLA models", ["EgoActor-8B, InternVLA-N1,", "LightNav-0, CityWalker"]),
           ("Embodied reasoning models", ["MiMo-Embodied-7B, Embodied-R1.5-8B,", "UnifoLM-ER-1-4B, Hy-Embodied-VLM-1.0"]),
           ("General vision-language models", ["Qwen3.5-9B, 27B, 122B-A10B, Qwen3.8-27B,", "Qwen3.8-Flash-Next, Gemma 4 31B"]))
    y = 68
    for name, lines in fam:
        item2(s, 756, y, 224, 52, name, lines, "cyan", size=11.2, small=10.2)
        y += 57
    item(756, y, 224, "Game agent: GameScaling-9B", "cyan", h=26, size=11)
    s.text(756, 290, "References on the same episodes", size=12, weight=500, anchor="start")
    item2(s, 756, 302, 224, 54, "Rules and planners", ["goal seeker, fixed rule,", "dynamic window, ORCA"], "green", size=12, small=10.2)
    item2(s, 756, 362, 224, 54, "Reinforcement-learning policies", ["privileged state, lidar,", "depth camera"], "green", size=12, small=10.2)
    s.arrow([(714, 225), (742, 225)])
    # the command closes the loop
    s.arrow([(868, 440), (868, 478), (160, 478), (160, 442)])
    s.text(514, 494, "velocity command (forward, lateral, yaw rate); the simulation pauses while the model decides", size=11.5, color="#333")
    # evaluation
    group(8, 516, 984, 76, "")
    s.body.pop()
    s.text(74, 554, "Evaluation", size=14.5, weight=500)
    item(140, 539, 304, "Outcome: goal, contacts, roadway, time-out", "orange", "circle-check", size=12)
    item(456, 539, 268, "Contact attribution: view and approach", "orange", "scan-eye", size=12)
    item(736, 539, 244, "192 paired episodes, 95% intervals", "orange", "chart-column", size=12)
    s.arrow([(50, 440), (50, 514.5)])
    return s


def to_pdf(svg_path, w, h):
    from playwright.sync_api import sync_playwright

    html = ('<html><head><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Google+Sans:ital,wght@0,400;0,500;0,700;1,400&display=block">'
            f'</head><body style="margin:0">{open(svg_path).read()}</body></html>')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=3)
        page.set_content(html, wait_until="networkidle")
        loaded = page.evaluate("document.fonts.ready.then(() => document.fonts.check(\"500 14px 'Google Sans'\"))")
        page.pdf(path=svg_path[:-4] + ".pdf", width=f"{w}px", height=f"{h}px", print_background=True, page_ranges="1")
        page.screenshot(path=svg_path[:-4] + ".png", clip={"x": 0, "y": 0, "width": w, "height": h})
        browser.close()
    return loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icons", required=True, help="lucide-static icons directory")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "sidestep_bench", "figures"))
    ap.add_argument("--tmp", default="/tmp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in (("teaser_scenarios", teaser), ("overview", overview)):
        s = fn(args.icons, args.tmp)
        path = os.path.join(args.out, name + ".svg")
        s.save(path)
        print("wrote", path, os.path.getsize(path) // 1024, "kB")
        try:
            print("wrote pdf and png; Google Sans loaded:", to_pdf(path, s.w, s.h))
        except ImportError:
            print("Playwright not installed: SVG only")


if __name__ == "__main__":
    main()
