"""Overview video for the project page: scenarios, commands, sensing, architecture, workflow and results (1920 x 1080).

Inputs are the demo renders (scripts/demo/nyc_demo.py), the figures (make_diagrams.py), the point cloud and sensor
images in docs/, and the evaluation results. Text is set in Google Sans (pass the TTF directory).

Usage:
    python scripts/figures/make_video.py --renders <renders dir> --fonts <dir with GoogleSans-*.ttf> \
        --gait_before <clip> --gait_after <clip> --out docs/media/overview.mp4
"""

import argparse
import json
import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "..", "docs")
W, H, FPS = 1920, 1080, 30
INK, INK2, RULE, BG = (32, 33, 36), (95, 99, 104), (218, 220, 224), (255, 255, 255)
BULLET = {"sidewalk": (0, 147, 60), "scaffold": (185, 51, 173), "crosswalk": (0, 57, 166),
          "construction": (255, 99, 25), "bottleneck": (153, 102, 51), "plaza": (252, 204, 10)}  # fmt: skip
LETTER = {"sidewalk": "W", "scaffold": "S", "crosswalk": "X", "construction": "C", "bottleneck": "D", "plaza": "P"}
TITLE = {"sidewalk": "Sidewalk", "scaffold": "Sidewalk shed", "crosswalk": "Crosswalk",
         "construction": "Construction zone", "bottleneck": "Doorway", "plaza": "Plaza"}  # fmt: skip

SHOTS = [  # (scenario, render clip, start s, caption)
    ("sidewalk", "side/sidewalk_L6_side_ep01_goal_reached.mp4", 1.5, "Sidewalk, level 7. Two-way pedestrian flow and street furniture along the curb."),
    ("scaffold", "chase/scaffold_L6_chase_ep02_goal_reached.mp4", 2.0, "Sidewalk shed, level 7. Scaffold posts and guard rails narrow the walkway."),
    ("crosswalk", "chase/crosswalk_L7_chase_ep06_goal_reached.mp4", 1.0, "Crosswalk, level 8. Curbs, potholes and opposing pedestrian flows."),
    ("construction", "drone/construction_L7_drone_ep01_goal_reached.mp4", 3.0, "Construction zone, level 8. Chicane of fences and jersey barriers."),
    ("bottleneck", "chase/bottleneck_L9_chase_ep01_goal_reached.mp4", 3.0, "Doorway, level 10. Pedestrians exit a subway stairwell next to the route."),
    ("plaza", "drone/plaza_L9_drone_ep02_goal_reached.mp4", 2.0, "Plaza, level 10. Crossing flows and stationary pedestrians."),
]


class Fonts:
    def __init__(self, d):
        self.d = d

    def __call__(self, size, weight=400):
        return ImageFont.truetype(os.path.join(self.d, f"GoogleSans-{weight}.ttf"), size)


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale and center-crop to w x h."""
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def contain(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def bullet(draw, cx, cy, r, key, font):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=BULLET[key])
    draw.text((cx, cy + 1), LETTER[key], font=font, fill=(17, 17, 17) if key == "plaza" else (255, 255, 255), anchor="mm")


class Clip:
    def __init__(self, path, start=0.0):
        self.reader = imageio.get_reader(path)
        self.fps = self.reader.get_meta_data()["fps"]
        self.n = self.reader.count_frames()
        self.start = start

    def frame(self, t):
        k = min(self.n - 1, int((self.start + t) * self.fps))
        return Image.fromarray(self.reader.get_data(k))


def wrap(text, font, width):
    """Split text into the fewest lines that fit ``width``, balancing line lengths."""
    words = text.split()
    for n in range(1, len(words) + 1):
        target = len(text) / n
        lines, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > target * 1.15:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        if len(lines) <= n and all(font.getlength(line) <= width for line in lines):
            return lines
    return [text]


def card(fonts, title, sub=None, extra=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f = fonts(64, 500)
    lines = wrap(title, f, W - 320)
    y = H // 2 - 40 - (len(lines) - 1) * 40
    for k, line in enumerate(lines):
        d.text((W // 2, y + 80 * k), line, font=f, fill=INK, anchor="mm")
    y += 80 * (len(lines) - 1)
    if sub:
        d.text((W // 2, y + 92), sub, font=fonts(34), fill=INK2, anchor="mm")
    if extra:
        d.text((W // 2, y + 150), extra, font=fonts(28), fill=INK2, anchor="mm")
    return img


def caption(img, fonts, text):
    """Lower-left caption: white rounded box with dark text."""
    d = ImageDraw.Draw(img, "RGBA")
    f = fonts(34)
    x0, y0 = 64, H - 64
    bbox = d.textbbox((0, 0), text, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.rounded_rectangle((x0, y0 - h - 40, x0 + w + 48, y0), radius=12, fill=(255, 255, 255, 236))
    d.text((x0 + 24, y0 - 20 - h / 2), text, font=f, fill=INK, anchor="lm")
    return img


def section_title(img, fonts, text):
    d = ImageDraw.Draw(img)
    d.text((96, 92), text, font=fonts(46, 500), fill=INK, anchor="lm")
    return img


def grid_frame(fonts, clips, t):
    img = Image.new("RGB", (W, H), BG)
    img = section_title(img, fonts, "Six scenario types, ten difficulty levels")
    d = ImageDraw.Draw(img)
    tw, th, gx, gy = 560, 315, 40, 84
    x0, y0 = (W - 3 * tw - 2 * gx) // 2, 170
    for k, (key, clip) in enumerate(clips):
        x, y = x0 + (k % 3) * (tw + gx), y0 + (k // 3) * (th + gy)
        img.paste(fit(clip.frame(t), tw, th), (x, y))
        d.rounded_rectangle((x, y, x + tw, y + th), radius=10, outline=RULE, width=2)
        bullet(d, x + 18, y + th + 34, 16, key, fonts(18, 700))
        d.text((x + 46, y + th + 35), TITLE[key], font=fonts(30, 500), fill=INK, anchor="lm")
    return img


def cloud_frames(path, n, size):
    """Rotating render of a MID-360 point cloud on white (people in orange)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.fromfile(path, dtype=np.int16).reshape(-1, 4)
    p, person = a[:, :3] / 100.0, a[:, 3] > 0
    keep = (np.abs(p[:, 0]) < 8) & (np.abs(p[:, 1]) < 8)
    p, person = p[keep], person[keep]
    rng = np.random.default_rng(0)
    sub = rng.random(len(p)) < 0.6
    p, person = p[sub], person[sub]
    z = np.clip((p[:, 2] + 0.1) / 2.6, 0, 1)
    env = np.stack([0.55 - 0.35 * z, 0.6 - 0.32 * z, 0.66 - 0.28 * z], 1)
    colors = np.where(person[:, None], np.array([[0.91, 0.44, 0.04]]), env)
    frames = []
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    for k in range(n):
        fig.clf()
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, s=0.6, depthshade=False, linewidths=0)
        ax.plot([0, 0], [0, 0], [0, 1.25], color="black", linewidth=3)
        ax.set_xlim(-5, 7); ax.set_ylim(-6, 6); ax.set_zlim(-0.5, 3.5)  # noqa: E702
        ax.set_box_aspect((12, 12, 4), zoom=1.85)
        ax.view_init(elev=26, azim=-130 + 40 * k / max(1, n - 1))
        ax.set_axis_off()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)
    return frames


def results_chart(fonts_dir, results, size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for w in (400, 500):
        font_manager.fontManager.addfont(os.path.join(fonts_dir, f"GoogleSans-{w}.ttf"))
    plt.rcParams["font.family"] = "Google Sans"
    keys = list(TITLE)
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    colors = {"easy": "#aecbfa", "medium": "#4285f4", "hard": "#174ea6"}
    bh = 0.26
    for j, lv in enumerate(["easy", "medium", "hard"]):
        vals = [results[k][lv]["goal"] * 100 for k in keys]
        ys = np.arange(len(keys)) + (j - 1) * bh
        ax.barh(ys, vals, height=bh * 0.92, color=colors[lv], label=lv)
        for y, v in zip(ys, vals):
            ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=15, color="#202124")
    ax.set_yticks(np.arange(len(keys)), [TITLE[k] for k in keys], fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=15)
    ax.xaxis.grid(True, color="#dadce0")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dadce0")
    ax.tick_params(colors="#5f6368", length=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=16, frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


POLICY_TITLE = {"teacher": "Teacher (privileged state)", "lidar": "Lidar student", "camera": "Camera student"}


def by_difficulty(per_scenario):
    """Success rate per difficulty bucket over all scenario types (episode-weighted)."""
    out = {}
    for lv in ("easy", "medium", "hard"):
        eps = sum(per_scenario[k][lv]["eps"] for k in TITLE)
        out[lv] = sum(per_scenario[k][lv]["goal"] * per_scenario[k][lv]["eps"] for k in TITLE) / max(eps, 1)
    return out


def policies_chart(fonts_dir, policies, summary, size):
    """Success rate per policy: all tiles and per difficulty bucket."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for w in (400, 500):
        font_manager.fontManager.addfont(os.path.join(fonts_dir, f"GoogleSans-{w}.ttf"))
    plt.rcParams["font.family"] = "Google Sans"
    keys = [k for k in POLICY_TITLE if k in policies]
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    colors = {"all": "#5f6368", "easy": "#aecbfa", "medium": "#4285f4", "hard": "#174ea6"}
    bh = 0.2
    for j, lv in enumerate(["all", "easy", "medium", "hard"]):
        vals = [100 * (summary[k]["goal"] if lv == "all" else by_difficulty(policies[k])[lv]) for k in keys]
        ys = np.arange(len(keys)) + (j - 1.5) * bh
        ax.barh(ys, vals, height=bh * 0.9, color=colors[lv], label="all tiles" if lv == "all" else lv)
        for y, v in zip(ys, vals):
            ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=15, color="#202124")
    ax.set_yticks(np.arange(len(keys)), [POLICY_TITLE[k] for k in keys], fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=15)
    ax.xaxis.grid(True, color="#dadce0")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dadce0")
    ax.tick_params(colors="#5f6368", length=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=16, frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--renders", required=True)
    parser.add_argument("--fonts", required=True)
    parser.add_argument("--results", default=os.path.join(DOCS, "data", "results.json"))
    parser.add_argument("--gait_before", default=None, help="Clip of the earlier teacher (clamped commands).")
    parser.add_argument("--gait_after", default=None, help="Clip of the final teacher, same scenario and seed.")
    parser.add_argument("--out", default=os.path.join(DOCS, "media", "overview.mp4"))
    args = parser.parse_args()
    fonts = Fonts(args.fonts)
    data = json.load(open(args.results))
    policies, summary = data["policies"], data["summary"]
    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=None, macro_block_size=8, pixelformat="yuv420p",
                                ffmpeg_params=["-crf", "23", "-preset", "slow", "-movflags", "+faststart"])
    last = [None]

    def emit(img):
        writer.append_data(np.asarray(img))
        last[0] = img

    def segment(duration, render):
        """Render a segment; the first 0.4 s cross-fade from the previous segment's last frame."""
        n = int(duration * FPS)
        prev = last[0]
        for k in range(n):
            img = render(k / FPS)
            if prev is not None and k < int(0.4 * FPS):
                img = Image.blend(prev, img, (k + 1) / int(0.4 * FPS))
            emit(img)

    # title
    title = card(fonts, "G1 Patrol: Humanoid Navigation in Dense Urban Pedestrian Environments",
                 "Mehmet Kerem Turkcan", "Unitree G1 · Isaac Lab 3.0 · Isaac Sim 6.1 · simulation")
    segment(4.0, lambda t: title)

    # scenario grid
    grid_clips = [(key, Clip(os.path.join(args.renders, clip), start)) for key, clip, start, _ in SHOTS]
    segment(6.0, lambda t: grid_frame(fonts, grid_clips, t))

    # full-frame shots with captions
    for key, clip, start, cap in SHOTS:
        c = Clip(os.path.join(args.renders, clip), start + 3.0)
        segment(5.0, lambda t, c=c, cap=cap: caption(fit(c.frame(t), W, H), fonts, cap))

    # velocity commands: earlier teacher (clamped) and final teacher side by side
    if args.gait_before and args.gait_after and "teacher_v2" in summary:
        before, after = Clip(args.gait_before, 1.0), Clip(args.gait_after, 1.0)
        m0, m1 = summary["teacher_v2"]["motion"], summary["teacher"]["motion"]
        pct = lambda v: f"{100 * v:.0f}%"
        lines = [
            ("Lateral command above 0.25 m/s", pct(m0["fraction_lateral_above_0.25mps"]), pct(m1["fraction_lateral_above_0.25mps"])),
            ("Command at a range limit", pct(m0["fraction_saturated"]), pct(m1["fraction_saturated"])),
            ("Yaw-rate command change", f'{m0["yaw_rate_command_change_radps2"]:.1f} rad/s²', f'{m1["yaw_rate_command_change_radps2"]:.1f} rad/s²'),
            ("Falls", f'{100 * summary["teacher_v2"]["fall"]:.1f}%', f'{100 * summary["teacher"]["fall"]:.1f}%'),
        ]

        def gait(t):
            img = section_title(Image.new("RGB", (W, H), BG), fonts, "Velocity commands: clamped (left) and smoothed (right)")
            d = ImageDraw.Draw(img)
            pw, ph = 880, 495
            for k, (clip, label) in enumerate(((before, "Earlier teacher: clamped commands"), (after, "Final teacher: tanh range, low-pass filter"))):
                x = 64 + k * (pw + 48)
                img.paste(fit(clip.frame(t), pw, ph), (x, 170))
                d.rounded_rectangle((x, 170, x + pw, 170 + ph), radius=10, outline=RULE, width=2)
                d.text((x, 170 + ph + 36), label, font=fonts(28, 500), fill=INK, anchor="lm")
            y = 170 + ph + 100
            for name, a, b in lines:
                d.text((64, y), name, font=fonts(26), fill=INK2, anchor="lm")
                d.text((64 + pw, y), a, font=fonts(26, 500), fill=INK, anchor="rm")
                d.text((64 + 2 * pw + 48, y), b, font=fonts(26, 500), fill=INK, anchor="rm")
                y += 46
            return img

        segment(9.0, gait)

    # sensing
    cloud = cloud_frames(os.path.join(DOCS, "data", "cloud_crosswalk.bin"), int(7 * FPS), (1100, 820))
    lidar = Image.open(os.path.join(DOCS, "media", "sensor_lidar_accumulated.png")).convert("RGB").resize((330, 330), Image.NEAREST)
    depth = Image.open(os.path.join(DOCS, "media", "sensor_depth.png")).convert("RGB").resize((340, 202), Image.NEAREST)
    depth_map = Image.open(os.path.join(DOCS, "media", "sensor_depth_map.png")).convert("RGB").resize((202, 202), Image.NEAREST)

    def sensing(t):
        img = section_title(Image.new("RGB", (W, H), BG), fonts, "Onboard sensing")
        d = ImageDraw.Draw(img)
        img.paste(cloud[min(len(cloud) - 1, int(t * FPS))], (40, 190))
        d.text((96, 1030), "Simulated Livox MID-360 returns around the robot (pedestrians in orange)", font=fonts(26), fill=INK2, anchor="lm")
        img.paste(lidar, (1240, 190))
        d.text((1240, 548), "Lidar student: height map accumulated over 3 s", font=fonts(24), fill=INK2, anchor="lm")
        img.paste(depth, (1240, 640))
        img.paste(depth_map, (1600, 640))
        d.text((1240, 878), "Camera student: depth images and height map", font=fonts(24), fill=INK2, anchor="lm")
        return img

    segment(7.0, sensing)

    # figures with a slow zoom
    for name, heading in (("architecture", "Policy architecture"), ("workflow", "Training workflow")):
        fig = Image.open(os.path.join(DOCS, "figures", f"{name}.png")).convert("RGB")

        def figure(t, fig=fig, heading=heading):
            img = section_title(Image.new("RGB", (W, H), BG), fonts, heading)
            z = 1.0 + 0.03 * t / 6.0
            f = contain(fig, int(1760 * z), int(860 * z))
            box = Image.new("RGB", (1760, 860), BG)
            box.paste(f, ((1760 - f.width) // 2, (860 - f.height) // 2))
            img.paste(box, (80, 170))
            return img

        segment(6.5, figure)

    # results
    chart = results_chart(args.fonts, policies["teacher"], (1500, 800))
    te = summary["teacher"]
    note = (f"Teacher, {te['episodes']:,} evaluation episodes on 120 tiles: {100 * te['goal']:.1f}% success, "
            f"{100 * te['ped']:.1f}% pedestrian contact, {100 * te['obst']:.1f}% obstacle contact, {100 * te['fall']:.1f}% falls")

    def results_frame(t):
        img = section_title(Image.new("RGB", (W, H), BG), fonts, "Teacher: success rate by scenario and difficulty")
        img.paste(chart, (210, 160))
        ImageDraw.Draw(img).text((96, 1010), note, font=fonts(26), fill=INK2, anchor="lm")
        return img

    segment(6.0, results_frame)

    if len(policies) > 1:
        pchart = policies_chart(args.fonts, policies, summary, (1500, 760))

        def policies_frame(t):
            img = section_title(Image.new("RGB", (W, H), BG), fonts, "Teacher and students: success rate")
            img.paste(pchart, (210, 170))
            ImageDraw.Draw(img).text((96, 1010), "Students observe proprioception, the goal and one onboard sensor; all policies are evaluated on the same 120 tiles",
                                     font=fonts(26), fill=INK2, anchor="lm")
            return img

        segment(6.0, policies_frame)

    end = card(fonts, "github.com/mkturkcan/g1_patrol", "Code, configurations and evaluation scripts")
    segment(3.5, lambda t: end)
    writer.close()
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
