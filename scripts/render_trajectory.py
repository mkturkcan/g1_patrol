"""Render a trajectory saved by ``scripts/evaluate.py`` as a stick-figure video.

This needs neither Isaac Sim nor a GPU, so it is a quick, lightweight look at a policy (RTX videos come from
``evaluate.py --video`` or ``isaaclab play --video``). The camera follows the robot. The green arrow is the commanded planar velocity and the blue arrow is the
actual one.

Usage:
    python scripts/render_trajectory.py logs/rsl_rl/g1_walk_flat/<run>/eval/trajectory_env0.npz
"""

import argparse
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=str, help="Path to trajectory_env*.npz")
    parser.add_argument("--output", type=str, default=None, help="Output .mp4 (default: next to the input).")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate (frames are subsampled to match).")
    parser.add_argument("--max_seconds", type=float, default=20.0, help="Maximum duration of simulated time.")
    parser.add_argument("--view_radius", type=float, default=1.0, help="Half-width of the view around the robot (m).")
    args = parser.parse_args()

    data = np.load(args.trajectory)
    body_pos, root_pos, root_yaw = data["body_pos"], data["root_pos"], data["root_yaw"]
    command, lin_vel, ang_vel = data["command"], data["lin_vel_yaw"], data["ang_vel_z"]
    link_names, parents = list(data["link_names"]), data["parent_indices"]
    falls = set(data["falls"].tolist())
    dt = float(data["dt"])

    stride = max(1, round(1.0 / (dt * args.fps)))
    num_steps = min(len(body_pos), int(args.max_seconds / dt))
    output = args.output or os.path.splitext(args.trajectory)[0] + ".mp4"

    finger_tokens = ("zero", "one", "two", "three", "four", "five", "six")
    is_finger = np.array([any(t in n for t in finger_tokens) for n in link_names])
    is_foot = np.array(["ankle_roll" in n for n in link_names])
    edges = [(i, p) for i, p in enumerate(parents) if 0 <= p < len(link_names)]

    fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    ax = fig.add_subplot(projection="3d")
    r = args.view_radius
    z_max = 1.5
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=8) as writer:
        for t in range(0, num_steps, stride):
            ax.cla()
            cx, cy = root_pos[t, 0], root_pos[t, 1]
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_zlim(0.0, z_max)
            ax.set_box_aspect((2 * r, 2 * r, z_max), zoom=1.6)
            ax.view_init(elev=15, azim=-60)
            ax.set_axis_off()

            # ground grid with 0.5 m spacing, fixed in the world so motion is visible
            for g in np.arange(np.floor((cx - r) * 2) / 2, cx + r, 0.5):
                ax.plot([g, g], [cy - r, cy + r], [0, 0], color="0.8", lw=0.6)
            for g in np.arange(np.floor((cy - r) * 2) / 2, cy + r, 0.5):
                ax.plot([cx - r, cx + r], [g, g], [0, 0], color="0.8", lw=0.6)

            # pelvis path over the last 3 s
            t0 = max(0, t - int(3.0 / dt))
            ax.plot(root_pos[t0 : t + 1, 0], root_pos[t0 : t + 1, 1], 0.0, color="tab:purple", lw=1.5, alpha=0.6)

            # skeleton
            pos = body_pos[t]
            for i, p in edges:
                lw = 0.8 if is_finger[i] else 3.0
                ax.plot(*zip(pos[i], pos[p]), color="0.2", lw=lw, solid_capstyle="round")
            ax.scatter(*pos[is_foot].T, color="tab:orange", s=30, depthshade=False)

            # commanded (green) and actual (blue) planar velocity, drawn from the ground below the pelvis
            yaw = root_yaw[t]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            for vel, color in ((command[t, :2], "tab:green"), (lin_vel[t], "tab:blue")):
                v = rot @ vel
                ax.quiver(cx, cy, 0.02, v[0], v[1], 0.0, color=color, lw=2.5, arrow_length_ratio=0.25)

            fell = any(f in falls for f in range(t, t + stride))
            ax.set_title(
                f"t = {t * dt:5.2f} s{'   FELL' if fell else ''}\n"
                f"cmd  vx={command[t, 0]:+.2f}  vy={command[t, 1]:+.2f}  wz={command[t, 2]:+.2f}\n"
                f"act  vx={lin_vel[t, 0]:+.2f}  vy={lin_vel[t, 1]:+.2f}  wz={ang_vel[t]:+.2f}",
                family="monospace",
                fontsize=10,
                loc="left",
            )
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
