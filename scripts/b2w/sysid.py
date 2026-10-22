"""System identification of the B2-W's sport-mode controller: does the real robot follow velocity commands as the
controller model the policy was trained with (:mod:`g1_patrol.controller_model`, ranges in
:data:`g1_patrol.b2w.CONTROLLER`)?

1. ``profile``: write a command profile (steps, small commands near the deadband, turns, arcs, sinusoids; about 2.5
   minutes, back and forth so the robot stays within a few meters of where it starts);
2. play it on the robot with the deployment node (``mode:=sysid``), which logs commands and odometry;
3. ``fit``: fit latency, gain, deadband, time constant, acceleration bound per axis and the yaw drift, and compare
   them with the training ranges. Values outside the ranges call for retraining with widened ranges.

``selftest`` fits a log simulated from known parameters (with noise) and checks that they are recovered.

Examples::

    python scripts/b2w/sysid.py profile --out profile.csv --scale 0.8
    python scripts/b2w/sysid.py fit --log sysid_log.csv
    python scripts/b2w/sysid.py selftest
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math

import numpy as np
import torch

from g1_patrol.b2w import CONTROLLER
from g1_patrol.controller_model import simulate

DT = 0.02
AXES = ("forward", "lateral", "yaw rate")


def make_profile(scale: float = 1.0) -> np.ndarray:
    """Rows (t, vx, vy, wz): piecewise-constant commands at 50 Hz."""
    segments = []  # (duration, vx, vy, wz)

    def hold(d, vx=0.0, vy=0.0, wz=0.0):
        segments.append((d, vx * scale, vy * scale, wz * scale))

    hold(2.0)
    for v, d in ((0.3, 3.0), (0.6, 2.5), (1.0, 2.0)):  # forward steps, each driven back
        hold(d, vx=v), hold(2.0), hold(d, vx=-min(v, 0.8)), hold(2.0)
    for v in (0.05, 0.1, -0.1):  # near the deadband
        hold(3.0, vx=v), hold(2.0)
    for w, d in ((0.5, 3.0), (-0.5, 3.0), (1.0, 2.0), (-1.0, 2.0), (0.1, 3.0), (-0.1, 3.0)):  # turning in place
        hold(d, wz=w), hold(2.0)
    for v in (0.2, -0.2):  # sidesteps
        hold(2.0, vy=v), hold(2.0)
    hold(4.0, vx=0.5, wz=0.5), hold(2.0), hold(4.0, vx=-0.4, wz=-0.5), hold(2.0)  # arcs
    rows, t = [], 0.0
    for d, vx, vy, wz in segments:
        for _ in range(int(round(d / DT))):
            rows.append((t, vx, vy, wz))
            t += DT
    for duration, axis, amp, freq in ((10.0, 0, 0.4, 0.3), (8.0, 2, 0.8, 0.5)):  # sinusoids
        for k in range(int(duration / DT)):
            c = [0.0, 0.0, 0.0]
            c[axis] = amp * scale * math.sin(2 * math.pi * freq * k * DT)
            rows.append((t, *c))
            t += DT
        for _ in range(int(2.0 / DT)):
            rows.append((t, 0.0, 0.0, 0.0))
            t += DT
    return np.array(rows)


def load_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Commands and measured body velocities resampled to 50 Hz: (T, 3) each."""
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    cmd = np.array([[float(r["cmd_vx"]), float(r["cmd_vy"]), float(r["cmd_wz"])] for r in rows])
    vel = np.array([[float(r["vx"]), float(r["vy"]), float(r["wz"])] for r in rows])
    grid = np.arange(t[0], t[-1], DT)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    vel_i = np.stack([np.interp(grid, t, vel[:, k]) for k in range(3)], axis=1)
    return cmd[idx], vel_i


def _fit_axis(cmd: torch.Tensor, vel: torch.Tensor, axis: int, delay: int, grids: dict, fixed: dict) -> tuple[float, dict]:
    """Grid search over one axis's parameters at a given delay (the other axes use ``fixed``)."""
    combos = list(itertools.product(grids["gain"], grids["tau"], grids["accel"], grids["deadband"], grids.get("drift", [0.0])))
    errs = []
    for start in range(0, len(combos), 2048):  # in chunks, to bound memory
        p = torch.tensor(combos[start : start + 2048], dtype=torch.float32)
        n = len(p)
        gain, tau, accel, deadband = (fixed[k].repeat(n, 1) for k in ("gain", "tau", "accel", "deadband"))
        gain[:, axis], tau[:, axis], accel[:, axis], deadband[:, axis] = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        drift = p[:, 4] if "drift" in grids else fixed["drift"].repeat(n)
        out = simulate(cmd.expand(n, -1, -1), DT, torch.full((n,), delay), gain, deadband, tau, accel, drift)
        errs.append(((out[..., axis] - vel[None, :, axis]) ** 2).mean(dim=1))
    err = torch.cat(errs)
    k = int(err.argmin())
    best = {"gain": combos[k][0], "tau": combos[k][1], "accel": combos[k][2], "deadband": combos[k][3], "drift": combos[k][4]}
    return float(err[k]), best


def fit(cmd_np: np.ndarray, vel_np: np.ndarray, max_delay: int = 12, verbose: bool = True) -> dict:
    cmd = torch.tensor(cmd_np, dtype=torch.float32)[None]
    vel = torch.tensor(vel_np, dtype=torch.float32)
    coarse = {"gain": np.linspace(0.6, 1.3, 8), "tau": [0.05, 0.12, 0.2, 0.3, 0.45, 0.7], "accel": [0.5, 1.0, 2.0, 4.0, 8.0],
              "deadband": [0.0]}
    fixed = {k: torch.tensor([[v] * 3], dtype=torch.float32) for k, v in (("gain", 1.0), ("tau", 0.2), ("accel", 3.0), ("deadband", 0.0))}
    fixed["drift"] = torch.zeros(1)
    # latency: the delay with the smallest error over the three axes (coarse grids)
    errors = []
    for d in range(max_delay + 1):
        total = sum(_fit_axis(cmd, vel, a, d, coarse, fixed)[0] / max(float(vel[:, a].var()), 1e-6) for a in range(3))
        errors.append(total)
    delay = int(np.argmin(errors))
    # per axis, finer grids at that delay (forward and lateral first, then the yaw rate with the drift)
    fine = {"gain": np.linspace(0.5, 1.3, 17), "tau": np.geomspace(0.04, 0.8, 14), "accel": np.geomspace(0.4, 10.0, 12),
            "deadband": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]}
    result = {"delay_s": delay * DT}
    for a in (0, 1, 2):
        err, best = _fit_axis(cmd, vel, a, delay, fine, fixed)
        for k in ("gain", "tau", "accel", "deadband"):
            fixed[k][0, a] = best[k]
        if a == 2:  # the yaw drift, then the yaw parameters again with it
            near = {k: [best[k]] for k in ("gain", "tau", "accel", "deadband")} | {"drift": np.linspace(-0.1, 0.1, 21)}
            fixed["drift"][0] = _fit_axis(cmd, vel, a, delay, near, fixed)[1]["drift"]
            err, best = _fit_axis(cmd, vel, a, delay, fine, fixed)
            for k in ("gain", "tau", "accel", "deadband"):
                fixed[k][0, a] = best[k]
        result[AXES[a]] = {k: best[k] for k in ("gain", "tau", "accel", "deadband")} | {"rmse": math.sqrt(err)}
    result["drift"] = float(fixed["drift"][0])
    # residual noise
    out = simulate(cmd, DT, torch.tensor([delay]), fixed["gain"], fixed["deadband"], fixed["tau"], fixed["accel"], fixed["drift"])[0]
    resid = (vel - out).numpy()
    for a in range(3):
        result[AXES[a]]["noise"] = float(resid[:, a].std())
    if verbose:
        report(result)
    return result


def report(result: dict):
    c = CONTROLLER
    def check(value, lo, hi):
        return "ok" if lo <= value <= hi else "OUTSIDE"
    d = result["delay_s"]
    print(f"latency {d:.2f} s   (training {c['delay_range'][0]:.2f}-{c['delay_range'][1]:.2f}: {check(d, *c['delay_range'])})")
    for a, name in enumerate(AXES):
        r = result[name]
        print(f"{name:9s} gain {r['gain']:.2f} ({check(r['gain'], *c['gain_range'][a])})  tau {r['tau']:.2f} s "
              f"({check(r['tau'], *c['tau_range'][a])})  accel {r['accel']:.1f} ({check(r['accel'], *c['accel_range'][a])})  "
              f"deadband {r['deadband']:.2f} ({check(r['deadband'], 0.0, c['deadband_max'][a])})  noise {r['noise']:.3f} "
              f"({check(r['noise'], 0.0, c['noise_max'][a])})  rmse {r['rmse']:.3f}")
    print(f"yaw drift {result['drift']:+.3f} rad/s per m/s ({check(abs(result['drift']), 0.0, c['drift_max'])})")


def selftest(seed: int = 0) -> bool:
    """Simulate the profile with known parameters and velocity noise, fit it, compare."""
    torch.manual_seed(seed)
    prof = make_profile()
    cmd = torch.tensor(prof[:, 1:], dtype=torch.float32)[None]
    true = {"delay": 5, "gain": [0.92, 0.8, 1.05], "tau": [0.3, 0.25, 0.15], "accel": [2.0, 1.2, 4.0], "deadband": [0.04, 0.02, 0.05], "drift": 0.03}
    vel = simulate(cmd, DT, torch.tensor([true["delay"]]), torch.tensor([true["gain"]]), torch.tensor([true["deadband"]]),
                   torch.tensor([true["tau"]]), torch.tensor([true["accel"]]), torch.tensor([true["drift"]]))[0]
    vel = vel + 0.02 * torch.randn_like(vel)
    result = fit(prof[:, 1:], vel.numpy())
    ok = abs(result["delay_s"] - true["delay"] * DT) <= DT + 1e-9
    for a, name in enumerate(AXES):
        r = result[name]
        ok &= abs(r["gain"] - true["gain"][a]) < 0.08
        ok &= abs(r["tau"] - true["tau"][a]) / true["tau"][a] < 0.35
    ok &= abs(result["drift"] - true["drift"]) < 0.03
    print("selftest", "passed" if ok else "FAILED")
    return bool(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--out", default="profile.csv")
    p.add_argument("--scale", type=float, default=1.0, help="scale the command amplitudes (start below 1)")
    f = sub.add_parser("fit")
    f.add_argument("--log", required=True)
    f.add_argument("--max_delay", type=int, default=12, help="largest latency tried, in 20 ms steps")
    f.add_argument("--json", default=None, help="also write the fit here (for scripts/b2w/evaluate.py --controller sysid:<json>)")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.cmd == "profile":
        prof = make_profile(args.scale)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "vx", "vy", "wz"])
            w.writerows([[f"{v:.3f}" for v in row] for row in prof])
        print(f"{len(prof)} commands, {prof[-1, 0]:.0f} s -> {args.out}")
    elif args.cmd == "fit":
        cmd, vel = load_log(args.log)
        result = fit(cmd, vel, args.max_delay)
        if args.json:
            import json

            with open(args.json, "w") as fh:
                json.dump(result, fh, indent=2)
    else:
        raise SystemExit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
