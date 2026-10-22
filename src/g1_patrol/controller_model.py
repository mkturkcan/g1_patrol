"""Model of how a legged robot's built-in controller tracks body velocity commands -- shared by simulation and
system identification.

Per axis (forward, lateral, yaw rate) the command passes through, in order:

1. a latency of ``delay`` steps;
2. a deadband: commands smaller than ``deadband`` are ignored;
3. a gain: the steady-state velocity is ``gain`` times the command; the yaw rate also drifts by ``drift`` times the
   forward velocity (rad/s per m/s), as with unevenly slipping wheels;
4. a first-order response with time constant ``tau``, whose change per step is bounded by ``accel`` * dt.

Pure torch (no simulator imports): :class:`g1_patrol.tasks.navigation.mdp.BaseVelocityAction` steps it in
simulation, ``scripts/b2w/sysid.py`` fits it to logs of the real robot.
"""

from __future__ import annotations

import torch


def track(
    tracked: torch.Tensor,
    command: torch.Tensor,
    dt: float,
    gain: torch.Tensor,
    deadband: torch.Tensor,
    tau: torch.Tensor,
    accel: torch.Tensor,
    drift: torch.Tensor,
) -> torch.Tensor:
    """One step of the tracking response (after the latency): the new velocities (n, 3) from the current ones
    ``tracked`` (n, 3) and the (delayed) ``command`` (n, 3); parameters broadcast to (n, 3), ``drift`` to (n,)."""
    target = torch.where(command.abs() < deadband, torch.zeros_like(command), command) * gain
    target = torch.cat([target[:, :2], target[:, 2:] + (drift * tracked[:, 0])[:, None]], dim=1)
    step = (target - tracked) * (1.0 - torch.exp(-dt / tau))
    return tracked + torch.maximum(torch.minimum(step, accel * dt), -accel * dt)


def simulate(
    commands: torch.Tensor,
    dt: float,
    delay: torch.Tensor,
    gain: torch.Tensor,
    deadband: torch.Tensor,
    tau: torch.Tensor,
    accel: torch.Tensor,
    drift: torch.Tensor,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Velocities (n, T, 3) in response to command sequences (n, T, 3) sampled every ``dt``, for ``n`` parameter
    sets (``delay`` (n,) in steps, the others as in :func:`track`). Before the first command the robot holds
    ``initial`` (default: rest) and the delayed command is zero."""
    n, steps, _ = commands.shape
    envs = torch.arange(n, device=commands.device)
    padded = torch.cat([torch.zeros_like(commands), commands], dim=1)  # index t + steps - delay
    tracked = torch.zeros(n, 3, device=commands.device) if initial is None else initial.clone()
    out = torch.empty_like(commands)
    for t in range(steps):
        tracked = track(tracked, padded[envs, t + steps - delay.long()], dt, gain, deadband, tau, accel, drift)
        out[:, t] = tracked
    return out
