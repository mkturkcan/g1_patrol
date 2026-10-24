"""Safety shield for velocity commands: the robot never drives into what its sensors see, keeps more room from tall
things (people) the faster it goes, and slows down early for things that move towards it -- independently of the
learned policy.

It runs after the policy on the robot (``deploy/b2w/navigator.py``) and, to measure what it prevents and what it
costs, in simulation (``scripts/b2w/evaluate.py --shield``). Pure torch, batched over robots.

**Admissible commands.** A command is admissible if the robot, executing it, can still stop before coming closer to
anything than its margin -- the admissible-velocity test of mobile robotics, combined with the speed-and-separation
principle of human-robot collaboration (ISO/TS 15066): the robot keeps the commanded twist (forward, lateral, yaw rate;
a circular arc) for the reaction time ``latency``, then decelerates at ``brake`` (``brake_yaw``) to rest;
``speed_factor`` allows for the controller moving faster than commanded and ``drift`` for it straying from the arc. The
footprint (a rectangle with rounded corners) is swept along this stopping path, and the command is scaled down
(bisection) until the robot comes to rest before any return breaches its margin:

* ``margin`` for obstacles; ``margin + drop_extra`` for drops (returns more than ``drop_z`` below the floor);
* for returns in a column taller than ``tall_z`` (people, but also machines, shelves, walls) a margin growing with
  speed, from ``margin`` at ``slow_speed`` to ``person_margin`` at ``fast_speed`` -- so the robot passes through
  doors slowly and gives people room when it is fast;
* returns that a person detector marks (``person``) keep ``person_margin`` at any speed;
* returns that ``moving`` marks (the caller compares the current scan with the previous one, see
  :func:`moving_returns`) are assumed to come towards the robot at ``human_speed`` while it moves towards them (not
  while it moves away: a retreating robot pushes no one).

A return already within its margin only blocks motion that brings the robot closer to it, so the robot can always
back or turn away. If the robot's measured velocity alone already breaches (something stepped in front of it), the
command is zero.

**Approach speed.** The closing speed towards a tall return at gap ``g`` may not exceed the speed from which the robot
can still slow to ``approach_speed`` at gap ``approach_gap``; driving along a wall or through a door closes little and
is not slowed by this.

Returns lower than ``z_min`` above the floor (the floor, cables, mats) and higher than ``z_max`` (ceilings) are
ignored; returns inside the footprint are the robot itself. The shield sees only what the sensors see: pass the
accumulated height map's obstacle cells as further points (:func:`map_points`) to remember low objects beside the
robot that left the sensors' view.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShieldCfg:
    half_length: float = 0.56
    half_width: float = 0.29
    corner_radius: float = 0.12
    """Footprint (m), as in training."""
    ground_z: float = -0.55
    """Height of the floor in the base frame (m): minus the base height."""
    z_min: float = 0.06
    z_max: float = 1.9
    drop_z: float = -0.15
    """Obstacle returns lie between ``z_min`` and ``z_max`` above the floor; returns lower than ``drop_z`` are drops."""
    margin: float = 0.1
    drop_extra: float = 0.15
    """Gaps (m) kept to obstacles, and additionally to drops."""
    tall_z: float = 1.2
    """A return higher than this above the floor makes its cell (``cell`` m) tall."""
    cell: float = 0.25
    person_margin: float = 0.45
    slow_speed: float = 0.25
    fast_speed: float = 0.75
    """The gap kept to tall things grows from ``margin`` at ``slow_speed`` (m/s) to ``person_margin`` at ``fast_speed``."""
    human_speed: float = 1.4
    """Speed (m/s) at which moving things are assumed to approach while the robot moves towards them (a fast walk)."""
    still_for: float = 1.0
    moving_cells: int = 1
    """Filters of the moving returns (see :func:`moving_returns`): a return counts as moving only where nothing tall
    was for ``still_for`` seconds before (the edges of static things flicker between floor and tall from scan to scan),
    and with at least ``moving_cells`` such cells within 0.2 m. In simulation (lab mixture, 0.5 m/s) the first removes
    about four fifths of the moving returns at furniture and a quarter of the stops, without more pedestrian contacts;
    requiring 2 cells also misses people who stand and then walk off."""
    approach_speed: float = 0.25
    approach_gap: float = 0.8
    """Closing speed (m/s) towards a tall thing allowed within ``approach_gap`` (m); more only as far as the robot
    could still slow down to it."""
    brake: float = 0.8
    brake_yaw: float = 2.0
    """Deceleration (m/s^2, rad/s^2) the shield counts on: below the identified controller's bounds, allowing for its
    first-order lag (``scripts/b2w/sysid.py``)."""
    latency: float = 0.35
    """Reaction time (s): sensor age, the policy step and the controller's latency."""
    speed_factor: float = 1.12
    """The controller may move this much faster than commanded (its gain)."""
    drift: float = 0.08
    """The robot may stray this far (m) from the commanded arc per meter driven (yaw drift, velocity noise)."""
    samples: int = 32
    """Poses checked along the stopping path."""
    max_points: int = 4096
    """Returns considered per robot: the closest ones."""


def map_points(height_map: torch.Tensor, resolution: float, ground_z: float, min_height: float = 0.1) -> torch.Tensor:
    """Obstacle cells of accumulated height maps (n, H, W) (heading frame, robot-centred, heights above the floor,
    unknown < -0.99) as points (n, H * W, 3) in the base frame (+inf for other cells)."""
    n, h, w = height_map.shape
    c = (torch.arange(h, device=height_map.device, dtype=torch.float) + 0.5) * resolution - h * resolution / 2
    x, y = torch.meshgrid(c, c, indexing="ij")
    occupied = height_map > min_height
    pts = torch.stack([x.expand(n, h, w), y.expand(n, h, w), height_map + ground_z], dim=-1).view(n, -1, 3)
    return torch.where(occupied.view(n, -1, 1), pts, torch.full_like(pts, float("inf")))


def moving_returns(points: torch.Tensor, height_map: torch.Tensor, resolution: float, ground_z: float,
                   appeared: float = 0.5, floor: float = 0.25, tall_age: torch.Tensor | None = None,
                   recent: float = 1.0, min_cells: int = 1) -> torch.Tensor:
    """Which returns (n, k, 3; base frame) lie in cells of the height map (n, 3, H, W: accumulated, current scan,
    previous scan; motion-compensated) that the previous scan saw as floor and the current scan as something taller
    than ``appeared``: the leading edges of things that move. With ``tall_age`` (n, H, W; seconds since each cell last
    held something tall, :attr:`g1_patrol.heightmap.HeightMap.tall_age_out`), only cells that held nothing tall for
    ``recent`` seconds count: the edges of static things flicker between floor and tall from scan to scan. With
    ``min_cells`` > 1, a return counts only if that many such cells lie within 0.2 m: a person walking into open floor
    fills several, a stray cell at the edge of something static one."""
    n, _, h, w = height_map.shape
    i = ((points[..., 0] + h * resolution / 2) / resolution).floor().long()
    j = ((points[..., 1] + w * resolution / 2) / resolution).floor().long()
    inside = (i >= 0) & (i < h) & (j >= 0) & (j < w) & torch.isfinite(points).all(dim=-1)
    flat = (i.clamp(0, h - 1) * w + j.clamp(0, w - 1))
    current = torch.gather(height_map[:, 1].reshape(n, -1), 1, flat)
    previous = torch.gather(height_map[:, 2].reshape(n, -1), 1, flat)
    was_floor = (previous > -0.99) & (previous < floor)
    moving = inside & (current > appeared) & was_floor & (points[..., 2] - ground_z > appeared)
    if tall_age is not None:
        moving &= torch.gather(tall_age.reshape(n, -1), 1, flat) > recent
    if min_cells > 1:
        cells = torch.zeros(n, h * w, device=points.device).scatter_add(1, flat, moving.float()) > 0
        k = int(round(0.2 / resolution)) * 2 + 1
        around = torch.nn.functional.avg_pool2d(cells.float().view(n, 1, h, w), k, stride=1, padding=k // 2,
                                                count_include_pad=True) * (k * k)
        moving &= torch.gather(around.view(n, -1), 1, flat) > min_cells - 0.5
    return moving


def _gap(xy: torch.Tensor, core: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gap between points (..., 2) and the footprint (core rectangle grown by ``radius``), the footprint's nearest
    point and the outward direction there."""
    nearest = torch.maximum(torch.minimum(xy, core), -core)
    diff = xy - nearest
    dist = torch.linalg.norm(diff, dim=-1)
    return dist - radius, nearest, diff / dist.clamp(min=1e-6)[..., None]


def _poses(twist: torch.Tensor, progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poses (x, y, yaw), each (n, J), after following body twists (n, 3) for ``progress`` (n, J) seconds."""
    vx, vy, w = twist[:, 0, None], twist[:, 1, None], twist[:, 2, None]
    th = w * progress
    small = th.abs() < 1e-4
    th_safe = torch.where(small, torch.ones_like(th), th)
    sin_c = torch.where(small, 1.0 - th * th / 6.0, torch.sin(th) / th_safe)
    cos_c = torch.where(small, th / 2.0, (1.0 - torch.cos(th)) / th_safe)
    x = (vx * sin_c - vy * cos_c) * progress
    y = (vx * cos_c + vy * sin_c) * progress
    return x, y, th


def shield(points: torch.Tensor, command: torch.Tensor, cfg: ShieldCfg = ShieldCfg(),
           velocity: torch.Tensor | None = None, moving: torch.Tensor | None = None,
           person: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Scale commands (n, 3) to admissible ones, given the latest returns (n, k, 3; base frame; +inf/NaN where there
    is none), the measured velocity (n, 3; body frame), and which returns move or are people (n, k; optional).
    Returns the commands and diagnostics: the scale, whether the measured motion alone was already too fast
    (``emergency``), the smallest gap to an obstacle, to a tall thing and to a moving thing or person."""
    n, dev = command.shape[0], command.device
    pts = torch.nan_to_num(points.float(), nan=float("inf"))
    finite = torch.isfinite(pts).all(dim=-1)
    pts = torch.where(finite[..., None], pts, torch.zeros_like(pts))
    h = pts[..., 2] - cfg.ground_z
    drop = finite & (h < cfg.drop_z)
    obstacle = finite & (((h > cfg.z_min) & (h < cfg.z_max)) | drop)
    r = cfg.corner_radius
    core = torch.tensor([cfg.half_length - r, cfg.half_width - r], device=dev)
    gap0, nearest, normal = _gap(pts[..., :2], core, r)
    obstacle &= gap0 > 0.0  # inside the footprint: the robot itself
    # tall columns: people (and machines, shelves, walls)
    size = 8.0
    cells = int(size / cfg.cell)
    ci = ((pts[..., 0] + size / 2) / cfg.cell).floor().long().clamp(0, cells - 1)
    cj = ((pts[..., 1] + size / 2) / cfg.cell).floor().long().clamp(0, cells - 1)
    flat = ci * cells + cj
    column = torch.full((n, cells * cells), -1e3, device=dev).scatter_reduce(
        1, flat, torch.where(obstacle, h, torch.full_like(h, -1e3)), reduce="amax")
    tall = obstacle & (torch.gather(column, 1, flat) > cfg.tall_z)
    people = obstacle & person if person is not None else torch.zeros_like(obstacle)
    approaching = obstacle & moving if moving is not None else torch.zeros_like(obstacle)
    # the closest returns only
    k = min(cfg.max_points, pts.shape[1])
    order = torch.where(obstacle, gap0, torch.full_like(gap0, 1e3)).topk(k, dim=1, largest=False).indices
    def take(t):
        return torch.gather(t, 1, order) if t.dim() == 2 else torch.gather(t, 1, order[..., None].expand(-1, -1, t.shape[-1]))
    pts, gap0, tall, people, approaching, drop, obstacle, nearest, normal = (
        take(t) for t in (pts, gap0, tall, people, approaching, drop, obstacle, nearest, normal))

    # the stopping path of the full command; progress p in seconds at full command speed
    twist = command * cfg.speed_factor
    v_lin, w_abs = torch.linalg.norm(twist[:, :2], dim=1), twist[:, 2].abs()
    moving_cmd = (v_lin > 1e-4) | (w_abs > 1e-4)
    alpha = torch.minimum(cfg.brake / v_lin.clamp(min=1e-4), cfg.brake_yaw / w_abs.clamp(min=1e-4))  # fraction / s
    t = cfg.latency
    p_full = t + 1.0 / (2.0 * alpha)
    frac = torch.linspace(0.0, 1.0, cfg.samples + 1, device=dev)[None]
    progress = p_full[:, None] * frac  # (n, J)
    x, y, th = _poses(twist, progress)
    c, s = torch.cos(th)[..., None], torch.sin(th)[..., None]
    dx, dy = pts[:, None, :, 0] - x[..., None], pts[:, None, :, 1] - y[..., None]
    gaps, _, _ = _gap(torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=-1), core, r)  # (n, J, k)
    gaps = gaps - cfg.drift * (v_lin[:, None] * progress)[..., None]  # straying from the arc

    def stop_progress(scale):  # progress needed to stop from scale (fraction of the command)
        return scale * t + scale * scale / (2.0 * alpha)

    def admissible(scale):
        speed = scale * v_lin
        grow = ((speed - cfg.slow_speed) / (cfg.fast_speed - cfg.slow_speed)).clamp(0.0, 1.0)
        tall_margin = cfg.margin + (cfg.person_margin - cfg.margin) * grow
        margin = torch.where(drop, cfg.margin + cfg.drop_extra, torch.where(tall, tall_margin[:, None], torch.full_like(gap0, cfg.margin)))
        margin = torch.where(people, torch.full_like(margin, cfg.person_margin), margin)
        # time to reach each sample at this scale (s): constant speed for the latency, then braking
        p_lat = (scale * t)[:, None]
        rem = (progress - p_lat).clamp(min=0.0)
        disc = (scale * scale)[:, None] - 2.0 * alpha[:, None] * rem
        t_brake = (scale[:, None] - torch.sqrt(disc.clamp(min=0.0))) / alpha[:, None]
        time = torch.where(progress <= p_lat, progress / scale.clamp(min=1e-6)[:, None], t + t_brake)
        effective = gaps - cfg.human_speed * time[..., None] * approaching[:, None]
        reached = progress <= stop_progress(scale)[:, None] + p_full[:, None] / cfg.samples  # the path actually driven
        closer = gaps < gap0[:, None] - 0.01  # the robot's own motion brings it closer
        breach = obstacle[:, None] & closer & (effective < margin[:, None]) & reached[..., None]
        return ~breach.any(dim=(1, 2))

    lo, hi = torch.zeros(n, device=dev), torch.ones(n, device=dev)
    full_ok = admissible(hi)
    for _ in range(10):
        mid = (lo + hi) / 2
        ok = admissible(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    scale = torch.where(full_ok, torch.ones_like(lo), lo)
    scale = torch.where(moving_cmd, scale, torch.ones_like(scale))
    emergency = torch.zeros(n, dtype=torch.bool, device=dev)
    if velocity is not None:  # the robot's own momentum along the command: can it still stop in time?
        weight = torch.tensor([1.0, 1.0, 0.25], device=dev)  # yaw rate counts with a 0.5 m lever
        along = (velocity * twist * weight).sum(dim=1) / (twist * twist * weight).sum(dim=1).clamp(min=1e-8)
        s_now = along.clamp(min=0.0, max=1.0)
        emergency = moving_cmd & (s_now > scale + 1e-3) & ~admissible(s_now)
        scale = torch.where(emergency, torch.zeros_like(scale), scale)
    # closing speed towards tall things: at most what still allows slowing to approach_speed at approach_gap
    u = normal[..., 0] * (twist[:, 0, None] - twist[:, 2, None] * nearest[..., 1]) + normal[..., 1] * (
        twist[:, 1, None] + twist[:, 2, None] * nearest[..., 0])
    at = cfg.brake * t
    room = (gap0 - cfg.approach_gap).clamp(min=0.0)
    u_cap = torch.sqrt(at * at + cfg.approach_speed**2 + 2.0 * cfg.brake * room) - at
    u_cap = torch.where(gap0 > cfg.approach_gap, u_cap.clamp(min=cfg.approach_speed), torch.full_like(u_cap, cfg.approach_speed))
    ratio = torch.where((tall | people) & (u > 1e-6), u_cap / u.clamp(min=1e-6), torch.full_like(u, float("inf")))
    scale = torch.minimum(scale, ratio.amin(dim=1).clamp(max=1.0))
    safe = command * scale[:, None]
    far = torch.full_like(gap0, 1e3)
    info = {
        "scale": scale,
        "emergency": emergency,
        "min_gap": torch.where(obstacle, gap0, far).amin(dim=1),
        "min_tall_gap": torch.where(tall, gap0, far).amin(dim=1),
        "min_moving_gap": torch.where(approaching, gap0, far).amin(dim=1),
        "min_person_gap": torch.where(people, gap0, far).amin(dim=1),
    }
    return safe, info
