"""Following a planned route (pure torch; shared by the simulation and the robot).

A route is a polyline from the robot's start to its goal, e.g. the path of a global planner. The robot's place along
it is the arc length of the closest point, searched near the previous place so that it does not jump to a part of the
route that passes close by (on the other side of a wall). The navigation policy steers by points of the route a few
meters ahead of that place (the carrots of a path follower), and its progress is measured along the route.
"""

from __future__ import annotations

import torch


def polyline(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Points (..., K, 2) and their cumulative arc lengths (..., K); padding repeats the last point."""
    seg = torch.linalg.norm(points[..., 1:, :] - points[..., :-1, :], dim=-1)
    return points, torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], dim=-1)


def path_at(pts: torch.Tensor, cum: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Point and unit direction at arc length ``s`` (...) of polylines with points ``pts`` (..., K, 2) and cumulative
    lengths ``cum`` (..., K); padded points repeat the last point. ``s`` is clamped to the polyline."""
    s = torch.minimum(s.clamp(min=0.0), cum[..., -1])
    k = (cum[..., 1:] < s[..., None]).sum(-1).clamp(max=pts.shape[-2] - 2)  # segment containing s
    a = torch.gather(pts, -2, k[..., None, None].expand(*k.shape, 1, 2)).squeeze(-2)
    b = torch.gather(pts, -2, (k + 1)[..., None, None].expand(*k.shape, 1, 2)).squeeze(-2)
    c0 = torch.gather(cum, -1, k[..., None]).squeeze(-1)
    u = (b - a) / torch.linalg.norm(b - a, dim=-1, keepdim=True).clamp(min=1e-6)
    return a + (s - c0)[..., None] * u, u


def path_project(pts: torch.Tensor, cum: torch.Tensor, x: torch.Tensor, s_prev: torch.Tensor | None = None) -> torch.Tensor:
    """Arc length of the point of the polylines closest to ``x`` (..., 2). With ``s_prev``, parts of the polyline more
    than 1 m behind or 3 m ahead of ``s_prev`` are penalized, so that a walker does not skip to a distant part."""
    a, b = pts[..., :-1, :], pts[..., 1:, :]
    seg_len = cum[..., 1:] - cum[..., :-1]
    u = (b - a) / seg_len[..., None].clamp(min=1e-6)
    t = torch.minimum(((x[..., None, :] - a) * u).sum(-1).clamp(min=0.0), seg_len)
    d = torch.linalg.norm(x[..., None, :] - (a + t[..., None] * u), dim=-1)
    s_k = cum[..., :-1] + t
    d = torch.where(seg_len > 1e-6, d, torch.full_like(d, 1e6))  # padding
    if s_prev is not None:
        d = d + 2.0 * ((s_prev[..., None] - 1.0 - s_k).clamp(min=0.0) + (s_k - s_prev[..., None] - 3.0).clamp(min=0.0))
    return torch.gather(s_k, -1, d.argmin(-1, keepdim=True)).squeeze(-1)


def remaining(pts: torch.Tensor, cum: torch.Tensor, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Distance (...) left to the end of the route from ``x`` (..., 2) at place ``s``: along the route, plus how far
    ``x`` is off it."""
    on, _ = path_at(pts, cum, s)
    return (cum[..., -1] - s) + torch.linalg.norm(x - on, dim=-1)


def waypoints(pts: torch.Tensor, cum: torch.Tensor, s: torch.Tensor, x: torch.Tensor, yaw: torch.Tensor,
              lookahead: torch.Tensor) -> torch.Tensor:
    """Points of the route ``lookahead`` (W,) meters ahead of place ``s`` (n,) (the goal where the route ends) in the
    frame of a robot at ``x`` (n, 2) with heading ``yaw`` (n,): (n, W, 2)."""
    w = lookahead.shape[0]
    ahead, _ = path_at(pts[:, None].expand(-1, w, -1, -1), cum[:, None].expand(-1, w, -1), s[:, None] + lookahead)
    rel = ahead - x[:, None]
    c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack([c * rel[..., 0] + sn * rel[..., 1], -sn * rel[..., 0] + c * rel[..., 1]], dim=-1)
