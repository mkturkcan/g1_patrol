"""Litter and trash bags on the New York street tiles: where they lie, what happens to them, and where they go.

Each environment has one rigid body per item of the scene's ``trash`` collection: litter (coffee cups, cans, bottles,
snack bags, takeout containers; see ``scripts/assets/make_nyc_trash.py``) and black trash bags. Per episode, a random
subset of them lies along the robot's route: litter on tiles with a litter basket, bags on tiles with a curbside bag
pile. The others fall freely far below the street, out of reach of everything (and are put back every few seconds),
which costs the physics engine almost nothing. An item is on the ground until the robot picks it up, held while the robot carries it, falling after it lets go, and deposited once it comes to rest inside a litter
basket (litter) or in the zone beside a bag pile (bags); otherwise it is back on the ground. Items that end up in the
roadway or on a crosswalk are lost: the robot does not step into the street to fetch them.
"""

from __future__ import annotations

import json
import math
import os

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from ..nyc.tiles import MAX_BASKETS, MAX_CROSSINGS, MAX_PILES

PARKED, ON_GROUND, HELD, FALLING, DEPOSITED = 0, 1, 2, 3, 4
LITTER, BAG = 0, 1
PARK_Z = -20.0
"""Height below the street from which unused items fall freely (they are put back every ``repark_interval`` s)."""


def load_trash_manifest(directory: str) -> dict:
    with open(os.path.join(directory, "manifest.json")) as f:
        return json.load(f)


def _rest_quat(size, kind: int, upright: bool = False) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) of an item on the ground: bags stand (knot up), and so does litter marked upright (an
    open pizza box); other litter lies with its smallest dimension upright (on its side, or flat)."""
    if kind == BAG or upright:
        return (0.0, 0.0, 0.0, 1.0)
    axis = min(range(3), key=lambda i: size[i])
    h = math.sqrt(0.5)
    return [(0.0, h, 0.0, h), (h, 0.0, 0.0, h), (0.0, 0.0, 0.0, 1.0)][axis]


def _in_rects(p: torch.Tensor, rects: torch.Tensor, valid: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Points (n, m, 2) inside any of the axis-aligned rectangles (n, r, 4: x0, x1, y0, y1) -> (n, m)."""
    x, y = p[..., None, 0], p[..., None, 1]
    r = rects[:, None]
    inside = (x > r[..., 0] - margin) & (x < r[..., 1] + margin) & (y > r[..., 2] - margin) & (y < r[..., 3] + margin)
    return (inside & valid[:, None]).any(dim=-1)


def _rect_point(p: torch.Tensor, rects: torch.Tensor) -> torch.Tensor:
    """Closest points (n, r, 2) of the rectangles (n, r, 4) to points p (n, 2)."""
    return torch.stack([p[:, None, 0].clamp(rects[..., 0], rects[..., 1]), p[:, None, 1].clamp(rects[..., 2], rects[..., 3])], dim=-1)


class TrashMixin:
    """Litter and bag state and queries; mixed into :class:`NycScenarioCommand` (uses its tile layout and robot)."""

    def _init_trash(self, cfg, env):
        n, dev = self.num_envs, self.device
        self.trash = env.scene[cfg.trash_name] if cfg.trash_name is not None else None
        self.num_trash = self.trash.num_bodies if self.trash is not None else 0
        k = self.num_trash
        self.trash_state = torch.full((n, k), PARKED, dtype=torch.long, device=dev)
        self.trash_release_time = torch.zeros(n, k, device=dev)
        self.held_item = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.baskets = torch.zeros(n, MAX_BASKETS, 4, device=dev)
        self.basket_valid = torch.zeros(n, MAX_BASKETS, dtype=torch.bool, device=dev)
        self.piles = torch.zeros(n, MAX_PILES, 4, device=dev)
        """Zones (x0, x1, y0, y1) beside the curbside bag piles, where bags are set down."""
        self.pile_valid = torch.zeros(n, MAX_PILES, dtype=torch.bool, device=dev)
        self.crossings = torch.zeros(n, MAX_CROSSINGS, 4, device=dev)
        self.crossing_valid = torch.zeros(n, MAX_CROSSINGS, dtype=torch.bool, device=dev)
        # per-step events (rewards) and per-episode counts (metrics)
        self.trash_picked = torch.zeros(n, device=dev)
        self.trash_deposited = torch.zeros(n, device=dev)
        self.bags_delivered = torch.zeros(n, device=dev)
        self.trash_lost = torch.zeros(n, device=dev)
        self.episode_trash = torch.zeros(n, device=dev)
        self.episode_bags = torch.zeros(n, device=dev)
        self.episode_picked = torch.zeros(n, device=dev)
        self.episode_deposited = torch.zeros(n, device=dev)
        self.episode_delivered = torch.zeros(n, device=dev)
        for name in ("trash_items", "trash_deposited", "bags", "bags_delivered", "trash_picked"):
            self.metrics[name] = torch.zeros(n, device=dev)
        self.trash_kind = torch.zeros(k, dtype=torch.long, device=dev)
        self.trash_mass = torch.zeros(k, device=dev)
        self.trash_size = torch.zeros(k, 3, device=dev)
        if k == 0:
            return
        manifest = load_trash_manifest(cfg.trash_dir)
        items = [manifest[name] for name in cfg.trash_items]
        kinds = [BAG if item.get("kind", "litter") == "bag" else LITTER for item in items]
        self.trash_kind[:] = torch.tensor(kinds, device=dev)
        self.trash_mass[:] = torch.tensor([item["mass"] for item in items], device=dev)
        self.trash_size[:] = torch.tensor([item["size"] for item in items], device=dev)
        upright = [kind == BAG or item.get("rest") == "upright" for item, kind in zip(items, kinds)]
        self.trash_rest_height = torch.tensor(
            [item["size"][2] / 2 if up else min(item["size"]) / 2 for item, up in zip(items, upright)], device=dev
        )
        self.trash_rest_quat = torch.tensor(
            [_rest_quat(item["size"], kind, up) for item, kind, up in zip(items, kinds, upright)], device=dev
        )
        self.trash_grasp_height = torch.where(self.trash_kind == BAG, self.trash_size[:, 2] / 2 - 0.05, torch.zeros_like(self.trash_mass))
        """Height of the grasp point above an item's center: a bag's knot (below the tie ends), the middle of litter."""

    @property
    def is_bag(self) -> torch.Tensor:
        return self.trash_kind == BAG

    # -- episode setup ------------------------------------------------------------------------------------------------

    def _trash_metrics(self, env_ids):
        self.metrics["trash_items"][env_ids] = self.episode_trash[env_ids]
        self.metrics["trash_deposited"][env_ids] = self.episode_deposited[env_ids]
        self.metrics["bags"][env_ids] = self.episode_bags[env_ids]
        self.metrics["bags_delivered"][env_ids] = self.episode_delivered[env_ids]
        self.metrics["trash_picked"][env_ids] = self.episode_picked[env_ids]

    def _street_mask(self, p: torch.Tensor, env_ids=None) -> torch.Tensor:
        """Tile-local points (m, q, 2) in the roadway or on a crosswalk."""
        e = slice(None) if env_ids is None else env_ids
        return _in_rects(p, self.forbidden[e], self.forbidden_valid[e]) | _in_rects(p, self.crossings[e], self.crossing_valid[e])

    def _spawn_trash(self, env_ids: torch.Tensor, route: torch.Tensor):
        """Litter along the robot's route on tiles with a basket, bags on tiles with a bag pile."""
        lay, dev = self.layout, self.device
        r, c = self.tile_row[env_ids], self.tile_col[env_ids]
        self.baskets[env_ids] = lay.baskets[r, c]
        self.basket_valid[env_ids] = torch.arange(MAX_BASKETS, device=dev) < lay.num_baskets[r, c, None]
        self.piles[env_ids] = lay.piles[r, c]
        self.pile_valid[env_ids] = torch.arange(MAX_PILES, device=dev) < lay.num_piles[r, c, None]
        self.crossings[env_ids] = lay.crossings[r, c]
        self.crossing_valid[env_ids] = torch.arange(MAX_CROSSINGS, device=dev) < lay.num_crossings[r, c, None]
        self.held_item[env_ids] = -1
        self.trash_release_time[env_ids] = 0.0
        self.episode_picked[env_ids] = 0.0
        self.episode_deposited[env_ids] = 0.0
        self.episode_delivered[env_ids] = 0.0
        if self.num_trash == 0:
            self.episode_trash[env_ids] = 0.0
            self.episode_bags[env_ids] = 0.0
            return
        n, k, cfg = len(env_ids), self.num_trash, self.cfg
        bag = self.is_bag[None].expand(n, k)
        lo, hi = cfg.trash_count
        litter_count = torch.randint(lo, hi + 1, (n,), device=dev) * self.basket_valid[env_ids].any(dim=1)
        lo, hi = cfg.bag_count
        bag_count = torch.randint(lo, hi + 1, (n,), device=dev) * self.pile_valid[env_ids].any(dim=1)
        # which items lie on the tile: random subsets of the litter and of the bags
        score = torch.rand(n, k, device=dev) + 2.0 * bag  # litter ranks first
        rank = score.argsort(dim=1).argsort(dim=1)
        num_litter = int((~self.is_bag).sum())
        active = torch.where(bag, (rank - num_litter) < bag_count[:, None], rank < litter_count[:, None])
        # candidate spots along the route, within ``trash_lateral`` of it
        m = cfg.trash_candidates
        start, goal = route[:, :2], route[:, 2:]
        along = goal - start
        length = torch.linalg.norm(along, dim=1, keepdim=True).clamp(min=1e-3)
        u = along / length
        normal = torch.stack([-u[:, 1], u[:, 0]], dim=1)
        t = 0.15 + 0.8 * torch.rand(n, k, m, device=dev)
        side = (2 * torch.rand(n, k, m, device=dev) - 1) * cfg.trash_lateral
        pts = start[:, None, None] + t[..., None] * along[:, None, None] + side[..., None] * normal[:, None, None]
        flat = pts.reshape(n, k * m, 2)
        clearance = torch.where(bag, cfg.bag_clearance, cfg.trash_clearance)[..., None].expand(n, k, m).reshape(n, k * m)
        ok = self.obstacle_clearance(flat, radius=0.0, env_ids=env_ids) > clearance
        ok &= ~self._street_mask(flat, env_ids)
        ok &= ~_in_rects(flat, self.piles[env_ids], self.pile_valid[env_ids], margin=0.3)  # not already delivered
        ok &= torch.linalg.norm(flat - self.start[env_ids][:, None], dim=-1) > 1.2
        ok &= torch.linalg.norm(flat - self.goal[env_ids][:, None], dim=-1) > 0.8
        ground = lay.ground_height(r, c, flat)
        start_ground = lay.ground_height(r, c, self.start[env_ids][:, None])
        ok &= (ground - start_ground).abs() < 0.3  # on the sidewalk level, not in a stairwell
        ok = ok.reshape(n, k, m)
        # first valid candidate per item; items keep 0.4 m apart (greedy, in slot order)
        pick = torch.where(ok.any(dim=2), ok.float().argmax(dim=2), torch.full((n, k), -1, device=dev))
        pos = torch.gather(pts, 2, pick.clamp(min=0)[..., None, None].expand(n, k, 1, 2)).squeeze(2)
        placed = active & (pick >= 0)
        for j in range(1, k):
            d = torch.linalg.norm(pos[:, j:j + 1] - pos[:, :j], dim=-1)
            clash = ((d < 0.4 + 0.3 * (bag[:, j:j + 1] | bag[:, :j])) & placed[:, :j]).any(dim=1)
            placed[:, j] &= ~clash
        self.trash_state[env_ids] = torch.where(placed, ON_GROUND, PARKED)
        self.episode_trash[env_ids] = (placed & ~bag).float().sum(dim=1)
        self.episode_bags[env_ids] = (placed & bag).float().sum(dim=1)
        # poses: on the ground with a random heading, or parked far below the street
        z = lay.ground_height(r, c, pos) + self.trash_rest_height[None] + 0.005
        park = torch.stack([torch.arange(k, device=dev) * 0.8 - 8.0, torch.zeros(k, device=dev)], dim=1)
        xy = torch.where(placed[..., None], pos, park[None].expand(n, k, 2))
        z = torch.where(placed, z, torch.full_like(z, PARK_Z))
        yaw = 2 * math.pi * torch.rand(n, k, device=dev)
        zeros = torch.zeros_like(yaw)
        quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw).reshape(-1, 4), self.trash_rest_quat[None].expand(n, k, 4).reshape(-1, 4))
        pose = torch.cat([xy + self.tile_center[env_ids][:, None], z[..., None], quat.reshape(n, k, 4)], dim=-1)
        self.trash.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        self.trash.write_body_velocity_to_sim_index(body_velocities=torch.zeros(n, k, 6, device=dev), env_ids=env_ids)

    def _repark(self):
        """Put the unused items, which fall freely below the street, back at ``PARK_Z`` (one write for all)."""
        n, k, dev = self.num_envs, self.num_trash, self.device
        parked = self.trash_state == PARKED
        pose = torch.cat([self.trash.data.body_link_pos_w.torch, self.trash.data.body_link_quat_w.torch], dim=-1).clone()
        vel = torch.cat([self.trash.data.body_link_lin_vel_w.torch, self.trash.data.body_link_ang_vel_w.torch], dim=-1).clone()
        park = torch.stack([torch.arange(k, device=dev) * 0.8 - 8.0, torch.zeros(k, device=dev)], dim=1)
        pose[..., :2] = torch.where(parked[..., None], park[None] + self.tile_center[:, None], pose[..., :2])
        pose[..., 2] = torch.where(parked, PARK_Z, pose[..., 2])
        vel = torch.where(parked[..., None], torch.zeros_like(vel), vel)
        self.trash.write_body_pose_to_sim_index(body_poses=pose)
        self.trash.write_body_velocity_to_sim_index(body_velocities=vel)

    # -- per step -----------------------------------------------------------------------------------------------------

    def trash_positions(self) -> torch.Tensor:
        """Tile-local item positions (n, k, 3); z is world height."""
        p = self.trash.data.body_link_pos_w.torch.clone()
        p[..., :2] -= self.tile_center[:, None]
        return p

    def update_trash(self, dt: float):
        """Resolve released items (in a basket or a bag zone, or back on the ground) and lost ones; set the per-step
        events."""
        self.trash_picked.zero_()
        self.trash_deposited.zero_()
        self.bags_delivered.zero_()
        self.trash_lost.zero_()
        if self.num_trash == 0:
            return
        p = self.trash_positions()
        speed = torch.linalg.norm(self.trash.data.body_link_lin_vel_w.torch, dim=-1)
        falling = self.trash_state == FALLING
        self.trash_release_time += dt * falling
        # litter inside a basket: within its inner radius and below its rim; bags in a zone beside a bag pile
        rel = p[:, :, None, :2] - self.baskets[:, None, :, :2]
        in_basket = (torch.linalg.norm(rel, dim=-1) < self.baskets[:, None, :, 2] - 0.03) & (p[:, :, None, 2] < self.baskets[:, None, :, 3])
        in_basket = (in_basket & self.basket_valid[:, None]).any(dim=2)
        in_zone = _in_rects(p[..., :2], self.piles, self.pile_valid, margin=0.15)
        inside = torch.where(self.is_bag[None], in_zone, in_basket)
        settled = speed < 0.3
        deposited = falling & inside & settled & (self.trash_release_time > 0.2)
        missed = falling & ~inside & (settled | (self.trash_release_time > 3.0)) & (self.trash_release_time > 0.3)
        self.trash_state[deposited] = DEPOSITED
        self.trash_state[missed] = ON_GROUND
        litter_in = (deposited & ~self.is_bag[None]).float().sum(dim=1)
        bags_in = (deposited & self.is_bag[None]).float().sum(dim=1)
        self.trash_deposited += litter_in
        self.bags_delivered += bags_in
        self.episode_deposited += litter_in
        self.episode_delivered += bags_in
        # items that rolled or were kicked into the roadway are lost
        lost = (self.trash_state == ON_GROUND) & self._street_mask(p[..., :2])
        self.trash_state[lost] = PARKED
        self.trash_lost += lost.float().sum(dim=1)
        self._repark_timer = getattr(self, "_repark_timer", 0.0) + dt
        if self._repark_timer >= self.cfg.repark_interval:
            self._repark_timer = 0.0
            self._repark()

    # -- queries (robot heading frame unless noted) -------------------------------------------------------------------

    def _to_heading(self, rel: torch.Tensor) -> torch.Tensor:
        yaw = self.robot_yaw()
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        return torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], dim=-1)

    def nearest_trash(self, count: int, max_range: float = 8.0) -> torch.Tensor:
        """Nearest ``count`` items on the ground: relative position (heading frame), height of the grasp point above
        the robot's feet level, presence, bag. Shape (n, 5 * count)."""
        n = self.num_envs
        out = torch.zeros(n, count, 5, device=self.device)
        if self.num_trash == 0:
            return out.reshape(n, -1)
        p = self.trash_positions()
        rel = p[..., :2] - self.robot_pos()[:, None]
        dist = torch.where(self.trash_state == ON_GROUND, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e3))
        order = dist.argsort(dim=1)[:, : min(count, self.num_trash)]
        d = torch.gather(dist, 1, order)
        rel_b = self._to_heading(torch.gather(rel, 1, order[..., None].expand(-1, -1, 2)))
        present = (d < max_range).float()
        feet = self.layout.ground_height(self.tile_row, self.tile_col, self.robot_pos()[:, None])
        h = torch.gather(p[..., 2], 1, order) + self.trash_grasp_height[order] - feet
        bag = self.is_bag[order].float()
        feats = torch.cat([rel_b, h[..., None], torch.ones_like(h[..., None]), bag[..., None]], dim=-1) * present[..., None]
        out[:, : order.shape[1]] = feats
        return out.reshape(n, -1)

    def nearest_baskets(self, count: int, max_range: float = 12.0) -> torch.Tensor:
        """Nearest ``count`` litter baskets: relative position of the center (heading frame), presence. (n, 3 * count)."""
        rel = self.baskets[..., :2] - self.robot_pos()[:, None]
        dist = torch.where(self.basket_valid, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e3))
        order = dist.argsort(dim=1)[:, :count]
        d = torch.gather(dist, 1, order)
        rel_b = self._to_heading(torch.gather(rel, 1, order[..., None].expand(-1, -1, 2)))
        present = (d < max_range).float()
        return torch.cat([rel_b * present[..., None], present[..., None]], dim=-1).reshape(self.num_envs, -1)

    def nearest_piles(self, count: int, max_range: float = 12.0) -> torch.Tensor:
        """Nearest ``count`` bag zones: relative position of the zone's closest point (heading frame), presence.
        (n, 3 * count)."""
        p = self.robot_pos()
        rel = _rect_point(p, self.piles) - p[:, None]
        dist = torch.where(self.pile_valid, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e3))
        order = dist.argsort(dim=1)[:, :count]
        d = torch.gather(dist, 1, order)
        rel_b = self._to_heading(torch.gather(rel, 1, order[..., None].expand(-1, -1, 2)))
        present = (d < max_range).float()
        return torch.cat([rel_b * present[..., None], present[..., None]], dim=-1).reshape(self.num_envs, -1)

    def reachable_item(self, ahead: tuple[float, float], side: float, bags: bool | None = None) -> torch.Tensor:
        """Index of the item on the ground closest to the reach zone center (``ahead`` m in front of the robot, within
        ``side`` m to either side), or -1; ``bags``: only bags (True), only litter (False). Shape (n,)."""
        if self.num_trash == 0:
            return torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        p = self._to_heading(self.trash_positions()[..., :2] - self.robot_pos()[:, None])
        x, y = p[..., 0], p[..., 1]
        ok = (self.trash_state == ON_GROUND) & (x > ahead[0]) & (x < ahead[1]) & (y.abs() < side)
        if bags is not None:
            ok &= self.is_bag[None] if bags else ~self.is_bag[None]
        score = torch.where(ok, (x - (ahead[0] + ahead[1]) / 2) ** 2 + y**2, torch.full_like(x, 1e3))
        best = score.argmin(dim=1)
        return torch.where(ok.any(dim=1), best, torch.full_like(best, -1))

    def reachable_basket(self, distance: tuple[float, float], half_angle: float) -> torch.Tensor:
        """Index of the basket whose center is ``distance`` m from the robot within ``half_angle`` of its heading, or -1."""
        rel = self.baskets[..., :2] - self.robot_pos()[:, None]
        d = torch.linalg.norm(rel, dim=-1)
        bearing = torch.atan2(rel[..., 1], rel[..., 0]) - self.robot_yaw()[:, None]
        bearing = torch.remainder(bearing + math.pi, 2 * math.pi) - math.pi
        ok = self.basket_valid & (d > distance[0]) & (d < distance[1]) & (bearing.abs() < half_angle)
        best = torch.where(ok, d, torch.full_like(d, 1e3)).argmin(dim=1)
        return torch.where(ok.any(dim=1), best, torch.full_like(best, -1))

    def reachable_pile(self, max_distance: float, half_angle: float) -> torch.Tensor:
        """Index of the bag zone whose closest point is within ``max_distance`` m of the robot and ``half_angle`` of
        its heading (or under it), or -1."""
        p = self.robot_pos()
        rel = _rect_point(p, self.piles) - p[:, None]
        d = torch.linalg.norm(rel, dim=-1)
        bearing = torch.atan2(rel[..., 1], rel[..., 0]) - self.robot_yaw()[:, None]
        bearing = torch.remainder(bearing + math.pi, 2 * math.pi) - math.pi
        ok = self.pile_valid & (d < max_distance) & ((bearing.abs() < half_angle) | (d < 0.05))
        best = torch.where(ok, d, torch.full_like(d, 1e3)).argmin(dim=1)
        return torch.where(ok.any(dim=1), best, torch.full_like(best, -1))

    def held_in_zone(self, margin: float = 0.05) -> torch.Tensor:
        """The robot holds a bag that hangs over a zone beside a bag pile (at least ``margin`` m inside it). (n,)"""
        held = self.held_item
        if self.num_trash == 0:
            return torch.zeros_like(held, dtype=torch.bool)
        k = held.clamp(min=0)
        p = self.trash_positions()[torch.arange(self.num_envs, device=self.device), k, :2]
        inside = _in_rects(p[:, None], self.piles, self.pile_valid, margin=-margin)[:, 0]
        return (held >= 0) & self.is_bag[k] & inside

    def pile_drop_point(self, pile: torch.Tensor, inset: float = 0.25) -> torch.Tensor:
        """Tile-local point (n, 2) inside bag zone ``pile`` (n,) nearest the robot, ``inset`` m in from its edges."""
        rect = self.piles[torch.arange(self.num_envs, device=self.device), pile.clamp(min=0)]
        p = self.robot_pos()
        cx, cy = (rect[:, 0] + rect[:, 1]) / 2, (rect[:, 2] + rect[:, 3]) / 2
        hx = ((rect[:, 1] - rect[:, 0]) / 2 - inset).clamp(min=0.0)
        hy = ((rect[:, 3] - rect[:, 2]) / 2 - inset).clamp(min=0.0)
        return torch.stack([p[:, 0].clamp(cx - hx, cx + hx), p[:, 1].clamp(cy - hy, cy + hy)], dim=1)
