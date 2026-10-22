"""Who else is on a New York sidewalk (civic tasks), and incidents the robot has to handle.

Besides the crowd of :class:`NycScenarioCommand`, a civic episode can have

* people who need more room and priority: wheelchair users and parents with strollers (wide bodies, slow) and blind
  people with a white cane (slow; they do not see the robot and only step aside at the last moment);
* dogs on a leash, walking around their owner: the robot must not pass between dog and owner;
* friends walking side by side, and lines of people waiting (at a shop door, a bus stop, a kiosk): the robot must not
  split a group or cut through a line;
* cyclists riding on the sidewalk (3-5 m/s; they swerve but do not slow down);

and at most one incident:

* someone blocks the robot's way (stands in front of it, following its moves) for a while, then walks off;
* someone attacks the robot: walks or runs at it, shoves and kicks it repeatedly while it is within reach, chases it
  for a while, then gives up and walks off. The robot cannot fight back and must not: it braces, keeps its footing,
  gets away, and keeps clear of bystanders and the roadway. Attack speed, strength and duration grow with the tile's
  difficulty level (a curriculum).

Pedestrian slots of the collection: ``num_regular`` regular people (capsules), then ``num_wide`` wide bodies
(wheelchairs, strollers), then ``num_dogs`` dogs.
"""

from __future__ import annotations

import math

import torch

from ..nyc.tiles import HALF_L, MAX_PATH_POINTS, MAX_QUEUES

NORMAL, CANE, WHEELCHAIR, CYCLIST, DOG, QUEUE = 0, 1, 2, 3, 4, 5
NO_INCIDENT, ATTACK, BLOCK, FALLEN = 0, 1, 2, 3
PENDING, ACTIVE, LEAVING, DONE = 0, 1, 2, 3
NO_LINK, LEASH, GROUP, LINE = 0, 1, 2, 3
_PARKED, _WALKING, _STANDING = 0, 1, 2  # pedestrian modes (as in nyc_scenario)


def _segment_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance from points p (n, 1, 2) to segments a-b (n, m, 2)."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
    return torch.linalg.norm(p - (a + t[..., None] * ab), dim=-1)


class SocialMixin:
    """Special pedestrians and incidents; mixed into :class:`NycScenarioCommand`. Disabled unless ``cfg.social``."""

    def _init_social(self, cfg):
        n, p, dev = self.num_envs, self.num_peds, self.device
        self.social = cfg.social
        self.ped_radius = torch.full((n, p), cfg.pedestrian_radius, device=dev)
        self.ped_height = torch.full((n, p), cfg.pedestrian_center_height, device=dev)
        self.ped_role = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.ped_vulnerable = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.ped_link = torch.full((n, p), -1, dtype=torch.long, device=dev)
        self.ped_link_kind = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.ped_group = torch.full((n, p), -1, dtype=torch.long, device=dev)
        self.ped_max_speed = torch.full((n, p), cfg.max_pedestrian_speed, device=dev)
        self.hostile = torch.zeros(n, p, dtype=torch.bool, device=dev)
        """Pedestrians who attack or block the robot (from the start of the incident until they have left)."""
        self.attacker = torch.zeros(n, p, dtype=torch.bool, device=dev)
        """The hostile pedestrians who attack (the others block)."""
        self.dog_angle = torch.zeros(n, p, device=dev)
        self.dog_len = torch.ones(n, p, device=dev)
        # incidents
        self.incident = torch.zeros(n, dtype=torch.long, device=dev)
        self.incident_phase = torch.full((n,), DONE, dtype=torch.long, device=dev)
        self.incident_ped = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.incident_start = torch.zeros(n, device=dev)
        self.incident_timer = torch.zeros(n, device=dev)
        self.episode_time = torch.zeros(n, device=dev)
        self.attack_speed = torch.zeros(n, device=dev)
        self.shove_max = torch.zeros(n, device=dev)
        self.shove_cooldown = torch.zeros(n, device=dev)
        self.shove_left = torch.zeros(n, device=dev)
        self.attack_force = torch.zeros(n, 3, 3, device=dev)
        """World-frame force (N) of the current shove or kick on the robot's torso, left knee and right knee."""
        self.shoved = torch.zeros(n, device=dev)
        """Shoves and kicks that started at this step."""
        self.episode_shoves = torch.zeros(n, device=dev)
        self.attacked_episode = torch.zeros(n, dtype=torch.bool, device=dev)
        self.under_attack = torch.zeros(n, dtype=torch.bool, device=dev)
        """An attacker is after the robot, or has just given up but is still close."""
        # crossing a leash, a group or a line (per-step events; ``inside`` keeps them to one per crossing)
        self.leash_crossed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.split_event = torch.zeros(n, device=dev)
        self._link_inside = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.metrics["shoves"] = torch.zeros(n, device=dev)
        self.metrics["attacked"] = torch.zeros(n, device=dev)
        regular = cfg.num_regular if cfg.num_regular is not None else p
        self.slot_kind = torch.zeros(p, dtype=torch.long, device=dev)  # 0 regular, 1 wide, 2 dog
        self.slot_kind[regular : regular + cfg.num_wide] = 1
        self.slot_kind[regular + cfg.num_wide : regular + cfg.num_wide + cfg.num_dogs] = 2

    # -- helpers ------------------------------------------------------------------------------------------------------

    def _is_regular(self) -> torch.Tensor:
        return self.slot_kind == 0

    def active_pedestrians(self, env_ids: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        """Which slots take part in an episode: the first ``budget`` regular slots; with social pedestrians, also each
        wide slot with probability ``wheelchair_ratio`` (dogs join their owners later)."""
        n, p, dev = len(env_ids), self.num_peds, self.device
        idx = torch.arange(p, device=dev)[None].expand(n, p)
        if not self.social:
            return idx < budget[:, None]
        regular = self._is_regular()[None] & (idx < budget[:, None])
        d = self.difficulty[env_ids][:, None]
        lo, hi = self.cfg.wheelchair_ratio
        wide = (self.slot_kind == 1)[None] & (torch.rand(n, p, device=dev) < lo + (hi - lo) * d)
        return regular | wide

    def _ranged(self, rng: tuple[float, float], d: torch.Tensor) -> torch.Tensor:
        return rng[0] + (rng[1] - rng[0]) * d

    # -- episode setup ------------------------------------------------------------------------------------------------

    def _spawn_social(self, env_ids: torch.Tensor, route: torch.Tensor):
        cfg, n, p, dev = self.cfg, len(env_ids), self.num_peds, self.device
        e = env_ids
        self.ped_role[e] = NORMAL
        self.ped_radius[e] = cfg.pedestrian_radius
        self.ped_height[e] = cfg.pedestrian_center_height
        self.ped_vulnerable[e] = False
        self.ped_link[e] = -1
        self.ped_link_kind[e] = NO_LINK
        self.ped_group[e] = -1
        self.ped_max_speed[e] = cfg.max_pedestrian_speed
        self.hostile[e] = False
        self.attacker[e] = False
        self._link_inside[e] = False
        self.attack_force[e] = 0.0
        self.shove_left[e] = 0.0
        self.shove_cooldown[e] = 0.0
        self.episode_shoves[e] = 0.0
        self.attacked_episode[e] = False
        self.under_attack[e] = False
        self.episode_time[e] = 0.0
        self.incident[e] = NO_INCIDENT
        self.incident_phase[e] = DONE
        self.incident_ped[e] = -1
        if not self.social:
            return
        d = self.difficulty[e]
        mode, timer = self.ped_mode[e], self.ped_timer[e]
        present = (mode != _PARKED) | (timer < 1e8)  # takes part in the episode (now or later)
        regular = self._is_regular()[None].expand(n, p)
        start_x = torch.gather(self.path_pts[e][..., 0, 0], 1, self.ped_path[e])  # first point of each walker's path
        along_street = start_x.abs() > HALF_L + 1.0  # enters from beyond the tile's ends (not out of a door)
        free = regular & present & (mode != _STANDING) & along_street  # walkers not yet given a role

        def assign(mask: torch.Tensor, **values):
            rows, cols = mask.nonzero(as_tuple=True)
            for name, value in values.items():
                t = getattr(self, name)
                t[e[rows], cols] = value if not callable(value) else value(len(rows))
            return rows, cols

        rand = lambda lo, hi: (lambda m: lo + (hi - lo) * torch.rand(m, device=dev))  # noqa: E731
        # wheelchairs and strollers (the wide slots), blind people with a white cane
        wide = (self.slot_kind == 1)[None] & present
        assign(wide, ped_role=WHEELCHAIR, ped_radius=cfg.wide_radius, ped_height=cfg.wide_center_height,
               ped_vulnerable=True, ped_aware=True, ped_speed=rand(0.6, 1.1))
        # wheelchairs and strollers keep to the middle of their path (they are wide and stay off the curb)
        hw = torch.gather(self.path_half_width[e], 1, self.ped_path[e])
        limit = (hw - (cfg.wide_radius - cfg.pedestrian_radius)).clamp(min=0.0)
        offset = self.ped_offset[e]
        new_offset = torch.where(wide, torch.maximum(torch.minimum(offset, limit), -limit), offset)
        self.ped_offset[e] = new_offset
        from .nyc_scenario import _left, path_at

        pts, cum = self.ped_paths(e)
        on_path, u_path = path_at(pts, cum, self.ped_s[e])
        moved = wide & (self.ped_mode[e] == _WALKING)
        self.ped_pos[e] = torch.where(moved[..., None], on_path + new_offset[..., None] * _left(u_path), self.ped_pos[e])
        cane = regular & present & (torch.rand(n, p, device=dev) < self._ranged(cfg.cane_ratio, d)[:, None])
        assign(cane, ped_role=CANE, ped_vulnerable=True, ped_aware=False, ped_speed=rand(0.6, 0.9))
        free &= ~cane
        # a cyclist on the sidewalk (along the street)
        k = self._pick(free, torch.rand(n, device=dev) < self._ranged(cfg.cyclist_ratio, d))
        rows = (k >= 0).nonzero().flatten()
        if len(rows):
            ee, kk = e[rows], k[rows]
            self.ped_role[ee, kk] = CYCLIST
            self.ped_radius[ee, kk] = cfg.cyclist_radius
            self.ped_speed[ee, kk] = 3.0 + 2.0 * torch.rand(len(rows), device=dev)
            self.ped_max_speed[ee, kk] = 6.0
            self.ped_aware[ee, kk] = False
            free[rows, kk] = False

        # dogs on a leash, with an owner walking along the street
        for slot in (self.slot_kind == 2).nonzero().flatten().tolist():
            k = self._pick(free, torch.rand(n, device=dev) < self._ranged(cfg.dog_ratio, d))
            rows = (k >= 0).nonzero().flatten()
            if not len(rows):
                continue
            ee, owner = e[rows], k[rows]
            free[rows, owner] = False
            self.ped_role[ee, slot] = DOG
            self.ped_radius[ee, slot] = cfg.dog_radius
            self.ped_height[ee, slot] = cfg.dog_center_height
            self.ped_link[ee, slot] = owner
            self.ped_link_kind[ee, slot] = LEASH
            self.ped_aware[ee, slot] = True
            self.dog_angle[ee, slot] = (2 * torch.rand(len(rows), device=dev) - 1) * 1.0
            self._mirror_owner(ee, slot, owner)
            # at the end of the leash if the owner is already in view, else walking out with the owner
            visible = self.ped_mode[ee, owner] != _PARKED
            length = cfg.leash_length[0] + (cfg.leash_length[1] - cfg.leash_length[0]) * torch.rand(len(rows), device=dev)
            self.dog_len[ee, slot] = torch.where(visible, length, torch.full_like(length, 0.2))
            if visible.any():
                heading = torch.nn.functional.normalize(self.ped_vel[ee, owner] + 1e-4, dim=-1)
                c, s_ = torch.cos(self.dog_angle[ee, slot]), torch.sin(self.dog_angle[ee, slot])
                offset = torch.stack([c * heading[:, 0] - s_ * heading[:, 1], s_ * heading[:, 0] + c * heading[:, 1]], -1)
                pos = self.ped_pos[ee, owner] + self.dog_len[ee, slot][:, None] * offset
                self.ped_pos[ee, slot] = torch.where(visible[:, None], pos, self.ped_pos[ee, slot])

        # friends walking side by side: a leader and one or two followers on the same path
        groups = (torch.rand(n, device=dev) * (1 + self._ranged(cfg.groups, d))).floor().long()
        for g in range(int(groups.max()) if n else 0):
            leader = self._pick(free, groups > g)
            rows = (leader >= 0).nonzero().flatten()
            if not len(rows):
                continue
            free[rows, leader[rows]] = False
            self.ped_group[e[rows], leader[rows]] = g
            size = 1 + (torch.rand(len(rows), device=dev) < 0.35).long()  # followers
            prev = leader[rows]
            for j in range(2):
                f = self._pick(free[rows], size > j)
                ok = f >= 0
                if not ok.any():
                    break
                rr, ff, ll = rows[ok], f[ok], leader[rows][ok]
                free[rr, ff] = False
                self._join_group(e[rr], ff, ll, prev[ok], g, side=1.0 if j == 0 else -1.0)
                prev = torch.where(ok, f, prev)

        # lines of people waiting (at a shop door, a bus stop, a kiosk)
        self._spawn_queues(e, free, d)

        # incidents: an attack, someone blocking the way, or (see AssistMixin) someone who has fallen
        u = torch.rand(n, device=dev)
        p_attack, p_block = self._ranged(cfg.attack_ratio, d), self._ranged(cfg.block_ratio, d)
        incident = torch.where(u < p_attack, ATTACK, torch.where(u < p_attack + p_block, BLOCK, NO_INCIDENT))
        self.incident[e] = incident
        self.incident_phase[e] = torch.where(incident != NO_INCIDENT, PENDING, DONE)
        self.incident_start[e] = cfg.incident_time[0] + (cfg.incident_time[1] - cfg.incident_time[0]) * torch.rand(n, device=dev)
        self.attack_speed[e] = self._ranged(cfg.attack_speed, d) * (0.85 + 0.3 * torch.rand(n, device=dev))
        self.shove_max[e] = self._ranged(cfg.shove_force_max, d)
        # walking velocities with the new speeds
        from .nyc_scenario import path_at

        pts, cum = self.ped_paths(e)
        _, u_dir = path_at(pts, cum, self.ped_s[e])
        walking = (self.ped_mode[e] == _WALKING) & (self.ped_role[e] != DOG)
        self.ped_vel[e] = torch.where(walking[..., None], self.ped_speed[e][..., None] * u_dir, self.ped_vel[e] * (self.ped_role[e] == DOG)[..., None])

    def _pick(self, candidates: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
        """A random candidate slot per row (n, p) where ``want`` (n,), else -1."""
        score = torch.where(candidates, torch.rand(candidates.shape, device=self.device), torch.full(candidates.shape, -1.0, device=self.device))
        best = score.argmax(dim=1)
        ok = want & candidates.any(dim=1)
        return torch.where(ok, best, torch.full_like(best, -1))

    def _mirror_owner(self, envs: torch.Tensor, slot: int, owner: torch.Tensor):
        """A dog takes part when its owner does: same mode, timer, path and position (until it walks out to the end
        of its leash)."""
        self.ped_mode[envs, slot] = self.ped_mode[envs, owner]
        self.ped_timer[envs, slot] = self.ped_timer[envs, owner]
        self.ped_path[envs, slot] = self.ped_path[envs, owner]
        self.ped_s[envs, slot] = self.ped_s[envs, owner]
        self.ped_offset[envs, slot] = self.ped_offset[envs, owner]
        self.ped_pos[envs, slot] = self.ped_pos[envs, owner]
        self.ped_vel[envs, slot] = self.ped_vel[envs, owner]

    def _join_group(self, envs, follower, leader, prev, g: int, side: float):
        """``follower`` walks beside ``leader``: same path, position along it, speed, timing and attention."""
        for name in ("ped_mode", "ped_timer", "ped_path", "ped_s", "ped_speed", "ped_aware"):
            t = getattr(self, name)
            t[envs, follower] = t[envs, leader]
        half_width = self.path_half_width[envs, self.ped_path[envs, leader]]
        offset = self.ped_offset[envs, leader] + side * self.cfg.group_spacing
        limit = half_width + 0.3
        self.ped_offset[envs, follower] = offset.clamp(-limit, limit)
        # place it on the path at the leader's position along it, at its own offset
        from .nyc_scenario import _left, path_at

        pts = self.path_pts[envs, self.ped_path[envs, follower]]
        cum = self.path_cum[envs, self.ped_path[envs, follower]]
        on_path, u = path_at(pts, cum, self.ped_s[envs, follower])
        pos = on_path + self.ped_offset[envs, follower][:, None] * _left(u)
        self.ped_pos[envs, follower] = torch.where((self.ped_mode[envs, leader] == _WALKING)[:, None], pos, self.ped_pos[envs, leader])
        self.ped_group[envs, follower] = g
        self.ped_link[envs, follower] = prev
        self.ped_link_kind[envs, follower] = GROUP

    def _spawn_queues(self, e: torch.Tensor, free: torch.Tensor, d: torch.Tensor):
        cfg, dev = self.cfg, self.device
        lay = self.layout
        r, c = self.tile_row[e], self.tile_col[e]
        queues = lay.queues[r, c]  # (n, Q, 5)
        valid = torch.arange(MAX_QUEUES, device=dev) < lay.num_queues[r, c, None]
        # people for the line: regular slots that are present or would join later (any role-free one)
        p = self.num_peds
        regular = self._is_regular()[None].expand(len(e), p)
        pool = regular & (self.ped_role[e] == NORMAL) & (self.ped_group[e] < 0) & free | (
            regular & (self.ped_role[e] == NORMAL) & (self.ped_group[e] < 0) & (self.ped_mode[e] == _STANDING)
        )
        # slots beyond the crowd budget are free to use too
        pool |= regular & (self.ped_mode[e] == _PARKED) & (self.ped_timer[e] > 1e8)
        for q in range(MAX_QUEUES):
            want = valid[:, q] & (torch.rand(len(e), device=dev) < self._ranged(cfg.queue_ratio, d))
            head, step, count = queues[:, q, :2], queues[:, q, 2:4], queues[:, q, 4].long()
            prev = torch.full((len(e),), -1, dtype=torch.long, device=dev)
            alive = want.clone()
            for i in range(int(count.max()) if len(e) else 0):
                alive &= i < count
                spot = head + (i * cfg.queue_spacing) * step + 0.08 * (torch.rand(len(e), 2, device=dev) - 0.5)
                clear = self.obstacle_clearance(spot[:, None], radius=cfg.pedestrian_radius, env_ids=e)[:, 0] > 0.05
                clear &= torch.linalg.norm(spot - self.start[e], dim=1) > 1.3
                clear &= torch.linalg.norm(spot - self.goal[e], dim=1) > 1.3
                alive &= clear  # the line ends where it would block a structure or the robot's start or goal
                k = self._pick(pool, alive)
                rows = (k >= 0).nonzero().flatten()
                if not len(rows):
                    break
                ee, kk = e[rows], k[rows]
                pool[rows, kk] = False
                free[rows, kk] = False
                self.ped_role[ee, kk] = QUEUE
                self.ped_mode[ee, kk] = _STANDING
                self.ped_timer[ee, kk] = 1e9  # they stay in line
                self.ped_pos[ee, kk] = spot[rows]
                self.ped_vel[ee, kk] = 0.0
                self.ped_aware[ee, kk] = True
                has_prev = prev[rows] >= 0
                self.ped_link[ee[has_prev], kk[has_prev]] = prev[rows][has_prev]
                self.ped_link_kind[ee[has_prev], kk[has_prev]] = LINE
                prev[rows] = kk
                alive &= k >= 0
        # line members are placed directly: project them onto some path so that ``ped_s`` is defined
        from .nyc_scenario import path_project

        pts, cum = self.ped_paths(e)
        self.ped_s[e] = torch.where(self.ped_role[e] == QUEUE, path_project(pts, cum, self.ped_pos[e]), self.ped_s[e])

    # -- per step (inside the pedestrian planner) -----------------------------------------------------------------------

    def social_masks(self) -> dict:
        """Per-pedestrian behaviour switches for the planner."""
        active = (self.incident_phase == ACTIVE)[:, None] & (torch.arange(self.num_peds, device=self.device)[None] == self.incident_ped[:, None])
        return {
            "dog": self.ped_role == DOG,
            "queue": self.ped_role == QUEUE,
            "cyclist": self.ped_role == CYCLIST,
            "incident": active,  # attacker or blocker, while after the robot
            "attacker": active & (self.incident == ATTACK)[:, None],
        }

    def social_desired_velocity(self, v_des: torch.Tensor, masks: dict, dt: float) -> torch.Tensor:
        """Desired velocities of dogs (around their owner), and of an attacker or blocker (at the robot)."""
        n, p, dev = self.num_envs, self.num_peds, self.device
        dog = masks["dog"]
        if dog.any():
            owner = self.ped_link.clamp(min=0)
            owner_pos = torch.gather(self.ped_pos, 1, owner[..., None].expand(n, p, 2))
            owner_vel = torch.gather(self.ped_vel, 1, owner[..., None].expand(n, p, 2))
            speed = torch.linalg.norm(owner_vel, dim=-1, keepdim=True)
            heading = torch.where(speed > 0.1, owner_vel / speed.clamp(min=1e-3), torch.nn.functional.normalize(self.ped_pos - owner_pos + 1e-3, dim=-1))
            # the dog sniffs around: its bearing and distance from the owner wander, within the leash
            self.dog_angle += dog * (2 * torch.rand(n, p, device=dev) - 1) * 1.5 * dt
            self.dog_angle.clamp_(-1.3, 1.3)
            target_len = self.cfg.leash_length[0] + (self.cfg.leash_length[1] - self.cfg.leash_length[0]) * (0.5 + 0.5 * torch.sin(self.dog_angle * 2.3))
            self.dog_len += dog * (target_len - self.dog_len).clamp(-0.6 * dt, 0.6 * dt)
            def leash_target(angle):
                c, s = torch.cos(angle), torch.sin(angle)
                offset = torch.stack([c * heading[..., 0] - s * heading[..., 1], s * heading[..., 0] + c * heading[..., 1]], -1)
                return owner_pos + self.dog_len[..., None] * offset

            def blocked(q):  # in a structure or out in the roadway
                local = self.tile_local(q)
                return (self.obstacle_clearance(local, radius=self.cfg.dog_radius) < 0.05) | self._street_mask(local)

            # the dog stays on the sidewalk: to the owner's other side if one side is blocked, else behind the owner
            target = leash_target(self.dog_angle)
            flip = dog & blocked(target)
            if flip.any():
                self.dog_angle = torch.where(flip, -self.dog_angle, self.dog_angle)
                target = leash_target(self.dog_angle)
                behind = owner_pos - 0.8 * heading
                target = torch.where((dog & blocked(target))[..., None], behind, target)
            v = owner_vel + 2.5 * (target - self.ped_pos)
            # a dog that has got into the roadway or between posts comes back to its owner
            stray = dog & blocked(self.ped_pos)
            v = torch.where(stray[..., None], 3.0 * (owner_pos - 0.5 * heading - self.ped_pos), v)
            v = v * (self.cfg.dog_max_speed / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=self.cfg.dog_max_speed))
            v_des = torch.where(dog[..., None], v, v_des)
        group = (self.ped_link_kind == GROUP) & (self.ped_mode == _WALKING)
        if group.any():
            # people walking together keep level with each other: speed up or slow down along the shared path
            partner = self.ped_link.clamp(min=0)
            same_path = torch.gather(self.ped_path, 1, partner) == self.ped_path
            ds = torch.gather(self.ped_s, 1, partner) - self.ped_s
            from .nyc_scenario import path_at

            pts, cum = self.ped_paths()
            _, u = path_at(pts, cum, self.ped_s)
            keep_up = (group & same_path)[..., None] * (0.8 * ds.clamp(-1.0, 1.0))[..., None] * u
            v_des = v_des + keep_up
            # a group that ended up on different paths (after leaving the scene) has split up
            split = group & ~same_path
            if split.any():
                self.ped_link[split] = -1
                self.ped_link_kind[split] = NO_LINK
                self.ped_group[split] = -1
        incident = masks["incident"]
        if incident.any():
            to_robot = self.robot_pos()[:, None] - self.ped_pos
            robot_vel = self.robot_vel()[:, None]
            attacker = masks["attacker"]
            # attacker: at the robot (where it is going); blocker: to a spot 1 m in front of the robot, towards its goal
            chase = to_robot + 0.4 * robot_vel
            v_attack = self.attack_speed[:, None, None] * torch.nn.functional.normalize(chase, dim=-1)
            goal_dir = torch.nn.functional.normalize(self.goal - self.robot_pos() + 1e-3, dim=-1)[:, None]
            spot = self.robot_pos()[:, None] + self.cfg.block_distance * goal_dir
            v_block = 2.0 * (spot - self.ped_pos)
            v_block = v_block * (1.3 / torch.linalg.norm(v_block, dim=-1, keepdim=True).clamp(min=1.3))
            v_des = torch.where(incident[..., None], torch.where(attacker[..., None], v_attack, v_block), v_des)
        return v_des

    def keep_wide_offsets(self):
        """Wheelchairs and strollers keep to the middle of their path (also on the paths they take when re-entering)."""
        wide = self.ped_role == WHEELCHAIR
        if not wide.any():
            return
        hw = torch.gather(self.path_half_width, 1, self.ped_path)
        limit = (hw - (self.cfg.wide_radius - self.cfg.pedestrian_radius)).clamp(min=0.0)
        self.ped_offset = torch.where(wide, torch.maximum(torch.minimum(self.ped_offset, limit), -limit), self.ped_offset)

    def pair_force_mask(self) -> torch.Tensor:
        """No repulsion between members of the same group, and between a dog and its owner."""
        same_group = (self.ped_group[:, :, None] == self.ped_group[:, None, :]) & (self.ped_group[:, :, None] >= 0)
        dog_owner = (self.ped_link_kind == LEASH)[:, :, None] & (self.ped_link[:, :, None] == torch.arange(self.num_peds, device=self.device)[None, None])
        return ~(same_group | dog_owner | dog_owner.transpose(1, 2))

    def social_after_planning(self, dt: float):
        """Dogs keep to their owner's timing; incidents start, run and end; shoves and kicks."""
        n, dev = self.num_envs, self.device
        self.episode_time += dt
        dog = self.ped_role == DOG
        if dog.any():
            envs, slots = dog.nonzero(as_tuple=True)
            owner = self.ped_link[envs, slots]
            owner_mode = self.ped_mode[envs, owner]
            parked = owner_mode == _PARKED
            # out of view with the owner (beyond the tile's end, or behind a door) and back with it
            self.ped_mode[envs, slots] = torch.where(parked, _PARKED, _WALKING)
            self.ped_timer[envs, slots] = self.ped_timer[envs, owner]
            self.ped_path[envs, slots] = self.ped_path[envs, owner]
            self.ped_s[envs, slots] = self.ped_s[envs, owner]
            self.ped_pos[envs[parked], slots[parked]] = self.ped_pos[envs[parked], owner[parked]]
            self.dog_len[envs[parked], slots[parked]] = 0.2
        self._update_incidents(dt)
        self._update_links()

    def _update_incidents(self, dt: float):
        cfg, n, p, dev = self.cfg, self.num_envs, self.num_peds, self.device
        envs = torch.arange(n, device=dev)
        rel = self.ped_pos - self.robot_pos()[:, None]
        dist = torch.linalg.norm(rel, dim=-1)
        # start: the incident's person is someone walking nearby (an ordinary regular walker)
        pending = (self.incident_phase == PENDING) & (self.episode_time >= self.incident_start)
        if pending.any():
            ok = (self.ped_role == NORMAL) & (self.ped_group < 0) & (self.ped_mode == _WALKING) & self._is_regular()[None]
            ok &= (dist > cfg.incident_distance[0]) & (dist < cfg.incident_distance[1])
            ok &= self.ped_link < 0  # not a dog, a group follower or in a line
            is_owner = torch.zeros_like(ok)
            leash = self.ped_link_kind == LEASH
            if leash.any():
                le, ls = leash.nonzero(as_tuple=True)
                is_owner[le, self.ped_link[le, ls]] = True
            ok &= ~is_owner
            k = self._pick(ok, pending)
            go = k >= 0
            self.incident_ped[go] = k[go]
            self.incident_phase[go] = ACTIVE
            duration = torch.where(self.incident == ATTACK, self._ranged(cfg.chase_time, self.difficulty), self._ranged(cfg.block_time, self.difficulty))
            self.incident_timer[go] = (duration * (0.7 + 0.6 * torch.rand(n, device=dev)))[go]
            self.hostile[envs[go], k[go]] = True
            self.attacker[envs[go], k[go]] = (self.incident == ATTACK)[go]
            self.ped_aware[envs[go], k[go]] = True
            self.attacked_episode |= go & (self.incident == ATTACK)
            # give up waiting for someone after a while
            late = pending & ~go & (self.episode_time > self.incident_start + 10.0)
            self.incident_phase[late] = DONE
        active = self.incident_phase == ACTIVE
        k = self.incident_ped.clamp(min=0)
        gap = dist[envs, k] - self.ped_radius[envs, k] - self.cfg.robot_radius
        # attacks: shoves and kicks while within reach
        self.shoved.zero_()
        self.shove_left -= dt
        self.shove_cooldown -= dt
        self.attack_force[self.shove_left <= 0.0] = 0.0
        attacking = active & (self.incident == ATTACK)
        hit = attacking & (gap < cfg.reach) & (self.shove_cooldown <= 0.0)
        if hit.any():
            m = int(hit.sum())
            away = -rel[envs[hit], k[hit]] / dist[envs[hit], k[hit]].clamp(min=1e-3)[:, None]  # from attacker to robot
            spread = (2 * torch.rand(m, device=dev) - 1) * 0.5
            c, s = torch.cos(spread), torch.sin(spread)
            away = torch.stack([c * away[:, 0] - s * away[:, 1], s * away[:, 0] + c * away[:, 1]], dim=1)
            kick = torch.rand(m, device=dev) < cfg.kick_probability
            magnitude = torch.where(
                kick,
                cfg.kick_force[0] + (cfg.kick_force[1] - cfg.kick_force[0]) * torch.rand(m, device=dev),
                cfg.shove_force_min + (self.shove_max[hit] - cfg.shove_force_min) * torch.rand(m, device=dev),
            )
            body = torch.where(kick, 1 + torch.randint(0, 2, (m,), device=dev), torch.zeros(m, dtype=torch.long, device=dev))
            force = torch.zeros(m, 3, 3, device=dev)
            force[torch.arange(m, device=dev), body, :2] = magnitude[:, None] * away
            self.attack_force[hit] = force
            self.shove_left[hit] = torch.where(kick, 0.08 + 0.07 * torch.rand(m, device=dev), 0.1 + 0.1 * torch.rand(m, device=dev))
            self.shove_cooldown[hit] = cfg.shove_interval[0] + (cfg.shove_interval[1] - cfg.shove_interval[0]) * torch.rand(m, device=dev)
            self.shoved[hit] = 1.0
            self.episode_shoves[hit] += 1.0
        # end: the person gives up and walks off along the nearest path leading away from the robot
        self.incident_timer -= dt * active
        end = active & (self.incident_timer <= 0.0)
        if end.any():
            self._leave(end.nonzero().flatten())
        # the threat is over once the attacker has left (or is far)
        leaving = self.incident_phase == LEAVING
        far = dist[envs, k] > 8.0
        gone = leaving & (far | (self.ped_mode[envs, k] == _PARKED))
        self.hostile[envs[gone], k[gone]] = False
        self.attacker[envs[gone], k[gone]] = False
        self.incident_phase[gone] = DONE
        self.under_attack = (self.incident == ATTACK) & ((self.incident_phase == ACTIVE) | (self.incident_phase == LEAVING))

    def _leave(self, rows: torch.Tensor):
        """The incident's person walks away: onto the path whose nearest point is closest, in the direction away from
        the robot, briskly."""
        from .nyc_scenario import path_at, path_project

        k = self.incident_ped[rows]
        pos = self.ped_pos[rows, k]  # (m, 2)
        pts, cum = self.path_pts[rows], self.path_cum[rows]  # (m, P, K, 2)
        s = path_project(pts, cum, pos[:, None].expand(-1, pts.shape[1], 2))
        on, u = path_at(pts, cum, s)
        d = torch.linalg.norm(on - pos[:, None], dim=-1)
        towards = ((u * (self.robot_pos()[rows][:, None] - pos[:, None])).sum(-1) > 0).float()
        score = torch.where(self.path_valid[rows], d + 4.0 * towards, torch.full_like(d, 1e6))
        best = score.argmin(dim=1)
        m = torch.arange(len(rows), device=self.device)
        self.ped_path[rows, k] = best
        self.ped_s[rows, k] = s[m, best]
        left = torch.stack([-u[m, best, 1], u[m, best, 0]], dim=-1)
        offset = ((pos - on[m, best]) * left).sum(-1)
        hw = self.path_half_width[rows, best]
        self.ped_offset[rows, k] = torch.maximum(torch.minimum(offset, hw), -hw)
        self.ped_speed[rows, k] = 1.1 + 0.4 * torch.rand(len(rows), device=self.device)
        self.ped_mode[rows, k] = _WALKING
        self.incident_phase[rows] = LEAVING

    def _update_links(self):
        """Events: the robot's footprint crosses a leash (owner-dog), or passes between people walking together or
        waiting in line."""
        n, p = self.num_envs, self.num_peds
        linked = self.ped_link >= 0
        self.leash_crossed[:] = False
        self.split_event.zero_()
        if not linked.any():
            return
        other = self.ped_link.clamp(min=0)
        a = self.ped_pos
        b = torch.gather(self.ped_pos, 1, other[..., None].expand(n, p, 2))
        visible = linked & (self.ped_mode != _PARKED) & torch.gather(self.ped_mode != _PARKED, 1, other)
        span = torch.linalg.norm(b - a, dim=-1)
        limit = torch.where(self.ped_link_kind == LEASH, 2.6, torch.where(self.ped_link_kind == GROUP, 1.6, 1.4))
        visible &= span < limit
        d = _segment_distance(self.robot_pos()[:, None], a, b)
        inside = visible & (d < self.cfg.robot_radius)
        new = inside & ~self._link_inside
        self._link_inside = inside
        self.leash_crossed[:] = (inside & (self.ped_link_kind == LEASH)).any(dim=1)
        self.split_event[:] = (new & (self.ped_link_kind != LEASH)).float().sum(dim=1)

    # -- queries ------------------------------------------------------------------------------------------------------

    def link_vectors(self) -> torch.Tensor:
        """Vector (n, p, 2) from each pedestrian to the one it is linked to (dog-owner, group, line); zero if none."""
        n, p = self.num_envs, self.num_peds
        other = self.ped_link.clamp(min=0)
        b = torch.gather(self.ped_pos, 1, other[..., None].expand(n, p, 2))
        return torch.where((self.ped_link >= 0)[..., None], b - self.ped_pos, torch.zeros_like(b))

    def threat(self) -> torch.Tensor:
        """Distance (m, gap between bodies) to the attacker while there is one, else a large value (n,)."""
        envs = torch.arange(self.num_envs, device=self.device)
        k = self.incident_ped.clamp(min=0)
        d = torch.linalg.norm(self.ped_pos[envs, k] - self.robot_pos(), dim=-1) - self.ped_radius[envs, k] - self.cfg.robot_radius
        return torch.where(self.under_attack, d, torch.full_like(d, 1e3))
