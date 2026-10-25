"""The safety shield in the loop of simulated B2-Ws, as on the robot (``deploy/b2w/navigator.py``): used by the
evaluation (``scripts/b2w/evaluate.py --shield``) and the demo (``scripts/demo/b2w_demo.py --shield``).

The shield (:class:`g1_patrol.shield.ShieldGuard`) is fed by the simulated lidar and depth cameras and, for a student,
by its height map: the map's obstacle cells (low things beside the robot that left the sensors' view) and its history
(what moved), with the floor as the student measures it. The policy's command is capped at the episode's speed limit
first, as the robot's ``max_speed`` does.
"""

from __future__ import annotations

import torch

from g1_patrol.b2w import MAP, shield_settings
from g1_patrol.heightmap import rotate_inverse
from g1_patrol.shield import ShieldCfg, ShieldGuard, map_points, moving_returns


def _settings(text: str | None) -> dict:
    """``key=value,...`` -> {key: float or tuple of floats (``a:b:c``)}."""
    pairs = dict(kv.split("=") for kv in text.split(",")) if text else {}
    return {k: tuple(float(x) for x in v.split(":")) if ":" in v else float(v) for k, v in pairs.items()}


class ShieldInLoop:
    """``actions, info = shield_in_loop(actions, obs)``: the policy's actions (pre-tanh, as the base action term takes
    them) for the commands the shield lets through, and the guard's diagnostics.

    ``shield_set``/``guard_set``: settings other than the defaults of :class:`g1_patrol.shield.ShieldCfg` and
    :class:`g1_patrol.shield.ShieldGuard` (``key=value,...``); ``person_oracle``: a perfect person detector marks the
    returns from pedestrians (an upper bound for a camera detector).
    """

    def __init__(self, base_env, shield_set: str | None = None, guard_set: str | None = None, person_oracle: bool = False):
        self.env = base_env
        self.base = base_env.action_manager.get_term("base")
        self.scenario = base_env.command_manager.get_term("nyc")
        self.cfg = ShieldCfg(**shield_settings(), **{k: type(getattr(ShieldCfg, k))(v) for k, v in _settings(shield_set).items()})
        # the shield with what the robot does around it (alternatives, backing off, crawling near people)
        self.guard = ShieldGuard(base_env.num_envs, self.cfg, base_env.device, **_settings(guard_set))
        om = base_env.observation_manager
        self.group = next((g for g in ("lidar_map", "height_map") if g in om.active_terms), None)
        if self.group is not None:  # the student's height map (its history tells what moved)
            self.height_map_term = om._group_obs_term_cfgs[self.group][om.active_terms[self.group].index(self.group)].func
        self.sensors = [base_env.scene.sensors[name] for name in ("lidar", "front_depth", "rear_depth")
                        if name in base_env.scene.sensors]
        self.person_oracle = person_oracle

    def reset(self, dones: torch.Tensor):
        self.guard.reset(dones.bool())

    def __call__(self, actions: torch.Tensor, obs) -> tuple[torch.Tensor, dict]:
        base, scenario, group = self.base, self.scenario, self.group
        robot = self.env.scene["robot"]
        root, quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        hits = torch.cat([(sn.data.ray_hits_w if hasattr(sn.data, "ray_hits_w") else sn.ray_hits_w).torch for sn in self.sensors], dim=1)
        hits = torch.nan_to_num(hits, nan=float("inf"), posinf=float("inf"))
        local = rotate_inverse(quat[:, None].expand(-1, hits.shape[1], -1), torch.nan_to_num(hits - root[:, None], posinf=1e4))
        local = torch.where((hits.abs() > 1e6).any(dim=-1, keepdim=True), torch.full_like(local, float("inf")), local)
        moving = None
        # the floor below the base as the robot measures it (the student's map), else the true one
        ground_z = -(self.height_map_term.ground_offset if group is not None else base.base_height)
        if group is not None:
            moving = moving_returns(local, obs[group], MAP["resolution"], ground_z,
                                    tall_age=self.height_map_term.map.tall_age_out, recent=self.cfg.still_for,
                                    min_cells=self.cfg.moving_cells)
            cells = map_points(obs[group][:, 0], MAP["resolution"], ground_z)
            local = torch.cat([local, cells], dim=1)
            moving = torch.cat([moving, torch.zeros(cells.shape[:2], dtype=torch.bool, device=cells.device)], dim=1)
        command = base._mid + base._half * torch.tanh(actions)
        speed = torch.linalg.norm(command[:, :2], dim=1, keepdim=True)  # the speed cap first, as on the robot
        command[:, :2] *= (base.speed_limit[:, None] / speed.clamp(min=1e-6)).clamp(max=1.0)
        person = None
        if self.person_oracle:  # a perfect person detector: the returns from pedestrians (an upper bound)
            ped = scenario.ped_pos + scenario.tile_center[:, None]
            d = torch.linalg.norm(hits[:, :, None, :2] - ped[:, None], dim=-1)
            d = torch.where(scenario.visible_peds()[:, None], d, torch.full_like(d, 1e3)).amin(dim=-1)
            person = d < scenario.cfg.pedestrian_radius + 0.1
            if moving is not None:
                person = torch.cat([person, torch.zeros(person.shape[0], local.shape[1] - person.shape[1],
                                                        dtype=torch.bool, device=person.device)], dim=1)
        safe, info = self.guard(local, command, self.env.step_dt, base.velocity_b, moving, person, ground_z)
        return torch.atanh(((safe - base._mid) / base._half).clamp(-0.999, 0.999)), info
