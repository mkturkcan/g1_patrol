"""Observation, reward, termination and curriculum terms of the civic navigation tasks (litter and trash bags, special
pedestrians, attacks, a fallen person, crosswalk signals)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .nyc_social import DOG
from .nyc_terms import nyc_adaptive_curriculum
from .nyc_trash import ON_GROUND

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

    from .nyc_manipulation import HandsAction
    from .nyc_scenario import NycScenarioCommand


def _scenario(env: ManagerBasedRLEnv, command_name: str) -> NycScenarioCommand:
    return env.command_manager.get_term(command_name)


def _hands(env: ManagerBasedRLEnv, action_name: str) -> HandsAction:
    return env.action_manager.get_term(action_name)


def _rotate(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], dim=-1)


"""
Privileged observations (teacher).
"""


def civic_pedestrians(env: ManagerBasedRLEnv, command_name: str, count: int = 10, max_range: float = 8.0) -> torch.Tensor:
    """Nearest ``count`` pedestrians (and dogs): relative position and velocity (heading frame), presence, body radius,
    needs room (wheelchair, stroller, white cane), attacking, blocking the robot's way, and the vector to the person or
    dog it is linked to (leash, walking together, waiting in line). Shape (n, 11 * count)."""
    s = _scenario(env, command_name)
    n = s.num_envs
    rel = s.ped_pos - s.robot_pos()[:, None]
    dist = torch.where(s.visible_peds(), torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e3))
    order = dist.argsort(dim=1)[:, :count]

    def take(t):
        return torch.gather(t, 1, order) if t.dim() == 2 else torch.gather(t, 1, order[..., None].expand(-1, -1, t.shape[-1]))

    yaw = s.robot_yaw()[:, None]
    near = (take(dist) < max_range).float()[..., None]
    vel = take(s.ped_vel) - s.robot_vel()[:, None]
    feats = torch.cat(
        [
            _rotate(take(rel), yaw),
            _rotate(vel, yaw),
            torch.ones_like(near),
            take(s.ped_radius)[..., None],
            take(s.ped_vulnerable).float()[..., None],
            take(s.attacker).float()[..., None],
            take(s.hostile & ~s.attacker).float()[..., None],
            _rotate(take(s.link_vectors()), yaw),
        ],
        dim=-1,
    )
    return (feats * near).reshape(n, -1)


def trash_state(env: ManagerBasedRLEnv, command_name: str, count: int = 3) -> torch.Tensor:
    return _scenario(env, command_name).nearest_trash(count)


def basket_state(env: ManagerBasedRLEnv, command_name: str, count: int = 2) -> torch.Tensor:
    return _scenario(env, command_name).nearest_baskets(count)


def pile_state(env: ManagerBasedRLEnv, command_name: str, count: int = 1) -> torch.Tensor:
    return _scenario(env, command_name).nearest_piles(count)


def fallen_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _scenario(env, command_name).fallen_features()


def signal_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _scenario(env, command_name).signal_features()


def hands_state(env: ManagerBasedRLEnv, action_name: str = "hands") -> torch.Tensor:
    """Holding litter, holding a bag, busy (picking, putting down, calling), bracing. (n, 4)."""
    h = _hands(env, action_name)
    holding, bag = h.holding(), h.holding_bag()
    return torch.stack([(holding & ~bag).float(), bag.float(), h.busy.float(), h.bracing.float()], dim=1)


def attack_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Force of the current shove on the torso (heading frame, / 100 N) and whether an attack is going on. (n, 3)."""
    s = _scenario(env, command_name)
    f = _rotate(s.attack_force[:, 0, :2], s.robot_yaw()) / 100.0
    return torch.cat([f, s.under_attack.float()[:, None]], dim=1)


"""
Rewards.
"""


def civic_goal_progress(env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.0) -> torch.Tensor:
    """Goal progress (m/s, clipped), not while an attacker is after the robot (getting away comes first)."""
    s = _scenario(env, command_name)
    progress = ((s.prev_goal_dist - s.goal_distance()) / env.step_dt).clamp(-max_speed, max_speed)
    return progress * ~s.under_attack


def civic_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return civic_success_termination(env, command_name).float()


def hands_event(env: ManagerBasedRLEnv, event: str, action_name: str = "hands") -> torch.Tensor:
    """Per-step events of the hands controller: ``picked``, ``failed`` (grasp missed), ``invalid`` (trigger with
    nothing to do), ``dropped``, ``alerted``."""
    return getattr(_hands(env, action_name), event)


def trash_event(env: ManagerBasedRLEnv, command_name: str, event: str) -> torch.Tensor:
    """Per-step litter events of the scenario: ``trash_deposited`` (in a basket), ``bags_delivered`` (beside a bag
    pile), ``trash_lost`` (into the roadway)."""
    return getattr(_scenario(env, command_name), event)


def litter_clawback(
    env: ManagerBasedRLEnv,
    command_name: str,
    values: tuple[float, float, float],
    failure_terms: tuple[str, ...] = ("pedestrian_collision", "obstacle_collision", "fall", "roadway"),
) -> torch.Tensor:
    """At a failure (contact, fall, the roadway), what the episode's litter work earned (``values``: the reward weights
    of an item picked up, an item in a basket, a bag set down), so that with weight -1 it is taken back: litter work
    pays only in episodes that end well, and cashing it in before a collision is no trade."""
    s = _scenario(env, command_name)
    tm = env.termination_manager
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in failure_terms:
        if name in tm.active_terms:
            failed |= tm.get_term(name)
    earned = values[0] * s.episode_picked + values[1] * s.episode_deposited + values[2] * s.episode_delivered
    return earned * failed


class trash_progress(ManagerTermBase):
    """Progress (m/s, clipped) towards the next step of clearing litter: towards the nearest item on the ground (from
    the reach zone in front of the robot) while empty-handed, towards a place to put it while holding something (a
    spot 0.75 m from a basket's center, the zone beside a bag pile). Only within ``max_distance``, not while picking or
    putting down, attacked or with a fallen person to help, and not across a change of target (picking up, putting
    down, another item becoming the nearest)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._key = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._key[ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        action_name: str = "hands",
        max_distance: float = 5.0,
        max_speed: float = 1.0,
    ) -> torch.Tensor:
        s, h = _scenario(env, command_name), _hands(env, action_name)
        n, dev = env.num_envs, env.device
        if s.num_trash == 0:
            return torch.zeros(n, device=dev)
        p, yaw = s.robot_pos(), s.robot_yaw()
        reach = p + 0.35 * torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        items = s.trash_positions()[..., :2]
        d_item = torch.where(s.trash_state == ON_GROUND, torch.linalg.norm(items - reach[:, None], dim=-1), torch.full_like(items[..., 0], 1e3))
        # empty-handed: an item, but only litter while there is a basket and only bags while there is a bag zone
        usable = torch.where(s.is_bag[None], s.pile_valid.any(dim=1, keepdim=True), s.basket_valid.any(dim=1, keepdim=True))
        d_item = torch.where(usable, d_item, torch.full_like(d_item, 1e3))
        item_d, item_k = d_item.min(dim=1)
        d_basket = torch.linalg.norm(s.baskets[..., :2] - p[:, None], dim=-1) - 0.75
        d_basket = torch.where(s.basket_valid, d_basket.abs(), torch.full_like(d_basket, 1e3))
        basket_d, basket_k = d_basket.min(dim=1)
        from .nyc_trash import _rect_point

        d_pile = torch.linalg.norm(_rect_point(p, s.piles) - p[:, None], dim=-1)
        d_pile = torch.where(s.pile_valid, (d_pile - 0.3).clamp(min=0.0), torch.full_like(d_pile, 1e3))
        pile_d, pile_k = d_pile.min(dim=1)
        holding, bag = h.holding(), h.holding_bag()
        d = torch.where(holding, torch.where(bag, pile_d, basket_d), item_d)
        key = torch.where(holding, torch.where(bag, 200 + pile_k, 100 + basket_k), item_k)
        progress = ((self._prev - d) / env.step_dt).clamp(-max_speed, max_speed)
        on = (key == self._key) & (d < max_distance) & (self._prev < max_distance) & ~h.busy
        if s.social:
            on &= ~s.under_attack & ~s.has_fallen
        self._prev[:] = d
        self._key[:] = key
        return progress * on


def flee(env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.5) -> torch.Tensor:
    """Increase (m/s, clipped) of the distance to an attacker who is after the robot."""
    s = _scenario(env, command_name)
    envs = torch.arange(env.num_envs, device=env.device)
    k = s.incident_ped.clamp(min=0)
    rel = s.robot_pos() - s.ped_pos[envs, k]
    u = rel / torch.linalg.norm(rel, dim=1, keepdim=True).clamp(min=1e-3)
    away = ((s.robot_vel() - s.ped_vel[envs, k]) * u).sum(-1)
    return away.clamp(-max_speed, max_speed) * s.under_attack


def threat_proximity(env: ManagerBasedRLEnv, command_name: str, distance: float = 1.5) -> torch.Tensor:
    """How far (m) an attacker is inside ``distance`` of the robot (gap between bodies)."""
    return (distance - _scenario(env, command_name).threat()).clamp(min=0.0)


def approach_threat(env: ManagerBasedRLEnv, command_name: str, distance: float = 3.0) -> torch.Tensor:
    """Robot speed (m/s) towards an attacker within ``distance``: the robot never goes at someone who attacks it."""
    s = _scenario(env, command_name)
    envs = torch.arange(env.num_envs, device=env.device)
    rel = s.ped_pos[envs, s.incident_ped.clamp(min=0)] - s.robot_pos()
    u = rel / torch.linalg.norm(rel, dim=1, keepdim=True).clamp(min=1e-3)
    towards = (s.robot_vel() * u).sum(-1).clamp(min=0.0)
    return towards * (s.threat() < distance)


def brace_under_threat(env: ManagerBasedRLEnv, command_name: str, action_name: str = "hands", distance: float = 1.0) -> torch.Tensor:
    """Braced while an attacker is within ``distance`` (gap between bodies)."""
    return (_hands(env, action_name).bracing & (_scenario(env, command_name).threat() < distance)).float()


def brace_without_threat(env: ManagerBasedRLEnv, command_name: str, action_name: str = "hands", distance: float = 3.0) -> torch.Tensor:
    """Braced with no attacker within ``distance``: a guard looks hostile to everyone else."""
    return (_hands(env, action_name).bracing & (_scenario(env, command_name).threat() > distance)).float()


def shoved(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Shoves and kicks that hit the robot at this step."""
    return _scenario(env, command_name).shoved


def vulnerable_space(env: ManagerBasedRLEnv, command_name: str, comfort_distance: float = 1.2) -> torch.Tensor:
    """How far (m) the robot is inside the (larger) personal space of someone in a wheelchair, with a stroller or a
    white cane."""
    s = _scenario(env, command_name)
    return (comfort_distance - s.pedestrian_clearance(only=s.ped_vulnerable)).clamp(min=0.0)


def vulnerable_delay(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Walking speed (m/s) that people who need more room give up because the robot is in their way."""
    s = _scenario(env, command_name)
    return (s.ped_delay * s.ped_vulnerable).sum(dim=1)


def split_group(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The robot passed between people walking together or cut through a line."""
    return _scenario(env, command_name).split_event


def walking_speed(env: ManagerBasedRLEnv, command_name: str, limit: float = 1.0) -> torch.Tensor:
    """Speed (m/s) above ``limit``; faster is for getting away from an attacker only."""
    s = _scenario(env, command_name)
    speed = torch.linalg.norm(s.robot_vel(), dim=1)
    return (speed - limit).clamp(min=0.0) * ~s.under_attack


"""
Terminations.
"""


def civic_success_termination(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reached the goal or, with someone who has fallen, raised the alarm next to them."""
    s = _scenario(env, command_name)
    reached = (s.goal_distance() < s.cfg.success_threshold) & ~s.has_fallen
    return reached | s.assisted


def civic_pedestrian_collision(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The robot touches a pedestrian or a dog (its footprint or an arm), walks into a leash, or itself walks into
    someone attacking or blocking it (they come close by themselves; that is not a collision)."""
    s = _scenario(env, command_name)
    hit = (s.pedestrian_clearance(exclude_hostile=True) < 0.0) | (s.limb_pedestrian_clearance(exclude_hostile=True) < 0.0)
    charged = s.pedestrian_clearance(only=s.hostile) < -0.05
    hit = hit | charged | s.leash_crossed
    if hit.any():
        s.record_contacts()
    return hit


"""
Curriculum.
"""


class civic_curriculum(nyc_adaptive_curriculum):
    """:class:`nyc_adaptive_curriculum` where an episode that runs out of time without a fall, collision or step into
    the roadway also counts as a success if the robot got through an attack (it may have had to give up on its goal
    to get away) or did civic work (picked up, put in a basket or set down litter or a bag: detours that the goal's
    time limit does not allow for). Extra parameters: ``command_name`` and ``failure_terms``."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_term: str = "goal_reached",
        ema: float = 0.02,
        unlock_success: float = 0.5,
        uniform_ratio: float = 0.2,
        explore_ratio: float = 0.05,
        floor: float = 0.02,
        command_name: str = "nyc",
        failure_terms: tuple[str, ...] = ("pedestrian_collision", "obstacle_collision", "fall", "roadway"),
    ) -> dict[str, float]:
        return super().__call__(env, env_ids, success_term, ema, unlock_success, uniform_ratio, explore_ratio, floor)

    def _won(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, success_term: str) -> torch.Tensor:
        tm = env.termination_manager
        params = self.cfg.params
        s = _scenario(env, params.get("command_name", "nyc"))
        failed = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        for name in params.get("failure_terms", ("pedestrian_collision", "obstacle_collision", "fall", "roadway")):
            if name in tm.active_terms:
                failed |= tm.get_term(name)[env_ids]
        worked = (s.episode_picked[env_ids] + s.episode_deposited[env_ids] + s.episode_delivered[env_ids]) > 0
        survived = (s.attacked_episode[env_ids] | worked) & tm.get_term("time_out")[env_ids] & ~failed
        return tm.get_term(success_term)[env_ids] | survived


def dogs_visible(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Number of dogs in view (diagnostics)."""
    s = _scenario(env, command_name)
    return ((s.ped_role == DOG) & s.visible_peds()).float().sum(dim=1)
