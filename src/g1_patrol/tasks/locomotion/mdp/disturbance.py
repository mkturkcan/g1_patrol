"""External forces on the robot: someone attacking it, and the weight of what it carries.

During attack windows (a share of the time), the robot is hit repeatedly: shoves on the torso (a horizontal force of
``shove_force`` N for ``shove_duration`` s) and, less often, kicks against a knee, from random directions, every
``interval`` s. A hard shove (250 N for 0.2 s, 50 N s) changes the 35 kg G1's velocity by about 1.4 m/s, a little more
than the strongest velocity kick of the walker's push events. The walker must keep (or regain) its footing; the
navigation policy decides whether to brace and when to get away.

Half the time, one hand carries something (litter, a trash bag) of up to ``hand_load_mass`` kg: its weight pulls the
palm down. The load is redrawn every few seconds (picked up, put down, handed over).

Every write to an articulation's wrench composer replaces all external forces of the written environments, so this
term owns them: it writes the attack forces and the hand loads together at every step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DisturbanceAction(ActionTerm):
    """Applies shoves and kicks during random attack windows, and hand loads. Takes no policy inputs."""

    cfg: DisturbanceActionCfg

    def __init__(self, cfg: DisturbanceActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        torso = self._asset.find_bodies(cfg.torso_body)[0]
        legs = self._asset.find_bodies(cfg.leg_bodies, preserve_order=True)[0]
        palms = self._asset.find_bodies(cfg.palm_bodies, preserve_order=True)[0]
        self._bodies = list(torso) + list(legs) + list(palms)  # 0: torso, 1-2: knees, 3-4: palms
        self.attacked = torch.zeros(n, dtype=torch.bool, device=dev)
        """Inside an attack window."""
        self.hand_load = torch.zeros(n, 2, device=dev)
        """Mass (kg) carried in the left and right hand."""
        self.hits = torch.zeros(n, device=dev)
        self._window = torch.zeros(n, device=dev)
        self._next = torch.zeros(n, device=dev)
        self._left = torch.zeros(n, device=dev)  # remaining duration of the current hit
        self._load_timer = torch.zeros(n, device=dev)
        self._force = torch.zeros(n, len(self._bodies), 3, device=dev)
        self._empty = torch.zeros(n, 0, device=dev)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids if isinstance(env_ids, slice) else slice(None)]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.attacked[env_ids] = False
        self._window[env_ids] = torch.rand(len(env_ids), device=self.device) * 4.0  # time until the first window decision
        self._next[env_ids] = 0.0
        self._left[env_ids] = 0.0
        self._force[env_ids] = 0.0
        self.hits[env_ids] = 0.0
        self._sample_loads(env_ids)

    def _sample_loads(self, env_ids: torch.Tensor):
        cfg, dev, k = self.cfg, self.device, len(env_ids)
        lo, hi = cfg.hand_load_mass
        loaded = torch.rand(k, device=dev) < cfg.hand_load_probability
        mass = torch.zeros(k, 2, device=dev)
        side = torch.randint(0, 2, (k,), device=dev)
        mass[torch.arange(k, device=dev), side] = (lo + (hi - lo) * torch.rand(k, device=dev)) * loaded
        self.hand_load[env_ids] = mass
        i_lo, i_hi = cfg.hand_load_interval
        self._load_timer[env_ids] = i_lo + (i_hi - i_lo) * torch.rand(k, device=dev)

    def apply_actions(self):
        env, cfg, dev = self._env, self.cfg, self.device
        dt = env.physics_dt * (env.cfg.decimation if getattr(env, "_physics_handles_decimation", False) else 1)
        # hand loads
        self._load_timer -= dt
        due = (self._load_timer <= 0.0).nonzero().flatten()
        if len(due):
            self._sample_loads(due)
        self._force[:, 3:5, 2] = -9.81 * self.hand_load
        # attack windows start and end
        self._window -= dt
        flip = self._window <= 0.0
        if flip.any():
            k = int(flip.sum())
            start = torch.rand(k, device=dev) < cfg.attack_ratio
            self.attacked[flip] = start
            lo, hi = cfg.window
            self._window[flip] = torch.where(start, lo + (hi - lo) * torch.rand(k, device=dev), 2.0 + 6.0 * torch.rand(k, device=dev))
            self._next[flip] = 0.3 * torch.rand(k, device=dev)
        # hits
        self._left -= dt
        self._force[self._left <= 0.0, :3] = 0.0
        self._next -= dt * self.attacked
        hit = self.attacked & (self._next <= 0.0)
        if hit.any():
            k = int(hit.sum())
            kick = torch.rand(k, device=dev) < cfg.kick_probability
            direction = 2 * math.pi * torch.rand(k, device=dev)
            f_lo, f_hi = cfg.shove_force
            k_lo, k_hi = cfg.kick_force
            magnitude = torch.where(kick, k_lo + (k_hi - k_lo) * torch.rand(k, device=dev), f_lo + (f_hi - f_lo) * torch.rand(k, device=dev))
            force = torch.zeros(k, 3, 3, device=dev)
            body = torch.where(kick, 1 + torch.randint(0, 2, (k,), device=dev), torch.zeros(k, dtype=torch.long, device=dev))
            force[torch.arange(k, device=dev), body, 0] = magnitude * torch.cos(direction)
            force[torch.arange(k, device=dev), body, 1] = magnitude * torch.sin(direction)
            self._force[hit, :3] = force
            d_lo, d_hi = cfg.shove_duration
            kd_lo, kd_hi = cfg.kick_duration
            self._left[hit] = torch.where(kick, kd_lo + (kd_hi - kd_lo) * torch.rand(k, device=dev), d_lo + (d_hi - d_lo) * torch.rand(k, device=dev))
            i_lo, i_hi = cfg.interval
            self._next[hit] = i_lo + (i_hi - i_lo) * torch.rand(k, device=dev)
            self.hits[hit] += 1.0
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._force, torques=torch.zeros_like(self._force), body_ids=self._bodies, is_global=True
        )


@configclass
class DisturbanceActionCfg(ActionTermCfg):
    class_type: type = DisturbanceAction
    torso_body: str = "torso_link"
    leg_bodies: list[str] = ["left_knee_link", "right_knee_link"]
    palm_bodies: list[str] = ["left_palm_link", "right_palm_link"]
    attack_ratio: float = 0.25
    """Probability that a window (re)drawn is an attack window."""
    window: tuple[float, float] = (2.0, 8.0)
    """Duration (s) of an attack window."""
    interval: tuple[float, float] = (0.8, 3.0)
    """Time (s) between hits during an attack."""
    shove_force: tuple[float, float] = (60.0, 250.0)
    shove_duration: tuple[float, float] = (0.1, 0.2)
    kick_force: tuple[float, float] = (80.0, 250.0)
    kick_duration: tuple[float, float] = (0.08, 0.15)
    kick_probability: float = 0.3
    hand_load_mass: tuple[float, float] = (0.0, 3.0)
    """Mass (kg) carried in one hand ..."""
    hand_load_probability: float = 0.5
    """... with this probability, redrawn every ``hand_load_interval`` s."""
    hand_load_interval: tuple[float, float] = (2.0, 6.0)
