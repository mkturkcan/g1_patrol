"""Action term that drives a frozen walking policy with velocity commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationManager
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyAction
from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action_cfg import PreTrainedPolicyActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WalkerPolicyAction(PreTrainedPolicyAction):
    """:class:`PreTrainedPolicyAction` that maps the navigation policy's output to a smooth velocity command inside
    the range the walker was trained on, and feeds the walker the observations it was trained with, including sensor
    noise, latencies and history, so that the navigator learns to drive the walker as it behaves on the robot.

    The policy output is squashed into ``command_ranges`` (``tanh``) or clamped to them, then low-pass filtered: each
    navigation step moves the command a fraction ``smoothing`` of the way to the requested value. The command starts
    at zero after a reset.

    With ``posture_command``, the walker also receives a posture (pelvis height, forward lean; set with
    :meth:`set_posture` by a controller that picks things up), and :attr:`hold` keeps its velocity command at zero.
    """

    cfg: WalkerPolicyActionCfg

    def __init__(self, cfg: WalkerPolicyActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # the walker is frozen: never track gradients through it, even outside torch.inference_mode()
        self.policy = torch.no_grad()(self.policy)
        ranges = torch.tensor(cfg.command_ranges, device=self.device)
        self._low, self._high = ranges[:, 0], ranges[:, 1]
        self._mid, self._half = (self._low + self._high) / 2, (self._high - self._low) / 2
        self.target = torch.zeros_like(self._raw_actions)
        """Command requested by the navigation policy at the current step."""
        self.prev_command = torch.zeros_like(self._raw_actions)
        """Command applied at the previous navigation step."""
        n = self.num_envs
        self.posture = torch.tensor([cfg.nominal_height, 0.0], device=self.device).repeat(n, 1)
        """Posture command (pelvis height above the ground, forward lean) of walkers that take one."""
        self.hold = torch.zeros(n, dtype=torch.bool, device=self.device)
        """Environments whose walker must stand still (velocity command zero), e.g. while picking something up."""
        if cfg.posture_command:
            # the walker's command input is the velocity command followed by the posture
            cfg.low_level_observations.velocity_commands.func = lambda dummy_env: torch.cat([self._raw_actions, self.posture], 1)
            self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

    @property
    def command(self) -> torch.Tensor:
        """Velocity command applied to the walker (forward, lateral, yaw rate)."""
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.squash == "tanh":
            target = self._mid + self._half * torch.tanh(actions)
        else:
            target = torch.clamp(actions, self._low, self._high)
        self.target[:] = target
        self.prev_command[:] = self._raw_actions
        self._raw_actions[:] = self._raw_actions + self.cfg.smoothing * (target - self._raw_actions)
        self._raw_actions[self.hold] = 0.0

    def set_posture(self, height: torch.Tensor, lean: torch.Tensor, dt: float):
        """Move the posture command towards (height, lean) at the rates the walker was trained with; ``dt`` is the
        time (s) since the previous call."""
        rate = torch.tensor([self.cfg.height_rate, self.cfg.lean_rate], device=self.device) * dt
        target = torch.stack([height, lean], dim=1)
        self.posture += (target - self.posture).clamp(-rate, rate)

    def apply_actions(self):
        # as PreTrainedPolicyAction, but recording the walker's observation history and latency buffers at its rate
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy", update_history=True)
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self.prev_command[ids] = 0.0
        self.target[ids] = 0.0
        self.posture[ids, 0] = self.cfg.nominal_height
        self.posture[ids, 1] = 0.0
        self.hold[ids] = False
        self._low_level_obs_manager.reset(env_ids)  # the walker's observation history and latencies


@configclass
class WalkerPolicyActionCfg(PreTrainedPolicyActionCfg):
    class_type: type = WalkerPolicyAction
    command_ranges: tuple[tuple[float, float], ...] = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))
    """(min, max) of the forward, lateral and yaw-rate commands (m/s, m/s, rad/s)."""
    squash: str = "clamp"
    """How policy outputs map to commands: ``"tanh"`` (smooth squashing into the ranges) or ``"clamp"``."""
    smoothing: float = 1.0
    """Fraction of the change to the requested command applied per navigation step (1: no filtering)."""
    posture_command: bool = False
    """The walker takes a posture command (pelvis height, forward lean) after the velocity command."""
    nominal_height: float = 0.72
    height_rate: float = 0.4
    lean_rate: float = 1.2
    """Rates (m/s, rad/s) at which the posture command moves towards its target, as in the walker's training."""
