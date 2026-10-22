"""Velocity commands to a robot's built-in locomotion controller, simulated with a randomized tracking model.

Unitree's B2-W is driven through its sport-mode controller, which takes body velocity commands (forward, lateral, yaw
rate) and handles balance, legs and wheels itself. The navigation policy therefore does not need a simulated
controller, only a faithful model of how the controller follows commands. The robot is a kinematic body moved by
:class:`BaseVelocityAction` at every physics step:

* latency: a command takes effect after a delay;
* tracking: per axis a gain error (the robot moves somewhat faster or slower than commanded), a deadband (small
  commands are ignored), a first-order response and bounded acceleration; a yaw drift proportional to the forward
  speed (wheels slip unevenly) and slowly varying velocity noise;
* terrain: the body rides at a fixed height over the ground under its wheels and pitches and rolls with it, so that
  its sensors move as on the robot. Ground more than ``max_step`` lower under one wheel than under another (a
  stairwell, a loading dock) is a fall.

Every episode draws its own parameters from the ``*_range`` intervals, so that the policy works with a controller whose
response is known only approximately. The tracking equations are in :mod:`g1_patrol.controller_model`;
``scripts/b2w/sysid.py`` fits them to logs of the real robot, and the fitted values should lie inside these intervals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from g1_patrol.b2w import BASE_HEIGHT, COMMAND_RANGES, CONTROLLER
from g1_patrol.controller_model import track

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .nyc_scenario import NycScenarioCommand


def _uniform(lo, hi, shape, device) -> torch.Tensor:
    lo, hi = torch.as_tensor(lo, device=device, dtype=torch.float), torch.as_tensor(hi, device=device, dtype=torch.float)
    return lo + (hi - lo) * torch.rand(shape, device=device)


class BaseVelocityAction(ActionTerm):
    """Velocity commands (forward, lateral, yaw rate) to a simulated built-in controller that moves a kinematic robot.

    The policy output is squashed into ``command_ranges`` with a tanh. :attr:`command` is the command of the current
    navigation step, :attr:`velocity_b` the velocity the robot actually moves with (body frame: forward, lateral, yaw
    rate) and :attr:`velocity_w` the same in the world frame. :attr:`hold` and :attr:`drive` let a controller take
    over (as for :class:`WalkerPolicyAction`).
    """

    cfg: BaseVelocityActionCfg

    def __init__(self, cfg: BaseVelocityActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.asset_name]
        self._scenario: NycScenarioCommand | None = None
        n, dev = self.num_envs, self.device
        ranges = torch.tensor(cfg.command_ranges, device=dev)
        self._low, self._high = ranges[:, 0], ranges[:, 1]
        self._mid, self._half = (self._low + self._high) / 2, (self._high - self._low) / 2
        self._raw = torch.zeros(n, 3, device=dev)
        self._command = torch.zeros(n, 3, device=dev)
        self.prev_command = torch.zeros(n, 3, device=dev)
        self.hold = torch.zeros(n, dtype=torch.bool, device=dev)
        self.drive = torch.zeros(n, 3, device=dev)
        self.speed_limit = torch.full((n,), float("inf"), device=dev)
        """Cap (m/s) on the commanded planar speed, e.g. for first trials on the robot."""
        # robot state (world frame)
        self.velocity_b = torch.zeros(n, 3, device=dev)
        self.velocity_w = torch.zeros(n, 3, device=dev)
        self._tracked = torch.zeros(n, 3, device=dev)  # controller's velocity before noise
        self._noise = torch.zeros(n, 3, device=dev)
        self._xy = torch.zeros(n, 2, device=dev)
        self._yaw = torch.zeros(n, device=dev)
        self._z = torch.zeros(n, device=dev)
        self._tilt = torch.zeros(n, 2, device=dev)  # roll, pitch
        self._sync = torch.ones(n, dtype=torch.bool, device=dev)
        self.fallen = torch.zeros(n, dtype=torch.bool, device=dev)
        # latency: command queue at the physics rate
        dt = env.physics_dt
        self._max_delay = int(math.ceil(cfg.delay_range[1] / dt))
        self._queue = torch.zeros(n, self._max_delay + 1, 3, device=dev)
        self._delay = torch.zeros(n, dtype=torch.long, device=dev)
        # controller parameters, drawn per episode
        self.gain = torch.ones(n, 3, device=dev)
        self.deadband = torch.zeros(n, 3, device=dev)
        self.tau = torch.full((n, 3), 0.2, device=dev)
        self.accel = torch.full((n, 3), 2.0, device=dev)
        self.drift = torch.zeros(n, device=dev)
        self.noise_std = torch.zeros(n, 3, device=dev)
        wx, wy = cfg.wheel_base
        self._wheels = torch.tensor([[wx, wy], [wx, -wy], [-wx, -wy], [-wx, wy]], device=dev)  # FL, FR, RR, RL
        self._sample(torch.arange(n, device=dev))

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._command

    @property
    def command(self) -> torch.Tensor:
        """Velocity command of the current navigation step (forward, lateral, yaw rate)."""
        return self._command

    @property
    def scenario(self) -> NycScenarioCommand:
        if self._scenario is None:
            self._scenario = self._env.command_manager.get_term(self.cfg.scenario_name)
        return self._scenario

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        command = self._mid + self._half * torch.tanh(actions)
        speed = torch.linalg.norm(command[:, :2], dim=1, keepdim=True)
        command[:, :2] *= (self.speed_limit[:, None] / speed.clamp(min=1e-6)).clamp(max=1.0)
        self.prev_command[:] = self._command
        self._command[:] = torch.where(self.hold[:, None], self.drive, command)

    def apply_actions(self):
        cfg, dt = self.cfg, self._env.physics_dt
        if self._sync.any():  # (re)spawned: start from the pose the scenario wrote
            ids = self._sync.nonzero(as_tuple=True)[0]
            q = self._asset.data.root_quat_w.torch[ids]
            self._xy[ids] = self._asset.data.root_pos_w.torch[ids, :2]
            self._yaw[ids] = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            self._z[ids] = self._ground(ids).mean(dim=1)
            self._tilt[ids] = self._ground_tilt(self._ground(ids))
            self._sync[ids] = False

        # latency
        self._queue = torch.roll(self._queue, 1, dims=1)
        self._queue[:, 0] = self._command
        command = self._queue[torch.arange(self.num_envs, device=self.device), self._delay]
        # tracking: deadband, gain, yaw drift, first-order response with bounded acceleration (shared with sysid), noise
        self._tracked = track(self._tracked, command, dt, self.gain, self.deadband, self.tau, self.accel, self.drift)
        decay = math.exp(-dt / cfg.noise_time)
        self._noise = self._noise * decay + self.noise_std * math.sqrt(1.0 - decay**2) * torch.randn_like(self._noise)
        moving = (self._tracked.abs() > 1e-3).any(dim=1, keepdim=True)  # a robot at rest does not jitter
        self.velocity_b[:] = self._tracked + self._noise * moving

        # integrate the planar pose
        self._yaw += self.velocity_b[:, 2] * dt
        c, s = torch.cos(self._yaw), torch.sin(self._yaw)
        vx, vy = self.velocity_b[:, 0], self.velocity_b[:, 1]
        self.velocity_w[:, 0], self.velocity_w[:, 1], self.velocity_w[:, 2] = c * vx - s * vy, s * vx + c * vy, self.velocity_b[:, 2]
        self._xy += self.velocity_w[:, :2] * dt

        # ride over the ground
        ground = self._ground()
        self.fallen |= (ground.amax(dim=1) - ground.amin(dim=1)) > cfg.max_step
        alpha = 1.0 - math.exp(-dt / cfg.terrain_time)
        self._z += (ground.mean(dim=1) - self._z) * alpha
        self._tilt += (self._ground_tilt(ground) - self._tilt) * alpha

        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :2] = self._xy
        pose[:, 2] = self._z + cfg.base_height
        pose[:, 3:] = quat_from_euler_xyz(self._tilt[:, 0], self._tilt[:, 1], self._yaw)
        self._asset.write_root_pose_to_sim_index(root_pose=pose)
        twist = torch.zeros(self.num_envs, 6, device=self.device)
        twist[:, :2], twist[:, 5] = self.velocity_w[:, :2], self.velocity_w[:, 2]
        self._asset.write_root_velocity_to_sim_index(root_velocity=twist)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._raw[ids] = 0.0
        self._command[ids] = 0.0
        self.prev_command[ids] = 0.0
        self.hold[ids] = False
        self.drive[ids] = 0.0
        self.velocity_b[ids] = 0.0
        self.velocity_w[ids] = 0.0
        self._tracked[ids] = 0.0
        self._noise[ids] = 0.0
        self._queue[ids] = 0.0
        self.fallen[ids] = False
        self._sync[ids] = True
        self._sample(ids)

    """
    Helpers.
    """

    def _sample(self, ids: torch.Tensor):
        cfg, m, dev = self.cfg, len(ids), self.device
        dt = self._env.physics_dt
        delay = _uniform(*cfg.delay_range, (m,), dev)
        self._delay[ids] = (delay / dt).round().long().clamp(0, self._max_delay)
        self.gain[ids] = _uniform([g[0] for g in cfg.gain_range], [g[1] for g in cfg.gain_range], (m, 3), dev)
        self.deadband[ids] = _uniform([0.0] * 3, cfg.deadband_max, (m, 3), dev)
        self.tau[ids] = _uniform([t[0] for t in cfg.tau_range], [t[1] for t in cfg.tau_range], (m, 3), dev)
        self.accel[ids] = _uniform([a[0] for a in cfg.accel_range], [a[1] for a in cfg.accel_range], (m, 3), dev)
        self.drift[ids] = _uniform(-cfg.drift_max, cfg.drift_max, (m,), dev)
        self.noise_std[ids] = _uniform([0.0] * 3, cfg.noise_max, (m, 3), dev)

    def _ground(self, ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
        """Ground height (m, 4) under the wheels."""
        s = self.scenario
        c, sn = torch.cos(self._yaw[ids]), torch.sin(self._yaw[ids])
        wx, wy = self._wheels[:, 0], self._wheels[:, 1]
        local = self._xy[ids][:, None] - s.tile_center[ids][:, None]
        pts = local + torch.stack([c[:, None] * wx - sn[:, None] * wy, sn[:, None] * wx + c[:, None] * wy], dim=-1)
        return s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], pts)

    def _ground_tilt(self, ground: torch.Tensor) -> torch.Tensor:
        """Roll and pitch (m, 2) of a body resting on wheels at these ground heights (FL, FR, RR, RL)."""
        wx, wy = self.cfg.wheel_base
        front, rear = (ground[:, 0] + ground[:, 1]) / 2, (ground[:, 2] + ground[:, 3]) / 2
        left, right = (ground[:, 0] + ground[:, 3]) / 2, (ground[:, 1] + ground[:, 2]) / 2
        # positive pitch lowers the nose, positive roll raises the left side
        return torch.stack([torch.atan((left - right) / (2 * wy)), torch.atan((rear - front) / (2 * wx))], dim=1)


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    """Configuration of :class:`BaseVelocityAction`. The defaults describe Unitree's B2-W in sport mode before system
    identification (``scripts/b2w/sysid.py``)."""

    class_type: type = BaseVelocityAction
    scenario_name: str = MISSING
    """Scenario command, whose tile layout gives the ground height under the wheels."""
    command_ranges: tuple[tuple[float, float], ...] = COMMAND_RANGES
    """(min, max) of the forward, lateral and yaw-rate commands (m/s, m/s, rad/s)."""
    base_height: float = BASE_HEIGHT
    """Height (m) of the root (base_link) above the ground."""
    wheel_base: tuple[float, float] = (0.33, 0.24)
    """Half distances (m) between the wheels' contact points along and across the body."""
    max_step: float = 0.3
    """Height difference (m) of the ground under two wheels beyond which the robot falls."""
    terrain_time: float = 0.08
    """Time constant (s) with which the body follows the ground (height, pitch, roll)."""
    delay_range: tuple[float, float] = CONTROLLER["delay_range"]
    """Command latency (s): network, the controller's own filtering and the motors."""
    gain_range: tuple[tuple[float, float], ...] = CONTROLLER["gain_range"]
    """Ratio of the steady-state velocity to the commanded one, per axis."""
    deadband_max: tuple[float, float, float] = CONTROLLER["deadband_max"]
    """Commands smaller than a deadband drawn up to these values are ignored, per axis."""
    tau_range: tuple[tuple[float, float], ...] = CONTROLLER["tau_range"]
    """Time constant (s) of the first-order response, per axis."""
    accel_range: tuple[tuple[float, float], ...] = CONTROLLER["accel_range"]
    """Acceleration bound (m/s^2, m/s^2, rad/s^2), per axis."""
    drift_max: float = CONTROLLER["drift_max"]
    """Yaw drift (rad/s per m/s of forward speed), drawn within +-this."""
    noise_max: tuple[float, float, float] = CONTROLLER["noise_max"]
    """Velocity noise (m/s, m/s, rad/s; standard deviation) drawn up to these values, per axis."""
    noise_time: float = CONTROLLER["noise_time"]
    """Correlation time (s) of the velocity noise."""


"""
Observations and terminations.
"""


def base_velocity(env: ManagerBasedRLEnv, action_name: str = "base") -> torch.Tensor:
    """Velocity of the robot in its body frame (forward, lateral, yaw rate), as the controller's odometry reports it."""
    return env.action_manager.get_term(action_name).velocity_b


def base_command(env: ManagerBasedRLEnv, action_name: str = "base") -> torch.Tensor:
    """Velocity command of the current step."""
    return env.action_manager.get_term(action_name).command


def base_fall(env: ManagerBasedRLEnv, action_name: str = "base") -> torch.Tensor:
    """The robot drove over a drop (see :class:`BaseVelocityActionCfg.max_step`)."""
    return env.action_manager.get_term(action_name).fallen
