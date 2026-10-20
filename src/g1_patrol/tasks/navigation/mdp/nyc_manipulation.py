"""The robot's hands in the civic navigation tasks: picking up litter and trash bags, putting them where they belong,
bracing against an attacker, and calling for help.

The navigation policy has two extra action inputs: a trigger and a brace (each positive: on).

* Trigger, empty-handed, with litter or a bag in the reach zone in front of the robot: the robot picks it up. It stops,
  squats and leans forward (posture command to the walker), reaches for the item with the arm on its side (damped
  least-squares inverse kinematics on the five arm joints), closes the hand, lifts the item and stands up. Litter is
  carried in front of the hip, a bag by its knot with the arm down at the side.
* Trigger, holding litter, with a litter basket within reach in front: the robot raises the hand over the basket's
  opening and lets go. Holding a bag that hangs over the zone beside a curbside bag pile: it bends its knees to lower
  the bag and lets go.
* Trigger next to a person who has fallen (facing them, at 0.5-1.8 m): the robot raises an arm and calls for help.
* Brace: the robot lowers its pelvis a little and raises its forearms in front of its head (a guard), and can walk in
  that stance, e.g. away from an attacker. Bracing interrupts a pick that has not grasped yet.

The walker keeps the robot balanced throughout; the navigation policy's velocity command is held at zero while picking,
putting down or calling. Grasping is simplified: when the hand closes with its grasp center (``grasp_offset`` from the palm
towards the fingertips, where the Dex3's fingers close around an object) within ``grasp_tolerance`` of the item's grasp
point (the middle of a piece of litter, a bag's knot), the item is held rigidly (it follows the palm); the fingers
close for the look of it. Reaching is not simplified: the walker has to bring the hand there, with the waist and the
arm (inverse kinematics), from a deep squat with a strong forward lean for litter on the ground (the G1's arm reaches
0.41 m from the shoulder). The held item's weight pulls on the palm.

This term also owns the robot's external forces (shoves and kicks from :class:`SocialMixin`, the held item's weight):
every write to the wrench composer replaces all of an environment's forces, so they are written together.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul

from g1_patrol.tasks.locomotion.mdp.upper_body import UPPER_BODY_JOINTS

from .nyc_trash import FALLING, HELD, ON_GROUND

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

IDLE, CROUCH, REACH, GRASP, LIFT, STAND, RAISE, RELEASE, LOWER, ALERT = range(10)
PICKING = (CROUCH, REACH, GRASP, LIFT, STAND)
DROPPING = (RAISE, RELEASE, LOWER)
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")
MIRROR = (1.0, -1.0, -1.0, 1.0, -1.0)
"""Sign of each arm joint of the left arm relative to the right arm."""
# right-arm joint angles (the left arm mirrors them), all within the ranges the walker was trained with. Shoulder pitch:
# negative raises the arm forward; elbow pitch: 0 bends the forearm forward at a right angle, about 1.57 straightens it
CARRY_RIGHT = (-0.1, -0.25, 0.0, 0.3, 0.0)
"""Holding litter in front of the hip (forearm forward)."""
HANG_RIGHT = (0.0, -0.42, 0.0, 0.2, 0.0)
"""Carrying a bag by its knot beside the hip, forearm forward and a little out, so that the bag clears the ground."""
GUARD_RIGHT = (-1.2, 0.1, 0.0, 1.6, 0.0)
"""Arm raised forward, open hand: with both arms, a guard that also says "stop"."""
ALERT_RIGHT = (-1.7, -0.2, 0.0, 0.3, 0.0)
"""Arm raised high, forearm up: calling for help."""
IK_RANGE_RIGHT = ((-2.0, 1.2), (-1.5, 0.3), (-1.8, 1.5), (-0.2, 2.5), (-1.7, 1.7))
"""Arm joint ranges the walker was trained with (its reach and random modes); inverse kinematics stays inside."""
CLOSED_RIGHT = {"zero": 0.0, "one": -0.3, "two": -1.0, "three": 1.0, "four": 1.0, "five": 1.0, "six": 1.0}
"""Right-hand finger angles of a grasp (the left hand mirrors all but the thumb rotation)."""


class HandsAction(ActionTerm):
    """Picks up and puts down litter and bags, braces and calls for help on the navigation policy's two inputs."""

    cfg: HandsActionCfg

    def __init__(self, cfg: HandsActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        robot = self._asset
        self._ids, names = robot.find_joints(UPPER_BODY_JOINTS, preserve_order=True)
        index = {name: k for k, name in enumerate(names)}
        self._default = robot.data.default_joint_pos.torch[:, self._ids].clone()
        lim = robot.data.soft_joint_pos_limits.torch[0, self._ids]
        self._lo, self._hi = lim[:, 0], lim[:, 1]
        self.targets = self._default.clone()
        joint_index = {name: k for k, name in enumerate(robot.joint_names)}
        mirror = torch.tensor(MIRROR, device=dev)
        self._arm_ub, self._arm_jac, self._palm, self._fingers, self._pose, self._ik_lo, self._ik_hi = {}, {}, {}, {}, {}, {}, {}
        for side in ("left", "right"):
            joints = [f"{side}_{j}_joint" for j in ARM]
            self._arm_ub[side] = torch.tensor([index[j] for j in joints], device=dev)
            self._arm_jac[side] = torch.tensor([6 + joint_index[j] for j in joints], device=dev)  # floating base first
            self._palm[side] = robot.find_bodies(f"{side}_palm_link")[0][0]
            fingers = [name for name in names if name.startswith(side) and name.split("_")[1] in CLOSED_RIGHT]
            self._fingers[side] = torch.tensor([index[f] for f in fingers], device=dev)
            sign = mirror if side == "left" else torch.ones(5, device=dev)
            self._pose[side] = {
                name: torch.tensor(pose, device=dev) * sign
                for name, pose in (("carry", CARRY_RIGHT), ("hang", HANG_RIGHT), ("guard", GUARD_RIGHT), ("alert", ALERT_RIGHT))
            }
            closed = [CLOSED_RIGHT[f.split("_")[1]] * (-1.0 if side == "left" and f.split("_")[1] != "zero" else 1.0) for f in fingers]
            self._pose[side]["closed"] = torch.tensor(closed, device=dev)
            r = torch.tensor(IK_RANGE_RIGHT, device=dev)
            bounds = torch.stack([r[:, 0] * sign, r[:, 1] * sign], dim=1).sort(dim=1).values
            self._ik_lo[side], self._ik_hi[side] = bounds[:, 0], bounds[:, 1]
        self._waist_ub = index["torso_joint"]
        waist_jac = 6 + joint_index["torso_joint"]
        self._tips, self._ik_jac = {}, {}
        for side in ("left", "right"):
            self._tips[side] = robot.find_bodies([f"{side}_four_link", f"{side}_six_link"], preserve_order=True)[0]
            self._ik_jac[side] = torch.cat([torch.tensor([waist_jac], device=dev), self._arm_jac[side]])
            self._ik_lo[side] = torch.cat([torch.tensor([-cfg.waist_range], device=dev), self._ik_lo[side]])
            self._ik_hi[side] = torch.cat([torch.tensor([cfg.waist_range], device=dev), self._ik_hi[side]])
        self._wrench_bodies = [
            robot.find_bodies(cfg.torso_body)[0][0],
            *robot.find_bodies(cfg.knee_bodies, preserve_order=True)[0],
            self._palm["left"],
            self._palm["right"],
        ]
        self.phase = torch.full((n,), IDLE, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, device=dev)
        self.side = torch.zeros(n, dtype=torch.long, device=dev)  # working / holding arm: 0 left, 1 right
        self.item = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.target = torch.full((n,), -1, dtype=torch.long, device=dev)  # basket being used
        self.drop_bag = torch.zeros(n, dtype=torch.bool, device=dev)  # putting down a bag (else litter)
        self.bracing = torch.zeros(n, dtype=torch.bool, device=dev)
        self._closest = torch.full((n,), 1e3, device=dev)  # closest approach of the grasp center while the hand closes
        self.hold_pos = torch.zeros(n, 3, device=dev)  # held litter relative to the palm (palm frame)
        self.hold_quat = torch.zeros(n, 4, device=dev)
        self.hold_quat[:, 3] = 1.0
        self._raw = torch.zeros(n, 2, device=dev)
        # per-step events for the rewards
        self.picked = torch.zeros(n, device=dev)
        self.failed = torch.zeros(n, device=dev)
        self.invalid = torch.zeros(n, device=dev)
        self.dropped = torch.zeros(n, device=dev)
        self.alerted = torch.zeros(n, device=dev)
        self._scenario = None
        self._walker = None
        self._count = 0

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def busy(self) -> torch.Tensor:
        return self.phase != IDLE

    def holding(self) -> torch.Tensor:
        return self._links()[0].held_item >= 0

    def holding_bag(self) -> torch.Tensor:
        s = self._links()[0]
        held = s.held_item
        return (held >= 0) & s.is_bag[held.clamp(min=0)] if s.num_trash else torch.zeros_like(held, dtype=torch.bool)

    def _links(self):
        if self._scenario is None:
            self._scenario = self._env.command_manager.get_term(self.cfg.scenario_name)
            self._walker = self._env.action_manager.get_term(self.cfg.walker_name)
        return self._scenario, self._walker

    def _dt(self) -> float:
        env = self._env
        return env.physics_dt * (env.cfg.decimation if getattr(env, "_physics_handles_decimation", False) else 1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = IDLE
        self.timer[ids] = 0.0
        self.item[ids] = -1
        self.target[ids] = -1
        self.bracing[ids] = False
        self.targets[ids] = self._default[ids]

    # -- inputs (navigation rate) ---------------------------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        s, walker = self._links()
        cfg = self.cfg
        for event in (self.picked, self.failed, self.invalid, self.dropped, self.alerted):
            event.zero_()
        self.bracing[:] = actions[:, 1] > 0.0
        # bracing interrupts a pick that has not grasped anything yet
        abort = self.bracing & ((self.phase == CROUCH) | (self.phase == REACH) | (self.phase == GRASP))
        self.phase[abort] = STAND
        self.timer[abort] = 0.0
        trigger = (actions[:, 0] > 0.0) & (self.phase == IDLE)
        if not trigger.any():
            return
        holding = s.held_item >= 0
        bag = self.holding_bag()
        alert = trigger & s.can_alert() if s.social else torch.zeros_like(trigger)
        basket = s.reachable_basket(cfg.basket_distance, cfg.basket_angle)
        item = s.reachable_item(cfg.reach_ahead, cfg.reach_side)
        drop_litter = trigger & ~alert & holding & ~bag & (basket >= 0)
        drop_bag = trigger & ~alert & holding & bag & s.held_in_zone()
        pick = trigger & ~alert & ~holding & (item >= 0) & ~self.bracing
        self.invalid += (trigger & ~alert & ~drop_litter & ~drop_bag & ~pick).float()
        if pick.any():
            self.item[pick] = item[pick]
            # the arm on the item's side
            p = s.trash_positions()[pick, item[pick], :2] - s.robot_pos()[pick]
            yaw = s.robot_yaw()[pick]
            lateral = -torch.sin(yaw) * p[:, 0] + torch.cos(yaw) * p[:, 1]
            self.side[pick] = (lateral < 0.0).long()
            self.phase[pick] = CROUCH
        drop = drop_litter | drop_bag
        self.target[drop_litter] = basket[drop_litter]
        self.drop_bag[drop] = drop_bag[drop]
        self.phase[drop] = RAISE
        self.phase[alert] = ALERT
        self.timer[pick | drop | alert] = 0.0

    # -- motion (physics rate) -----------------------------------------------------------------------------------------

    def apply_actions(self):
        # every ``update_every`` physics steps (the joint targets and forces stay set in between)
        self._count += 1
        if self._count < self.cfg.update_every:
            return
        cfg = self.cfg
        s, walker = self._links()
        dt = self._dt() * self._count
        self._count = 0
        n, dev = self.num_envs, self.device
        self.timer += dt * self.busy
        walker.hold[:] = self.busy
        holding = s.held_item >= 0
        bag = self.holding_bag()
        working_bag = s.is_bag[self.item.clamp(min=0)] if s.num_trash else torch.zeros(n, dtype=torch.bool, device=dev)
        # posture: squat and lean to pick up (less for a bag), a braced stance, else the walking posture
        squat = (self.phase == CROUCH) | (self.phase == REACH) | (self.phase == GRASP) | (self.phase == LIFT)
        height = torch.full((n,), walker.cfg.nominal_height, device=dev)
        lean = torch.zeros(n, device=dev)
        brace = self.bracing & ~squat
        height = torch.where(brace, cfg.brace_posture[0], height)
        lean = torch.where(brace, cfg.brace_posture[1], lean)
        height = torch.where(squat, torch.where(working_bag, cfg.bag_posture[0], cfg.pick_posture[0]), height)
        lean = torch.where(squat, torch.where(working_bag, cfg.bag_posture[1], cfg.pick_posture[1]), lean)
        putting_down = (self.phase == RAISE) | (self.phase == RELEASE)
        lowering = putting_down & self.drop_bag
        height = torch.where(lowering, cfg.bag_drop_posture[0], height)
        lean = torch.where(lowering, cfg.bag_drop_posture[1], lean)
        walker.set_posture(height, lean, dt)
        # arm targets
        targets = self._default.clone()
        ik_phase = (self.phase == CROUCH) | (self.phase == REACH) | (self.phase == GRASP) | (putting_down & ~self.drop_bag)
        goal = self._ik_goal(s) if ik_phase.any() else None
        for k, side in enumerate(("left", "right")):
            mine = self.side == k
            pose = self._pose[side]
            arm, fingers = self._arm_ub[side], self._fingers[side]

            def put(mask, cols, value):
                if mask.any():
                    rows = targets[mask]
                    rows[:, cols] = value
                    targets[mask] = rows

            carrying = holding & mine & ~ik_phase
            put(carrying & ~bag, arm, pose["carry"])
            put(carrying & bag, arm, pose["hang"])
            put((holding & mine) | ((self.phase == GRASP) & mine), fingers, pose["closed"])
            # a free arm guards while bracing; the right arm signals while calling for help
            put(brace & ~(holding & mine) & ~(ik_phase & mine), arm, pose["guard"])
            if k == 1:
                put(self.phase == ALERT, arm, pose["alert"])
            ik = mine & ik_phase
            if ik.any():
                cols = torch.cat([torch.tensor([self._waist_ub], device=dev), arm])
                q = self._ik_step(side, goal, self.targets[:, cols])
                put(ik, cols, q[ik])
        # move the targets towards their goals at a limited joint speed
        step = cfg.joint_speed * dt
        self.targets += (targets - self.targets).clamp(-step, step)
        self.targets = torch.maximum(torch.minimum(self.targets, self._hi), self._lo)
        self._asset.set_joint_position_target_index(target=self.targets, joint_ids=self._ids)
        self._advance(s, dt)
        self._carry_item(s)
        self._write_forces(s)

    def _hand(self, side_index: torch.Tensor):
        """Palm position and orientation, and the grasp center ``grasp_offset`` from the palm towards the fingertips
        (world frame) of each environment's working hand."""
        robot = self._asset
        envs = torch.arange(self.num_envs, device=self.device)
        palms = torch.tensor([self._palm["left"], self._palm["right"]], device=self.device)[side_index]
        tips = torch.tensor([self._tips["left"], self._tips["right"]], device=self.device)[side_index]  # (n, 2)
        pos_w = robot.data.body_link_pos_w.torch
        palm_pos, palm_quat = pos_w[envs, palms], robot.data.body_link_quat_w.torch[envs, palms]
        tip = pos_w[envs[:, None], tips].mean(dim=1)
        u = torch.nn.functional.normalize(tip - palm_pos, dim=1)
        return palm_pos, palm_quat, palm_pos + self.cfg.grasp_offset * u

    def _ik_goal(self, s) -> torch.Tensor:
        """World position for the hand's grasp center, from where it (or the held item) should be: above the item's
        grasp point (approach, then grasp), or over a basket's opening."""
        n, dev, cfg = self.num_envs, self.device, self.cfg
        envs = torch.arange(n, device=dev)
        palm_pos, _, grasp = self._hand(self.side)
        target = grasp.clone()  # where the grasp center should go
        if s.num_trash:
            item = self.item.clamp(min=0)
            p = s.trash.data.body_link_pos_w.torch[envs, item]
            above = torch.where(self.phase == CROUCH, cfg.approach_height, 0.0) + s.trash_grasp_height[item]
            target = p + torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), above], dim=1)
        dropping = (self.phase == RAISE) | (self.phase == RELEASE)
        if dropping.any() and s.num_trash:
            held = s.held_item.clamp(min=0)
            item_pos = s.trash.data.body_link_pos_w.torch[envs, held]
            # the item over the near part of the basket's opening, above the rim
            b = s.baskets[envs, self.target.clamp(min=0)]
            to_robot = torch.nn.functional.normalize(s.robot_pos() - b[:, :2] + 1e-4, dim=1)
            over = b[:, :2] + (b[:, 2:3] - cfg.drop_inset) * to_robot
            item_goal = torch.cat([over + s.tile_center, (b[:, 3] + cfg.drop_height)[:, None]], dim=1)
            # the held item moves with the hand: shift the grasp center by the item's offset from its goal
            target = torch.where(dropping[:, None], grasp + (item_goal - item_pos), target)
        return target

    def _ik_step(self, side: str, goal: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """One damped least-squares step of the waist and arm joints (``q``: their current targets) towards putting the
        hand's grasp center at ``goal`` (position only; the grasp center moves with the palm, so the step also turns
        the hand: fingers pointing down reach lower)."""
        robot = self._asset
        palm = self._palm[side]
        jac6 = robot.data.body_link_jacobian_w.torch[:, palm][:, :, self._ik_jac[side]]  # (n, 6, 6)
        side_index = torch.full((self.num_envs,), 0 if side == "left" else 1, dtype=torch.long, device=self.device)
        palm_pos, _, grasp = self._hand(side_index)
        r = grasp - palm_pos
        skew = torch.zeros(self.num_envs, 3, 3, device=self.device)
        skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -r[:, 2], r[:, 1], -r[:, 0]
        skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = r[:, 2], -r[:, 1], r[:, 0]
        jac = jac6[:, 0:3] - skew @ jac6[:, 3:6]  # velocity of the grasp center: v + w x r
        err = goal - grasp
        lam = self.cfg.ik_damping
        jjt = jac @ jac.transpose(1, 2) + lam**2 * torch.eye(3, device=self.device)
        dq = self.cfg.ik_gain * (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err[..., None])).squeeze(-1)
        step = self.cfg.ik_speed * self._dt() * self.cfg.update_every
        dq = dq.clamp(-step, step)
        return torch.maximum(torch.minimum(q + dq, self._ik_hi[side]), self._ik_lo[side])

    def _advance(self, s, dt: float):
        cfg = self.cfg
        t, phase = self.timer, self.phase
        n, dev = self.num_envs, self.device
        envs = torch.arange(n, device=dev)
        nxt = phase.clone()
        nxt[(phase == CROUCH) & (t >= cfg.crouch_time)] = REACH
        nxt[(phase == REACH) & (t >= cfg.reach_time)] = GRASP
        # the closing fingers catch the item if the grasp center comes close enough while they close
        closing = phase == GRASP
        if closing.any() and s.num_trash:
            _, _, grasp = self._hand(self.side)
            item = self.item.clamp(min=0)
            grasp_point = s.trash.data.body_link_pos_w.torch[envs, item].clone()
            grasp_point[:, 2] += s.trash_grasp_height[item]
            self._closest = torch.where(closing, torch.minimum(self._closest, torch.linalg.norm(grasp_point - grasp, dim=1)), self._closest)
        grasp_done = (phase == GRASP) & (t >= cfg.grasp_time)
        if grasp_done.any():
            palm_pos, palm_quat, grasp = self._hand(self.side)
            item = self.item.clamp(min=0)
            ok = torch.zeros_like(grasp_done)
            if s.num_trash:
                p = s.trash.data.body_link_pos_w.torch[envs, item]
                q = s.trash.data.body_link_quat_w.torch[envs, item]
                close = self._closest < cfg.grasp_tolerance
                ok = grasp_done & close & (s.trash_state[envs, item] == ON_GROUND) & (self.item >= 0)
                if ok.any():
                    self.hold_pos[ok] = quat_apply_inverse(palm_quat[ok], p[ok] - palm_pos[ok])
                    self.hold_quat[ok] = quat_mul(quat_inv(palm_quat[ok]), q[ok])
                    s.trash_state[ok, item[ok]] = HELD
                    s.held_item[ok] = item[ok]
                    s.trash_picked[ok] += 1.0
                    s.episode_picked[ok] += 1.0
                    self.picked += ok.float()
            failed = grasp_done & ~ok
            self.failed += failed.float()
            nxt[ok] = LIFT
            nxt[failed] = STAND
        nxt[(phase == LIFT) & (t >= cfg.lift_time)] = STAND
        nxt[(phase == STAND) & (t >= cfg.stand_time)] = IDLE
        nxt[(phase == RAISE) & (t >= cfg.raise_time)] = RELEASE
        release = (phase == RAISE) & (nxt == RELEASE)
        if release.any():  # let go of the item at the start of RELEASE
            rows = release.nonzero().flatten()
            held = s.held_item[rows]
            ok = held >= 0
            rows, held = rows[ok], held[ok]
            s.trash_state[rows, held] = FALLING
            s.trash_release_time[rows, held] = 0.0
            s.held_item[rows] = -1
            self.dropped[rows] += 1.0
        nxt[(phase == RELEASE) & (t >= cfg.release_time)] = LOWER
        nxt[(phase == LOWER) & (t >= cfg.lower_time)] = IDLE
        alert_done = (phase == ALERT) & (t >= cfg.alert_time)
        if alert_done.any():
            s.assisted |= alert_done & s.has_fallen
            self.alerted += alert_done.float()
            nxt[alert_done] = IDLE
        changed = nxt != phase
        self.timer[changed] = 0.0
        self._closest[changed & (nxt == GRASP)] = 1e3
        self.phase[:] = nxt
        self.item[(nxt == IDLE) | ((nxt == STAND) & ~(s.held_item >= 0))] = -1

    def _carry_item(self, s):
        """Held items follow the palm: litter rigidly, a bag hanging upright below the hand (written for each item
        slot that some robot is holding)."""
        if s.num_trash == 0:
            return
        held = s.held_item
        holding = held >= 0
        if not holding.any():
            return
        palm_pos, palm_quat, grasp = self._hand(self.side)
        item = held.clamp(min=0)
        bag = s.is_bag[item]
        pos = palm_pos + quat_apply(palm_quat, self.hold_pos)
        quat = quat_mul(palm_quat, self.hold_quat)
        hang = grasp.clone()  # the knot in the hand, the bag below it
        hang[:, 2] -= s.trash_grasp_height[item]
        zeros = torch.zeros(self.num_envs, device=self.device)
        upright = quat_from_euler_xyz(zeros, zeros, s.robot_yaw())
        pos = torch.where(bag[:, None], hang, pos)
        quat = torch.where(bag[:, None], upright, quat)
        for k in held[holding].unique().tolist():
            envs = (held == k).nonzero().flatten()
            pose = torch.cat([pos[envs], quat[envs]], dim=1)[:, None]
            s.trash.write_body_pose_to_sim_index(body_poses=pose, body_ids=[k], env_ids=envs)
            s.trash.write_body_velocity_to_sim_index(body_velocities=torch.zeros(len(envs), 1, 6, device=self.device),
                                                     body_ids=[k], env_ids=envs)

    def _write_forces(self, s):
        n, dev = self.num_envs, self.device
        force = torch.zeros(n, len(self._wrench_bodies), 3, device=dev)
        if s.social:
            force[:, :3] = s.attack_force
        if s.num_trash:
            held = s.held_item
            weight = torch.where(held >= 0, s.trash_mass[held.clamp(min=0)], torch.zeros(n, device=dev)) * 9.81
            force[torch.arange(n, device=dev), 3 + self.side, 2] = -weight
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force, torques=torch.zeros_like(force), body_ids=self._wrench_bodies, is_global=True
        )



@configclass
class HandsActionCfg(ActionTermCfg):
    class_type: type = HandsAction
    scenario_name: str = MISSING
    walker_name: str = MISSING
    torso_body: str = "torso_link"
    knee_bodies: list[str] = ["left_knee_link", "right_knee_link"]
    reach_ahead: tuple[float, float] = (0.04, 0.22)
    """Items between these distances (m) in front of the robot's pelvis (between and just ahead of its feet; the
    pelvis moves back by about 0.2 m when the robot squats) ..."""
    reach_side: float = 0.22
    """... and within this distance (m) to either side can be picked up."""
    basket_distance: tuple[float, float] = (0.55, 0.75)
    """Baskets whose center is this far (m) from the robot (its footprint almost touching the basket) ..."""
    basket_angle: float = 0.6
    """... within this angle (rad) of its heading can be reached."""
    pick_posture: tuple[float, float] = (0.41, 1.18)
    """Pelvis height (m) and forward lean (rad) for picking something up from the ground (the deepest squat and
    strongest lean the walker was trained with)."""
    bag_posture: tuple[float, float] = (0.55, 0.7)
    """... and for taking a bag by its knot."""
    bag_drop_posture: tuple[float, float] = (0.6, 0.25)
    """... and for lowering a bag to set it down."""
    brace_posture: tuple[float, float] = (0.64, 0.15)
    """... and for bracing."""
    approach_height: float = 0.12
    """Height (m) of the grasp center above the item's grasp point while approaching (then on it)."""
    grasp_offset: float = 0.09
    """Distance (m) from the palm link towards the fingertips at which the fingers close around an object."""
    grasp_tolerance: float = 0.05
    drop_height: float = 0.10
    drop_inset: float = 0.08
    """Litter is let go this high (m) above a basket's rim, this far (m) inside its near edge."""
    crouch_time: float = 1.2
    reach_time: float = 1.5
    grasp_time: float = 0.3
    lift_time: float = 0.6
    stand_time: float = 1.2
    raise_time: float = 1.2
    release_time: float = 0.4
    lower_time: float = 0.6
    alert_time: float = 1.5
    joint_speed: float = 2.5
    """Largest speed (rad/s) of the upper-body joint targets (as in the walker's training)."""
    update_every: int = 4
    """Physics steps between updates (4: 50 Hz, the walker's rate)."""
    waist_range: float = 0.5
    """Largest waist rotation (rad) for reaching (as in the walker's training)."""
    ik_damping: float = 0.05
    ik_speed: float = 6.0
    """Largest arm joint change (rad/s) of the inverse kinematics."""
    ik_gain: float = 0.5
    """Fraction of the damped least-squares step taken per update (the joints lag their targets)."""
