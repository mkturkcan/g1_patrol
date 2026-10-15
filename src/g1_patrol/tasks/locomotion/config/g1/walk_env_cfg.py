"""Velocity-tracking locomotion for the Unitree G1 humanoid.

The policy receives a base-velocity command (forward, lateral, yaw rate) and outputs joint position targets.
This is the low-level "walking" skill that later patrol/navigation tasks will drive by issuing velocity commands.

The MDP (scene, observations, actions, events, terminations, physics backend presets) is inherited from Isaac Lab's
generic locomotion velocity environment; this module only specialises it for the G1.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    CurriculumCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from isaaclab_assets import G1_MINIMAL_CFG

from ... import mdp
from ...terrains import CITY_TERRAINS_CFG, STREET_TERRAINS_CFG


@configclass
class G1WalkRewards(RewardsCfg):
    """Reward terms for G1 walking."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # encourage alternating steps with a reasonable swing duration
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # keep joints that are not essential for locomotion near their default pose
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


def _follow_robot_camera(cfg: LocomotionVelocityRoughEnvCfg):
    """Point the viewport camera (used for headless video) at robot 0 and follow it."""
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (2.5, 2.5, 1.2)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)


##
# Rough terrain (procedurally generated stairs, slopes, rubble) -- the next step towards outdoor patrol.
##


@configclass
class G1WalkRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1WalkRewards = G1WalkRewards()

    def __post_init__(self):
        super().__post_init__()
        # physics
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # scene
        # PhysX and Newton enumerate the G1's joints/bodies in different orders, and checkpoints store columns by
        # position; pin one order so a policy trained on either backend runs on the other (and in later tasks)
        self.scene.robot = replace(
            G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot", joint_ordering="mjwarp", body_ordering="mjwarp"
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # terminations: falling over (torso touching the ground) ends the episode
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # events: randomize torso mass, keep the default pushes and reset velocities
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    def play_mode(self):
        super().play_mode()
        # evaluate with the robot's sensor noise (Isaac Lab's play mode turns observation corruption off)
        self.observations.policy.enable_corruption = True
        _follow_robot_camera(self)


##
# Flat ground -- the first, simplest task.
##


@configclass
class G1WalkFlatEnvCfg(G1WalkRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # physics: a flat plane needs far fewer contact constraints, and Newton/MuJoCo-Warp is the fastest backend here
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 95
        newton_mjwarp.solver_cfg.nconmax = 10
        self.sim.physics.default = newton_mjwarp
        # scene: flat plane, no height scanner or terrain curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # commands: forward 0-1 m/s, sideways +-0.5 m/s, turning +-1 rad/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )


##
# Street -- curbs, sills, stairs and rough pavement, with shoves from passers-by. The walker the navigator drives.
##


@configclass
class StreetCurriculumCfg(CurriculumCfg):
    # shoves grow from 0.5 to 1.2 m/s velocity kicks between PPO iterations 1000 and 3500
    # (a hard human shove, ~150 N for 0.2 s, changes the 35 kg G1's velocity by ~1 m/s)
    push_strength = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.push_robot.params",
            "modify_fn": mdp.ramp_push_velocity,
            "modify_params": {"initial": 0.5, "final": 1.2, "start_step": 24 * 1000, "end_step": 24 * 3500},
        },
    )


@configclass
class G1WalkStreetEnvCfg(G1WalkRoughEnvCfg):
    curriculum: StreetCurriculumCfg = StreetCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.terrain.terrain_generator = STREET_TERRAINS_CFG
        # commands: also step backwards and sideways (making room in a crowd), and stand still more often
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.1
        # events: a shove every 4-8 s (strength set by the curriculum)
        self.events.push_robot.interval_range_s = (4.0, 8.0)

    def play_mode(self):
        super().play_mode()
        # play mode removes the push events, and with them the push curriculum
        self.curriculum.push_strength = None


##
# City (v2) -- an upright, periodic gait with realistic motor limits on New York street surfaces, and shoves.
##

GAIT_PERIOD = 0.8
"""Seconds per gait cycle (two steps)."""

FEET = ".*_ankle_roll_link"


def realistic_g1(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """The G1 with motor models close to the real robot's.

    Leg and ankle joints are explicit PD controllers at the gains Unitree uses on the robot (hip 100, knee 150,
    ankle 40 Nm/rad), with the real motors' torque and speed limits and 0-20 ms of actuation latency. Isaac Lab's
    default G1 allows 300 Nm at every leg joint, 2-3x what the real motors produce, which lets a policy get away with a
    crouched, high-effort gait that would not transfer.
    """
    robot = replace(G1_MINIMAL_CFG, prim_path=prim_path, joint_ordering="mjwarp", body_ordering="mjwarp")
    actuators = dict(robot.actuators)
    legs_effort = {".*_hip_.*": 88.0, ".*_knee_joint": 139.0, "torso_joint": 88.0}
    actuators["legs"] = DelayedPDActuatorCfg(
        joint_names_expr=actuators["legs"].joint_names_expr,
        stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 150.0, "torso_joint": 200.0},
        damping={".*_hip_.*": 2.0, ".*_knee_joint": 4.0, "torso_joint": 5.0},
        armature=0.01,
        actuator_effort_limit=legs_effort,
        joint_effort_limit=legs_effort,
        joint_velocity_limit={".*_hip_.*": 32.0, ".*_knee_joint": 20.0, "torso_joint": 32.0},
        min_delay=0,
        max_delay=4,  # physics steps of 5 ms
    )
    actuators["feet"] = DelayedPDActuatorCfg(
        joint_names_expr=actuators["feet"].joint_names_expr,
        stiffness=40.0,
        damping=2.0,
        armature=0.01,
        actuator_effort_limit=50.0,
        joint_effort_limit=50.0,
        joint_velocity_limit=37.0,
        min_delay=0,
        max_delay=4,
    )
    actuators["arms"] = replace(
        actuators["arms"],
        joint_effort_limit={
            ".*_shoulder_.*": 25.0,
            ".*_elbow_.*": 25.0,
            ".*_(zero|one|two|three|four|five|six)_joint": 5.0,
        },
    )
    robot.actuators = actuators
    return robot


@configclass
class BlindObservationsCfg(ObservationsCfg):
    """The walker sees only what the G1 measures (IMU, joint encoders), its last action, the command and the gait
    clock, with sensor noise, per-episode sensor biases (IMU mounting and gyro bias, encoder offsets), 0-20 ms of
    latency and 5 control steps of history. The critic additionally sees the base velocity and the terrain
    (asymmetric actor-critic, training only)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.2, n_max=0.2), bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02)
            ),
            delay_max_lag=1,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.05, n_max=0.05), bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03)
            ),
            delay_max_lag=1,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.01, n_max=0.01), bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02)
            ),
            delay_max_lag=1,
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), delay_max_lag=1)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1CityRewards(G1WalkRewards):
    """G1 walking rewards plus posture and gait shaping for an upright, natural, periodic walk."""

    # stay tall and upright
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.76, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    torso_upright = RewTerm(
        func=mdp.body_flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}
    )
    # step in time with the gait clock, lifting the swing foot and keeping the feet hip-width apart
    gait = RewTerm(
        func=mdp.gait_contact,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "period": GAIT_PERIOD,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET), "target_height": 0.08},
    )
    feet_distance = RewTerm(
        func=mdp.feet_lateral_distance,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET), "min_distance": 0.18, "max_distance": 0.40},
    )
    # hold still when asked to stand
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)


@configclass
class G1BlindCityRewards(G1CityRewards):
    """City rewards plus smoothness (mechanical power, impacts, jerky targets) and curb safety for a blind walker."""

    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    feet_impact = RewTerm(
        func=mdp.feet_impact, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)}
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)}
    )
    action_jerk = RewTerm(func=mdp.action_second_difference, weight=-0.002)
    # hold the commanded line: no sustained sideways or heading drift (1 s low-pass removes the gait's sway)
    velocity_drift = RewTerm(
        func=mdp.velocity_drift, weight=-10.0, params={"command_name": "base_velocity", "time_constant": 1.0}
    )


@configclass
class G1WalkCityEnvCfg(G1WalkRoughEnvCfg):
    """Blind walker for the navigation tasks: proprioception only, deployment-like motors and latencies, randomized
    dynamics, smooth gait shaping, on curbs, sills, potholes, cobblestones and ramps, with shoves."""

    observations: BlindObservationsCfg = BlindObservationsCfg()
    rewards: G1BlindCityRewards = G1BlindCityRewards()
    curriculum: StreetCurriculumCfg = StreetCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = realistic_g1()
        self.scene.terrain.terrain_generator = CITY_TERRAINS_CFG
        # contact friction = the robot's (randomized) foot friction in both simulators: MuJoCo-Warp takes the larger of
        # the two coefficients, so a ground at 1.0 would hide every value below it; a low ground coefficient combined
        # with "max" (PhysX) resolves to the robot's value, as the multiplied 1.0 ground of the navigation tasks does
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max", restitution_combine_mode="multiply", static_friction=0.1, dynamic_friction=0.1
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        # actions: smaller joint offsets per unit action (smoother, as on the real robot)
        self.actions.joint_pos.scale = 0.25
        # commands: walking pace, small sidesteps, turning, backing off, and frequent standing
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.rel_standing_envs = 0.15
        # rewards: tighter velocity tracking; stronger posture, smoothness and calm upper body than the base G1 recipe
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.params["target_height"] = 0.10  # blind: lift the feet clear of curbs and sills
        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.3
        # events: friction of wet, painted and metal surfaces, a payload of up to 5 kg and the torso's center of mass,
        # link masses, motor gains and strength, joint friction and armature, shoves every 4-8 s (curriculum)
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.25, 1.0)
        self.events.add_base_mass.params.update(
            {"mass_distribution_params": (-1.0, 5.0), "operation": "add", "distribution": "uniform"}
        )
        self.events.motor_strength = EventTerm(
            func=mdp.randomize_motor_strength,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "actuators": ["legs", "feet"], "strength_range": (0.8, 1.0)},
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
            },
        )
        self.events.link_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis|.*_(hip|knee|ankle)_.*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.85, 1.15),
                "operation": "scale",
            },
        )
        leg_joints = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        self.events.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={"asset_cfg": leg_joints, "friction_distribution_params": (0.0, 0.3), "operation": "abs"},
        )
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={"asset_cfg": leg_joints, "armature_distribution_params": (0.8, 1.2), "operation": "scale"},
        )
        self.events.push_robot.interval_range_s = (4.0, 8.0)

    def play_mode(self):
        super().play_mode()
        self.curriculum.push_strength = None
