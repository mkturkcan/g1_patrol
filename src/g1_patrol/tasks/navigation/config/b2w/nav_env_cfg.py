"""Unitree B2-W navigation on New York streets and indoors -- privileged teacher and lidar student.

The B2-W, a wheeled quadruped, is driven through Unitree's built-in sport-mode controller: the navigation policy
commands body velocities (forward, lateral, yaw rate) at 10 Hz, and the robot is simulated as a kinematic body that
follows them through a randomized model of the controller (latency, response time, acceleration limits, gain errors,
deadband, drift; :class:`mdp.BaseVelocityActionCfg`). No legs are simulated, which makes a step cheap.

Episodes take place on the street tiles of the G1 task and on indoor tiles (labs, corridors, doorways between rooms,
offices, warehouses with a loading dock, lobbies with a stairwell; :mod:`g1_patrol.tasks.navigation.nyc.indoor`), with
the same pedestrian traffic and difficulty curriculum. The robot's footprint is a 1.12 x 0.58 m rectangle with rounded
corners; structures whose bottom is higher than 1 m do not block it (it is 0.76 m tall with its lidar); driving over a
drop (stairwell, loading dock) ends the episode.

The teacher sees privileged state: the nearest pedestrians, distances to structures per direction, the street edge or
drop and a terrain scan. The student (:class:`B2WNavLidarStudentEnvCfg`) sees what the robot has: its lidar, accumulated
into a height map, odometry and the goal.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

from g1_patrol.b2w import BASE_HEIGHT, BODY, CORNER_RADIUS, CROP_BOX, FOOTPRINT, LIDAR, LIDAR_MOUNT, LIDARS, MAP
from g1_patrol.noise import PerEpisodeAdditiveBiasCfg

from ... import mdp
from ...nyc.indoor import TRAINING_KINDS
from ...nyc.terrain_cfg import NycTerrainImporterCfg, nyc_terrain_generator_cfg

##
# Robot.
##

NUM_PEDESTRIANS = 20

SCENARIO = mdp.NycScenarioCommandCfg(
    asset_name="robot",
    pedestrians_name="pedestrians",
    debug_vis=True,
    robot_radius=CORNER_RADIUS,
    footprint=FOOTPRINT,
    overhead_z=1.0,
    velocity_action="base",
    limb_bodies=None,
    robot_spawn_height=BASE_HEIGHT,
    robot_nominal_speed=1.0,
    spawn_heading_noise=0.3,
    standing_spread=0.2,
    standing_nearest_path=True,
)


def _pedestrians(count: int) -> RigidObjectCollectionCfg:
    """Kinematic capsules (about 0.5 m wide, 1.7 m tall), parked below the ground until the scenario places them."""
    shape = sim_utils.CapsuleCfg(
        radius=SCENARIO.pedestrian_radius,
        height=1.2,
        axis="Z",
        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        mass_props=sim_utils.MassCfg(mass=70.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.1)),
    )
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"ped_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Ped_{i}",
                spawn=shape,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for i in range(count)
        }
    )


ROBOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.CuboidCfg(
        size=BODY,
        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        mass_props=sim_utils.MassCfg(mass=85.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BASE_HEIGHT)),
)
"""The B2-W as a kinematic body (base_link), moved by :class:`mdp.BaseVelocityActionCfg`."""


@configclass
class B2WSceneCfg(InteractiveSceneCfg):
    terrain = NycTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=nyc_terrain_generator_cfg(kinds=TRAINING_KINDS, route_half_width=FOOTPRINT[1]),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )
    robot = ROBOT_CFG
    # teacher's terrain scan (ground only: curbs, potholes, drops), 4.2 x 4.2 m around the robot
    nav_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.3, size=[4.2, 4.2]),
        mesh_prim_paths=["/World/ground"],
    )
    pedestrians = _pedestrians(NUM_PEDESTRIANS)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


##
# MDP.
##


@configclass
class ActionsCfg:
    # velocity commands to the built-in controller (forward -0.8..1.5 m/s, lateral +-0.3 m/s, yaw rate +-1.2 rad/s)
    base = mdp.BaseVelocityActionCfg(asset_name="robot", scenario_name="nyc")
    # moves the pedestrians at every physics step; takes no policy inputs
    pedestrians = mdp.PedestrianMotionActionCfg(asset_name="pedestrians", scenario_name="nyc")


@configclass
class CommandsCfg:
    nyc = SCENARIO


@configclass
class ObservationsCfg:
    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged state."""

        goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "nyc"})
        velocity = ObsTerm(func=mdp.base_velocity, params={"action_name": "base"})
        pedestrians = ObsTerm(func=mdp.nearby_pedestrians, params={"command_name": "nyc", "count": 10})
        obstacles = ObsTerm(func=mdp.obstacle_sectors, params={"command_name": "nyc", "num_sectors": 72})
        street = ObsTerm(func=mdp.street_state, params={"command_name": "nyc"})
        terrain = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("nav_scanner")}, clip=(-1.0, 1.0)
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "base"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: TeacherCfg = TeacherCfg()


@configclass
class RewardsCfg:
    # terms are scaled by the step length (0.1 s): weight 2 on progress pays 2 per meter gained, weights of +-100 on
    # one-step events pay +-10
    progress = RewTerm(func=mdp.nyc_goal_progress, weight=2.0, params={"command_name": "nyc", "max_speed": 1.5})
    success = RewTerm(func=mdp.nyc_goal_reached, weight=100.0, params={"command_name": "nyc"})
    pedestrian_contact = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "pedestrian_collision"})
    obstacle_contact = RewTerm(func=mdp.is_terminated_term, weight=-150.0, params={"term_keys": "obstacle_collision"})
    fall = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "fall"})
    roadway = RewTerm(func=mdp.is_terminated_term, weight=-150.0, params={"term_keys": "roadway"})
    # people: keep 0.8 m between bodies, slow down near them (0.3 m/s within 0.3 m, 1.5 m/s from 2 m), avoid close
    # passes predicted within 2 s, give way
    personal_space = RewTerm(
        func=mdp.personal_space, weight=-6.0, params={"command_name": "nyc", "comfort_distance": 0.8}
    )
    social_speed = RewTerm(
        func=mdp.social_speed, weight=-6.0, params={"command_name": "nyc", "far": 2.0, "free_speed": 1.5}
    )
    collision_risk = RewTerm(
        func=mdp.collision_risk, weight=-4.0, params={"command_name": "nyc", "horizon": 2.0, "margin": 0.3}
    )
    pedestrian_delay = RewTerm(func=mdp.pedestrian_delay, weight=-1.0, params={"command_name": "nyc"})
    # structures: keep 0.2 m away (doors are only 0.3 m wider than the robot) and slow down where it is tight
    obstacle_margin = RewTerm(func=mdp.obstacle_margin, weight=-10.0, params={"command_name": "nyc", "margin": 0.2})
    obstacle_speed = RewTerm(
        func=mdp.obstacle_speed, weight=-2.0, params={"command_name": "nyc", "far": 0.5, "free_speed": 1.5}
    )
    # streets and drops: stay away from the curb of the roadway and from the edges of stairwells and docks
    street_margin = RewTerm(func=mdp.street_margin, weight=-10.0, params={"command_name": "nyc", "margin": 0.3})
    # smooth commands; mostly forward (the lidar sees all around, but people expect the robot to face its way)
    command_rate = RewTerm(func=mdp.command_rate, weight=-2.0, params={"action_name": "base"})
    command_lateral = RewTerm(func=mdp.command_lateral, weight=-1.5, params={"action_name": "base"})
    command_backward = RewTerm(func=mdp.command_backward, weight=-0.5, params={"action_name": "base"})
    command_yaw_rate = RewTerm(func=mdp.command_yaw_rate, weight=-0.15, params={"action_name": "base"})
    policy_output = RewTerm(func=mdp.policy_output_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.nyc_goal_reached_termination, params={"command_name": "nyc"})
    pedestrian_collision = DoneTerm(func=mdp.nyc_pedestrian_collision, params={"command_name": "nyc"})
    obstacle_collision = DoneTerm(func=mdp.nyc_obstacle_collision, params={"command_name": "nyc"})
    roadway = DoneTerm(func=mdp.nyc_roadway, params={"command_name": "nyc"})
    fall = DoneTerm(func=mdp.base_fall, params={"action_name": "base"})


@configclass
class CurriculumCfg:
    levels = CurrTerm(func=mdp.nyc_adaptive_curriculum, params={"unlock_success": 0.35, "explore_ratio": 0.15})


@configclass
class B2WNavEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    scene: B2WSceneCfg = B2WSceneCfg(num_envs=4096, env_spacing=15.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # timing: 50 Hz physics (kinematic bodies only), 10 Hz navigation
        self.sim.dt = 0.02
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0
        # thousands of kinematic robots and pedestrians over the tile meshes need many broad-phase pairs
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        self.sim.physics.isaacsim_physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.nav_scanner.update_period = self.decimation * self.sim.dt
        # camera (headless video) follows robot 0 from behind and above
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-4.0, 0.0, 3.0)
        self.viewer.lookat = (1.5, 0.0, 0.3)

    def play_mode(self):
        super().play_mode()
        # evaluate students with their sensor noise (play mode turns observation corruption off)
        if hasattr(self.observations, "student"):
            self.observations.student.enable_corruption = True


##
# Student: the robot's own lidar (and odometry).
##

SENSOR_TARGETS = [
    "/World/ground",
    "/World/structures",
    *[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Ped_{i}") for i in range(NUM_PEDESTRIANS)],
    MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot"),
]
"""What the lidar sees: the ground, the structures, the pedestrians and the robot's own body (whose returns the map
drops, as on the robot)."""


def lidar_cfg(model: str = LIDAR) -> MultiMeshRayCasterCfg:
    """A RoboSense lidar on the B2-W's lidar mount (10 Hz; 1 deg horizontal resolution)."""
    channels, vfov, max_range = LIDARS[model]
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=LIDAR_MOUNT),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels, vertical_fov_range=vfov, horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.0
        ),
        max_distance=max_range,
        mesh_prim_paths=SENSOR_TARGETS,
    )


@configclass
class B2WStudentSceneCfg(B2WSceneCfg):
    lidar = lidar_cfg()


@configclass
class StudentProprioCfg(ObsGroup):
    """What the robot knows without exteroception: the goal (from localization, with 10 cm error), its velocity (from
    the controller's odometry: noise plus a bias that stays for the episode), its tilt and the last command."""

    goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "nyc"}, noise=Unoise(n_min=-0.1, n_max=0.1))
    velocity = ObsTerm(
        func=mdp.base_velocity,
        params={"action_name": "base"},
        noise=PerEpisodeAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.05, n_max=0.05), bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03)
        ),
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
    actions = ObsTerm(func=mdp.last_action, params={"action_name": "base"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class LidarMapCfg(ObsGroup):
    """Lidar height map, 8 x 8 m at 0.1 m around the robot (heading frame): accumulated over 8 s with drifting
    odometry, current scan, previous scan (3 x 80 x 80); 0-100 ms old. Returns from the robot itself and from above
    1 m (what the robot passes under) are dropped."""

    lidar_map = ObsTerm(
        func=mdp.accumulated_height_map,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "size": MAP["size"],
            "resolution": MAP["resolution"],
            "memory_s": MAP["memory_s"],
            "max_range": MAP["max_range"],
            "noise_std": 0.01,
            "dropout": 0.02,
            "z_range": MAP["z_range"],
            "unknown_value": MAP["unknown_value"],
            "robot_ground_offset": BASE_HEIGHT,
            "drift_pos": 0.005,
            "drift_yaw": 0.002,
            "crop_box": CROP_BOX,
            "drop_above": MAP["drop_above"],
        },
        delay_max_lag=1,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LidarStudentObservationsCfg(ObservationsCfg):
    student: StudentProprioCfg = StudentProprioCfg()
    lidar_map: LidarMapCfg = LidarMapCfg()


@configclass
class B2WNavLidarStudentEnvCfg(B2WNavEnvCfg):
    """Student with the lidar only. The teacher's privileged state stays in the ``policy`` group for distillation."""

    scene: B2WStudentSceneCfg = B2WStudentSceneCfg(num_envs=4096, env_spacing=15.0)
    observations: LidarStudentObservationsCfg = LidarStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.decimation * self.sim.dt

