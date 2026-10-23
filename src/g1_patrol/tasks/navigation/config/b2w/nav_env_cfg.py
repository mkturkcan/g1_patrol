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

The robot follows a planned route, as it would follow a global planner's path on the real robot (e.g. Nav2's): the
goal it is given is the points of the route 1 and 2.5 m ahead of its place along it, and its progress is measured along
the route (:mod:`g1_patrol.route`). In 30% of the episodes the route is planned without the small structures (chairs,
carts, bins, posts), which the robot then has to get around by itself; in a quarter it starts facing a random way.
Each episode caps the robot's planar speed somewhere between 0.4 and 1.5 m/s, and the policy is told the cap, so that
the speed set on the robot is one it was trained with.

The teacher sees privileged state: the nearest pedestrians, distances to structures per direction, the street edge or
drop and a terrain scan. The students see what the robot has, odometry and the route, and a height map from its RoboSense
RS-Helios-5515 lidar (:class:`B2WNavLidarStudentEnvCfg`) or from the lidar and the front and rear RealSense D435i depth
cameras, which see the ground close around the robot that the lidar cannot (:class:`B2WNavFusedStudentEnvCfg`).
"""

import math

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
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

from g1_patrol.b2w import (
    BASE_HEIGHT,
    BODY,
    CORNER_RADIUS,
    CROP_BOX,
    D435I,
    DEPTH_CAMERAS,
    FOOTPRINT,
    LIDAR,
    LIDAR_MOUNT,
    LIDARS,
    MAP,
    ROUTE_WAYPOINTS,
    SPEED_LIMIT_RANGE,
    depth_camera_quat,
)
from g1_patrol.noise import PerEpisodeAdditiveBiasCfg

from ... import mdp
from ...nyc.kinds import LAB_KINDS, TRAINING_KINDS
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
    spawn_turned_ratio=0.25,
    route_waypoints=ROUTE_WAYPOINTS,
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
    # velocity commands to the built-in controller (forward -0.8..1.5 m/s, lateral +-0.3 m/s, yaw rate +-1.2 rad/s),
    # planar speed capped per episode at 0.4..1.5 m/s (observed; the robot's max_speed)
    base = mdp.BaseVelocityActionCfg(asset_name="robot", scenario_name="nyc", speed_limit_range=SPEED_LIMIT_RANGE)
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
        speed_limit = ObsTerm(func=mdp.speed_limit, params={"action_name": "base"})
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
    pedestrian_contact = RewTerm(func=mdp.is_terminated_term, weight=-300.0, params={"term_keys": "pedestrian_collision"})
    obstacle_contact = RewTerm(func=mdp.is_terminated_term, weight=-300.0, params={"term_keys": "obstacle_collision"})
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
    # structures: keep 0.2 m away (doors are only 0.3 m wider than the robot) and slow down where it is tight: the
    # footprint's fastest point (turning swings the corners) at most 0.3 m/s within 0.1 m, 1.5 m/s from 1 m
    obstacle_margin = RewTerm(func=mdp.obstacle_margin, weight=-20.0, params={"command_name": "nyc", "margin": 0.2})
    obstacle_speed = RewTerm(
        func=mdp.obstacle_speed,
        weight=-5.0,
        params={"command_name": "nyc", "slow_speed": 0.3, "near": 0.1, "far": 1.0, "free_speed": 1.5, "turning": True},
    )
    # follow the planned route (it keeps clear of structures) within 0.2 m, unless something is in the way
    route_deviation = RewTerm(func=mdp.route_deviation, weight=-3.0, params={"command_name": "nyc", "free": 0.2})
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
        self.episode_length_s = 60.0
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


CAMERA_TARGETS = SENSOR_TARGETS[:-1]
"""What the depth cameras see: as the lidar, but not the body (they sit inside the box standing in for it and do not
see the robot)."""


def depth_camera_cfg(name: str) -> MultiMeshRayCasterCameraCfg:
    """A RealSense D435i depth stream at its URDF mount (``front`` or ``rear``; 45 deg down)."""
    pos, _ = DEPTH_CAMERAS[name]
    hfov, vfov = math.radians(D435I["hfov"]), math.radians(D435I["vfov"])
    return MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=pos, rot=depth_camera_quat(name), convention="world"),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2.0 * math.tan(hfov / 2),
            vertical_aperture=2.0 * math.tan(vfov / 2),
            width=D435I["width"],
            height=D435I["height"],
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=6.0,
        mesh_prim_paths=CAMERA_TARGETS,
    )


@configclass
class B2WStudentSceneCfg(B2WSceneCfg):
    lidar = lidar_cfg()


@configclass
class B2WFusedStudentSceneCfg(B2WStudentSceneCfg):
    front_depth = depth_camera_cfg("front")
    rear_depth = depth_camera_cfg("rear")


@configclass
class StudentProprioCfg(ObsGroup):
    """What the robot knows without exteroception: its route (waypoints from localization: an error of up to 10 cm
    that stays for the episode plus 5 cm noise), its velocity (from the controller's odometry: noise plus a bias that
    stays for the episode), its tilt, the last command and its speed limit."""

    goal = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "nyc"},
        noise=PerEpisodeAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.05, n_max=0.05), bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1)
        ),
    )
    velocity = ObsTerm(
        func=mdp.base_velocity,
        params={"action_name": "base"},
        noise=PerEpisodeAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.05, n_max=0.05), bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03)
        ),
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
    actions = ObsTerm(func=mdp.last_action, params={"action_name": "base"})
    speed_limit = ObsTerm(func=mdp.speed_limit, params={"action_name": "base"})

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


DEPTH_IN_MAP = tuple(
    {"sensor_cfg": SceneEntityCfg(f"{name}_depth"), "min_range": D435I["min_range"], "max_range": D435I["max_range"],
     "noise_std": 0.03, "dropout": 0.1, "episode_dropout": 0.05}
    for name in DEPTH_CAMERAS
)
"""The depth cameras' returns in the fused map: 0.3-4 m, stereo noise growing with distance, 10 % missing pixels,
and a camera that delivers nothing for 5 % of the episodes."""


@configclass
class FusedMapCfg(ObsGroup):
    """Height map from the lidar and the front and rear depth cameras (as :class:`LidarMapCfg`)."""

    height_map = ObsTerm(
        func=mdp.accumulated_height_map,
        params={**LidarMapCfg().lidar_map.params, "extra_sensors": DEPTH_IN_MAP},
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
class FusedStudentObservationsCfg(ObservationsCfg):
    student: StudentProprioCfg = StudentProprioCfg()
    height_map: FusedMapCfg = FusedMapCfg()


@configclass
class B2WNavLidarStudentEnvCfg(B2WNavEnvCfg):
    """Student with the lidar only. The teacher's privileged state stays in the ``policy`` group for distillation."""

    scene: B2WStudentSceneCfg = B2WStudentSceneCfg(num_envs=4096, env_spacing=15.0)
    observations: LidarStudentObservationsCfg = LidarStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.decimation * self.sim.dt



@configclass
class B2WNavFusedStudentEnvCfg(B2WNavEnvCfg):
    """Student with the lidar and the two depth cameras fused into one height map."""

    scene: B2WFusedStudentSceneCfg = B2WFusedStudentSceneCfg(num_envs=4096, env_spacing=15.0)
    observations: FusedStudentObservationsCfg = FusedStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for sensor in (self.scene.lidar, self.scene.front_depth, self.scene.rear_depth):
            sensor.update_period = self.decimation * self.sim.dt


##
# The lab: half the tiles are sections of the lab hall (:mod:`...nyc.makerlab`), half the general mixture.
##

LAB_TERRAIN = nyc_terrain_generator_cfg(
    kinds=TRAINING_KINDS + LAB_KINDS,
    proportions={kind: float(len(TRAINING_KINDS)) for kind in LAB_KINDS},
    route_half_width=FOOTPRINT[1],
)
"""13 x 2 columns of 10 levels: 13 of the lab, 13 of the street and indoor kinds."""
LAB_FLOOR = (0.0, 0.7)
"""Share of the floor returns beyond 2 m lost per episode (the lab's polished concrete)."""


@configclass
class B2WLabNavEnvCfg(B2WNavEnvCfg):
    """Teacher specialized on the lab (fine-tuned from the general teacher)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LAB_TERRAIN


@configclass
class B2WLabFusedStudentEnvCfg(B2WNavFusedStudentEnvCfg):
    """Fused student (lidar and depth cameras) on the lab mixture, with the polished floor."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LAB_TERRAIN
        self.observations.height_map.height_map.params["floor_dropout"] = LAB_FLOOR


@configclass
class B2WLabLidarStudentEnvCfg(B2WNavLidarStudentEnvCfg):
    """Lidar-only student on the lab mixture, with the polished floor."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LAB_TERRAIN
        self.observations.lidar_map.lidar_map.params["floor_dropout"] = LAB_FLOOR
