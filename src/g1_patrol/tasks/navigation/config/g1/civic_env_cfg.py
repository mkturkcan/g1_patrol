"""Civic New York street navigation for the Unitree G1 -- privileged teacher.

The New York street task (:mod:`.nyc_env_cfg`) with what a robot on patrol meets besides the crowd, and the jobs it
does along the way:

* litter on the sidewalk, which it picks up and drops into a litter basket, and tied trash bags, which it carries by
  the knot to the bag pile at the curb (:mod:`..mdp.nyc_trash`, :mod:`..mdp.nyc_manipulation`);
* people who need more room (wheelchairs, strollers, white canes), dogs on leashes, friends walking together, lines of
  people waiting, cyclists on the sidewalk; someone who blocks its way, someone who attacks it (it braces, keeps its
  footing and gets away, without going at the attacker or into bystanders or the street) (:mod:`..mdp.nyc_social`);
* someone who has fallen, whom it walks up to and calls help for, and pedestrian signals at crosswalks
  (:mod:`..mdp.nyc_assist`).

The navigation policy runs at 10 Hz and commands the frozen reaching walker (``G1Patrol-Walk-Reach``: velocity and
posture, while the arms are driven by the hands controller). Incidents and special pedestrians become more frequent,
faster and stronger with the tile's difficulty level, which the success-driven curriculum raises.
"""

import json

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

from g1_patrol.assets import G1_PATROL_ASSETS_DIR
from g1_patrol.noise import PerEpisodeAdditiveBiasCfg
from g1_patrol.tasks.locomotion.config.g1.walk_env_cfg import G1WalkReachEnvCfg

from ... import mdp
from ...mdp.nyc_trash import PARK_Z
from ...nyc.terrain_cfg import nyc_terrain_generator_cfg
from .nyc_env_cfg import (
    D435I_CFG,
    MID360_CFG,
    DepthCfg,
    DepthMapCfg,
    G1NycNavEnvCfg,
    LidarMapCfg,
    NycSceneCfg,
    RewardsCfg,
)

REACH_WALKER_ENV_CFG = G1WalkReachEnvCfg()
"""Environment of the frozen reaching walker; its robot, sensors, physics, observations and actions are reused."""

REACH_WALKER_POLICY_PATH = str(G1_PATROL_ASSETS_DIR / "policies" / "g1_walk_reach.pt")

TRASH_DIR = str(G1_PATROL_ASSETS_DIR / "trash")
TRASH_ITEMS = (
    # trash bags (imported models, see the assets' ATTRIBUTION.md)
    "bag_crumpled_black", "bag_kitchen_black", "bag_scanned_black", "bag_tied_black", "bag_squat_black",
    "bag_recycling_clear",
    # litter: imported models and generated ones (a "thank you" bag, a bottle in a paper bag, a foam clamshell)
    "cup_to_go", "cup_paper_dented", "can_tallboy", "can_dented", "bottle_crushed", "chip_bag_fresh",
    "takeout_box_crumpled", "pizza_box_open", "paper_wad", "cigarette_pack", "beer_bottle", "plastic_bag_tied",
    "newspaper_folded", "deli_bag", "bottle_in_bag", "foam_clamshell",
)
"""Items of the trash manifest in the scene, one rigid body each per environment."""
with open(f"{TRASH_DIR}/manifest.json") as _f:
    _manifest = json.load(_f)
TRASH_FILES = {name: _manifest[name]["file"] for name in TRASH_ITEMS}

NUM_REGULAR, NUM_WIDE, NUM_DOGS = 20, 2, 2

CIVIC_SCENARIO = mdp.NycScenarioCommandCfg(
    asset_name="robot",
    pedestrians_name="pedestrians",
    trash_name="trash",
    trash_dir=TRASH_DIR,
    trash_items=TRASH_ITEMS,
    trash_count=(0, 3),
    bag_count=(0, 1),
    fallen_name="fallen",
    social=True,
    num_regular=NUM_REGULAR,
    num_wide=NUM_WIDE,
    num_dogs=NUM_DOGS,
    signals=True,
    debug_vis=True,
)


def _civic_pedestrians() -> RigidObjectCollectionCfg:
    """Kinematic bodies, parked below the ground until the scenario places them: people (capsules about 0.5 m wide,
    1.7 m tall), wheelchairs and strollers (0.8 m wide, 1.3 m tall), dogs (lying capsules 0.75 m long, 0.6 m high)."""
    s = CIVIC_SCENARIO

    def body(radius, height, axis, color):
        return sim_utils.CapsuleCfg(
            radius=radius,
            height=height,
            axis=axis,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=70.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )

    shapes = [body(s.pedestrian_radius, 1.2, "Z", (0.9, 0.45, 0.1))] * NUM_REGULAR
    shapes += [body(s.wide_radius, 0.5, "Z", (0.2, 0.45, 0.9))] * NUM_WIDE
    shapes += [body(s.dog_radius, 0.4, "X", (0.55, 0.35, 0.2))] * NUM_DOGS
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"ped_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Ped_{i}", spawn=shape, init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0))
            )
            for i, shape in enumerate(shapes)
        }
    )


def _trash() -> RigidObjectCollectionCfg:
    """One dynamic rigid body per litter item and trash bag (visual meshes, a convex collision hull and the mass from
    the asset), falling freely far below the street until the scenario places it."""
    return RigidObjectCollectionCfg(
        rigid_objects={
            name: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Trash_{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{TRASH_DIR}/{TRASH_FILES[name]}",
                    rigid_props=[
                        sim_utils.UsdPhysicsRigidBodyCfg(),
                        PhysxRigidBodyCfg(linear_damping=0.2, angular_damping=1.0, max_depenetration_velocity=1.0),
                    ],
                    collision_props=[PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8 * k - 8.0, 0.0, PARK_Z)),
            )
            for k, name in enumerate(TRASH_ITEMS)
        }
    )


@configclass
class CivicSceneCfg(NycSceneCfg):
    pedestrians = _civic_pedestrians()
    trash = _trash()
    # someone lying on the ground (kinematic), parked below the ground unless the episode has one
    fallen = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fallen",
        spawn=sim_utils.CapsuleCfg(
            radius=0.2,
            height=1.3,
            axis="X",
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=70.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )


@configclass
class CivicActionsCfg:
    walker = mdp.WalkerPolicyActionCfg(
        asset_name="robot",
        policy_path=REACH_WALKER_POLICY_PATH,
        low_level_decimation=REACH_WALKER_ENV_CFG.decimation,
        low_level_actions=REACH_WALKER_ENV_CFG.actions.joint_pos,
        low_level_observations=REACH_WALKER_ENV_CFG.observations.policy,
        # a walking pace (1.3 m/s only to get away from an attacker; see the walking-speed reward), small sidesteps
        command_ranges=((-0.4, 1.3), (-0.15, 0.15), (-1.5, 1.5)),
        squash="tanh",
        smoothing=0.5,
        posture_command=True,
        debug_vis=False,
    )
    # moves the pedestrians at every physics step; takes no policy inputs
    pedestrians = mdp.PedestrianMotionActionCfg(asset_name="pedestrians", scenario_name="nyc", update_every=4)
    # trigger (pick up / put down / call for help) and brace
    hands = mdp.HandsActionCfg(asset_name="robot", scenario_name="nyc", walker_name="walker")


@configclass
class CivicCommandsCfg:
    nyc = CIVIC_SCENARIO


@configclass
class CivicObservationsCfg:
    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged state."""

        goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "nyc"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pedestrians = ObsTerm(func=mdp.civic_pedestrians, params={"command_name": "nyc", "count": 10})
        obstacles = ObsTerm(func=mdp.obstacle_sectors, params={"command_name": "nyc", "num_sectors": 72})
        street = ObsTerm(func=mdp.street_state, params={"command_name": "nyc"})
        terrain = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("nav_scanner")}, clip=(-1.0, 1.0))
        trash = ObsTerm(func=mdp.trash_state, params={"command_name": "nyc", "count": 3})
        baskets = ObsTerm(func=mdp.basket_state, params={"command_name": "nyc", "count": 2})
        piles = ObsTerm(func=mdp.pile_state, params={"command_name": "nyc", "count": 1})
        hands = ObsTerm(func=mdp.hands_state, params={"action_name": "hands"})
        fallen = ObsTerm(func=mdp.fallen_state, params={"command_name": "nyc"})
        signal = ObsTerm(func=mdp.signal_state, params={"command_name": "nyc"})
        attack = ObsTerm(func=mdp.attack_state, params={"command_name": "nyc"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: TeacherCfg = TeacherCfg()


@configclass
class CivicRewardsCfg(RewardsCfg):
    # the street task's terms (contacts, falls, the roadway, personal space, giving way, margins, walking style), with
    # goal progress paused while getting away from an attacker and success also for helping someone who has fallen
    progress = RewTerm(func=mdp.civic_goal_progress, weight=2.0, params={"command_name": "nyc", "max_speed": 1.0})
    success = RewTerm(func=mdp.civic_success, weight=100.0, params={"command_name": "nyc"})
    walking_speed = RewTerm(func=mdp.walking_speed, weight=-4.0, params={"command_name": "nyc", "limit": 1.0})
    # litter and bags: +15 per item picked up, +40 per item in a basket, +50 per bag at the bag pile (worth the detour
    # and the time standing still), and progress towards the next item or place to put it (as much per meter as
    # progress towards the goal)
    picked = RewTerm(func=mdp.hands_event, weight=150.0, params={"event": "picked"})
    pick_failed = RewTerm(func=mdp.hands_event, weight=-20.0, params={"event": "failed"})
    # a small cost for triggering with nothing to do, so that the trigger keeps being explored
    invalid_trigger = RewTerm(func=mdp.hands_event, weight=-0.2, params={"event": "invalid"})
    deposited = RewTerm(func=mdp.trash_event, weight=400.0, params={"command_name": "nyc", "event": "trash_deposited"})
    bag_delivered = RewTerm(func=mdp.trash_event, weight=500.0, params={"command_name": "nyc", "event": "bags_delivered"})
    trash_lost = RewTerm(func=mdp.trash_event, weight=-20.0, params={"command_name": "nyc", "event": "trash_lost"})
    trash_progress = RewTerm(func=mdp.trash_progress, weight=2.0, params={"command_name": "nyc", "max_distance": 12.0})
    # attacks: get away, keep the attacker at a distance, never go at them, brace when they are close
    flee = RewTerm(func=mdp.flee, weight=2.0, params={"command_name": "nyc"})
    threat_proximity = RewTerm(func=mdp.threat_proximity, weight=-4.0, params={"command_name": "nyc", "distance": 1.5})
    approach_threat = RewTerm(func=mdp.approach_threat, weight=-4.0, params={"command_name": "nyc"})
    brace = RewTerm(func=mdp.brace_under_threat, weight=2.0, params={"command_name": "nyc", "distance": 1.0})
    brace_idle = RewTerm(func=mdp.brace_without_threat, weight=-1.0, params={"command_name": "nyc"})
    shoved = RewTerm(func=mdp.shoved, weight=-30.0, params={"command_name": "nyc"})
    # people who need more room (1.2 m comfort distance, and three times the cost of making them wait), groups and lines
    vulnerable_space = RewTerm(func=mdp.vulnerable_space, weight=-6.0, params={"command_name": "nyc", "comfort_distance": 1.2})
    vulnerable_delay = RewTerm(func=mdp.vulnerable_delay, weight=-2.0, params={"command_name": "nyc"})
    split_group = RewTerm(func=mdp.split_group, weight=-100.0, params={"command_name": "nyc"})


@configclass
class CivicTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.civic_success_termination, params={"command_name": "nyc"})
    pedestrian_collision = DoneTerm(func=mdp.civic_pedestrian_collision, params={"command_name": "nyc"})
    obstacle_collision = DoneTerm(func=mdp.nyc_obstacle_collision, params={"command_name": "nyc"})
    roadway = DoneTerm(func=mdp.nyc_roadway, params={"command_name": "nyc"})
    fall = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class CivicCurriculumCfg:
    levels = CurrTerm(
        func=mdp.civic_curriculum, params={"unlock_success": 0.35, "explore_ratio": 0.15, "command_name": "nyc"}
    )


@configclass
class G1CivicNavEnvCfg(G1NycNavEnvCfg):
    scene: CivicSceneCfg = CivicSceneCfg(num_envs=2048, env_spacing=15.0)
    actions: CivicActionsCfg = CivicActionsCfg()
    commands: CivicCommandsCfg = CivicCommandsCfg()
    observations: CivicObservationsCfg = CivicObservationsCfg()
    rewards: CivicRewardsCfg = CivicRewardsCfg()
    terminations: CivicTerminationsCfg = CivicTerminationsCfg()
    curriculum: CivicCurriculumCfg = CivicCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        walker = REACH_WALKER_ENV_CFG
        # the reaching walker's robot, sensors and randomization (same robot model and timing as the city walker)
        self.scene.robot = walker.scene.robot
        self.events = walker.events
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        self.scene.terrain.terrain_generator = nyc_terrain_generator_cfg(civic=True)
        self.episode_length_s = 60.0
        physx = self.sim.physics.isaacsim_physx
        physx.gpu_max_rigid_patch_count = 2**19
        physx.gpu_max_rigid_contact_count = 2**23


##
# Students: the G1's own head sensors (and the city's map and signal data; see :mod:`..mdp.civic_perception`).
##

CIVIC_SENSOR_TARGETS = [
    "/World/ground",
    "/World/structures",
    *[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Ped_{i}") for i in range(NUM_REGULAR + NUM_WIDE + NUM_DOGS)],
    MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Fallen"),
    *[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Trash_{name}") for name in TRASH_ITEMS if name.startswith("bag_")],
]
"""What the lidar and depth camera see: the ground, the structures, people, dogs, wheelchairs, someone lying on the
ground and trash bags (litter is too small for the height maps; the color image detects it)."""


@configclass
class CivicStudentSceneCfg(CivicSceneCfg):
    mid360 = MID360_CFG.replace(mesh_prim_paths=CIVIC_SENSOR_TARGETS)
    d435i = D435I_CFG.replace(mesh_prim_paths=CIVIC_SENSOR_TARGETS)


@configclass
class CivicStudentCfg(ObsGroup):
    """The robot's own state and estimates (as the street students: goal from localization, motion from the state
    estimator, last command), its hands, what it knows from the city map and data, tracked people and remembered
    litter."""

    goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "nyc"}, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        noise=PerEpisodeAdditiveBiasCfg(noise_cfg=Unoise(n_min=-0.1, n_max=0.1), bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05)),
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=PerEpisodeAdditiveBiasCfg(noise_cfg=Unoise(n_min=-0.2, n_max=0.2), bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02)),
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    hands = ObsTerm(func=mdp.hands_state, params={"action_name": "hands"})
    places = ObsTerm(func=mdp.known_places, params={"command_name": "nyc"})
    people = ObsTerm(func=mdp.tracked_people, params={"command_name": "nyc", "count": 8, "sensor": "lidar", "max_range": 10.0})
    litter = ObsTerm(func=mdp.detected_litter, params={"command_name": "nyc", "count": 3})
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CivicLidarStudentObservationsCfg(CivicObservationsCfg):
    student: CivicStudentCfg = CivicStudentCfg()
    lidar_map: LidarMapCfg = LidarMapCfg()


@configclass
class CivicCameraStudentObservationsCfg(CivicObservationsCfg):
    student: CivicStudentCfg = CivicStudentCfg()
    depth: DepthCfg = DepthCfg()
    depth_map: DepthMapCfg = DepthMapCfg()

    def __post_init__(self):
        # the camera-only robot tracks people in the depth camera's view
        self.student.people.params.update({"sensor": "camera", "max_range": 6.0})


@configclass
class G1CivicNavLidarStudentEnvCfg(G1CivicNavEnvCfg):
    """Civic student with the MID-360 lidar (people, height map) and the D435i color image (litter). The teacher's
    privileged state stays in the ``policy`` group for distillation."""

    scene: CivicStudentSceneCfg = CivicStudentSceneCfg(num_envs=2048, env_spacing=15.0)
    observations: CivicLidarStudentObservationsCfg = CivicLidarStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.d435i = None
        self.scene.mid360.update_period = self.decimation * self.sim.dt

    def play_mode(self):
        super().play_mode()
        self.observations.student.enable_corruption = True


@configclass
class G1CivicNavCameraStudentEnvCfg(G1CivicNavEnvCfg):
    """Camera-only civic student: the D435i (depth for people and the height map, color for litter), no lidar."""

    scene: CivicStudentSceneCfg = CivicStudentSceneCfg(num_envs=2048, env_spacing=15.0)
    observations: CivicCameraStudentObservationsCfg = CivicCameraStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.mid360 = None
        self.scene.d435i.update_period = self.decimation * self.sim.dt

    def play_mode(self):
        super().play_mode()
        self.observations.student.enable_corruption = True
