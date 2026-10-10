"""Crowd navigation for the Unitree G1: reach a goal 4-8 m away through walking pedestrians, poles and walls.

The navigation policy runs at 10 Hz and outputs (forward, lateral, yaw-rate) commands to the frozen street-walking
policy (``G1Patrol-Walk-Street``), which keeps the robot balanced at 50 Hz. It perceives the world through a
360-degree, single-plane lidar on the torso (360 rays, 1 degree apart, 6 m range, reduced to the nearest return in each
of 72 sectors; last 3 scans) plus the goal in its own frame.

Pedestrians, poles and walls are kinematic bodies laid out anew on every reset by the ``crowd`` command term.
They are invisible to the walker's terrain height scan, so obstacle avoidance is entirely the navigator's job.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg

from g1_patrol.assets import G1_PATROL_ASSETS_DIR
from g1_patrol.tasks.locomotion.config.g1.walk_env_cfg import G1WalkStreetEnvCfg

from ... import mdp

WALKER_ENV_CFG = G1WalkStreetEnvCfg()
"""Environment of the frozen walking policy; its robot, sensors, physics, observations and actions are reused."""

WALKER_POLICY_PATH = str(G1_PATROL_ASSETS_DIR / "policies" / "g1_walk_street.pt")
"""TorchScript export of the street walker (``isaaclab play`` writes it to ``<run>/exported/policy.pt``)."""

NUM_PEDESTRIANS = 10
NUM_POLES = 8
NUM_WALLS = 4
LIDAR_MAX_DISTANCE = 6.0
LIDAR_SECTORS = 72

SCENARIO = mdp.CrowdScenarioCommandCfg(
    asset_name="robot", pedestrians_name="pedestrians", poles_name="poles", walls_name="walls", debug_vis=True
)


def _kinematic_bodies(name: str, count: int, shape: sim_utils.SpawnerCfg) -> RigidObjectCollectionCfg:
    """``count`` kinematic copies of ``shape``, parked below the ground until the scenario places them."""
    shape.rigid_props = [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
    shape.collision_props = sim_utils.UsdPhysicsCollisionCfg()
    shape.mass_props = sim_utils.MassCfg(mass=70.0)
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"{name.lower()}_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}_{i}",
                spawn=shape,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for i in range(count)
        }
    )


@configclass
class CrowdSceneCfg(MySceneCfg):
    pedestrians = _kinematic_bodies(
        "Ped",
        NUM_PEDESTRIANS,
        sim_utils.CapsuleCfg(
            radius=SCENARIO.pedestrian_radius,
            height=1.2,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.1)),
        ),
    )
    poles = _kinematic_bodies(
        "Pole",
        NUM_POLES,
        sim_utils.CylinderCfg(
            radius=SCENARIO.pole_radius,
            height=SCENARIO.pole_height,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
    )
    walls = _kinematic_bodies(
        "Wall",
        NUM_WALLS,
        sim_utils.CuboidCfg(
            size=SCENARIO.wall_size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.3))
        ),
    )
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        # one target per body: cloned environments are matched per asset
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/{name}_{i}")
            for name, count in (("Ped", NUM_PEDESTRIANS), ("Pole", NUM_POLES), ("Wall", NUM_WALLS))
            for i in range(count)
        ],
        # "base": the scan rotates with the torso like a mounted lidar ("yaw" would rotate only the ray origins and
        # keep the ray directions fixed in the world, hiding where obstacles are relative to the robot's heading)
        ray_alignment="base",
        max_distance=LIDAR_MAX_DISTANCE,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.0
        ),
    )


@configclass
class ActionsCfg:
    walker = mdp.WalkerPolicyActionCfg(
        asset_name="robot",
        policy_path=WALKER_POLICY_PATH,
        low_level_decimation=WALKER_ENV_CFG.decimation,
        low_level_actions=WALKER_ENV_CFG.actions.joint_pos,
        low_level_observations=WALKER_ENV_CFG.observations.policy,
    )
    # moves the pedestrians at every physics step; takes no policy inputs
    crowd = mdp.CrowdMotionActionCfg(asset_name="pedestrians", scenario_name="crowd")


@configclass
class CommandsCfg:
    crowd = SCENARIO


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "crowd"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        lidar = ObsTerm(
            func=mdp.lidar_distances,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": LIDAR_MAX_DISTANCE, "sectors": LIDAR_SECTORS},
            history_length=3,
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "walker"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # reward terms are scaled by the step length (0.1 s): weight 2 on progress pays 2 per meter gained,
    # weights of +-100 on one-step events pay +-10
    progress = RewTerm(func=mdp.goal_progress, weight=2.0, params={"command_name": "crowd"})
    success = RewTerm(func=mdp.goal_reached, weight=100.0, params={"command_name": "crowd"})
    collision = RewTerm(func=mdp.is_terminated_term, weight=-100.0, params={"term_keys": ".*_collision"})
    fall = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "fall"})
    pedestrian_proximity = RewTerm(
        func=mdp.pedestrian_proximity, weight=-4.0, params={"command_name": "crowd", "comfort_distance": 0.5}
    )
    obstacle_proximity = RewTerm(
        func=mdp.obstacle_proximity, weight=-4.0, params={"command_name": "crowd", "comfort_distance": 0.2}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached_termination, params={"command_name": "crowd"})
    pedestrian_collision = DoneTerm(func=mdp.pedestrian_collision, params={"command_name": "crowd"})
    obstacle_collision = DoneTerm(func=mdp.obstacle_collision, params={"command_name": "crowd"})
    fall = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class G1CrowdNavEnvCfg(ManagerBasedRLEnvCfg):
    scene: CrowdSceneCfg = CrowdSceneCfg(num_envs=4096, env_spacing=15.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        walker = WALKER_ENV_CFG
        # timing: walker at 50 Hz, navigator at 10 Hz
        self.sim = walker.sim
        self.decimation = walker.decimation * 5
        self.episode_length_s = 30.0
        # scene: the walker's robot and sensors, on flat ground
        self.scene.robot = walker.scene.robot
        self.scene.height_scanner.prim_path = walker.scene.height_scanner.prim_path
        self.scene.height_scanner.update_period = walker.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.sim.physics_material = self.scene.terrain.physics_material
        # events: the walker's randomization, with an occasional shove while navigating
        self.events = walker.events
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        # camera (headless video) follows robot 0 from higher up, to show the crowd around it
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (4.0, 4.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

    def play_mode(self):
        super().play_mode()
        self.events.push_robot = None
        self.events.base_external_force_torque = None
