"""B2-W camera student: the fused student (lidar and depth cameras) that also sees the front color camera's image,
through a frozen pretrained encoder (:func:`...mdp.camera_features`).

The camera sees the training environment drawn as in the demo (:mod:`.camera_look`: the demo's dressing on every
tile, animated people for the pedestrians); the capsules that the simulation moves and the lidar and depth cameras see
are not drawn. After training, the encoder is aligned to realistic images (image-to-image transfers of the student's
replays, which keep the scene's layout) so that realistic images give the features of their renders; the policy is
used unchanged with the aligned encoder.
"""

import math
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from g1_patrol.b2w import OPTICAL_CAMERAS, optical_frame_quat

from ... import mdp
from .nav_env_cfg import B2WFusedStudentSceneCfg, B2WLabFusedStudentEnvCfg, FusedStudentObservationsCfg

CAMERA_SIZE = (336, 196)
"""Width and height (pixels) of the camera image: multiples of the encoder's 14-pixel patches (24 x 14 patches)."""
CAMERA_HFOV = 90.0
"""Horizontal field of view (deg) assumed for the B2-W's front color camera."""


def front_rgb_cfg() -> TiledCameraCfg:
    """The front color camera at its URDF mount (f_oc_link), looking straight ahead."""
    pos, rpy = OPTICAL_CAMERAS["front"]
    aperture = 20.955
    width, height = CAMERA_SIZE
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_rgb",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=optical_frame_quat(rpy), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=aperture / (2 * math.tan(math.radians(CAMERA_HFOV) / 2)),
            horizontal_aperture=aperture,
            vertical_aperture=aperture * height / width,
            clipping_range=(0.05, 40.0),
        ),
        width=width,
        height=height,
    )


def spawn_hidden_pedestrian(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """The pedestrian's capsule (rigid body, collisions, ray-cast by the lidar and depth cameras) with its geometry not
    drawn: the camera sees the character that :mod:`.camera_look` puts in its place."""
    from .camera_look import spawn_hidden_capsule  # (imports USD, which may only load once the app runs)

    return spawn_hidden_capsule(prim_path, cfg, translation, orientation, **kwargs)


def spawn_demo_look(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """The demo's dressing of the whole terrain (:func:`.camera_look.spawn_demo_look`)."""
    from .camera_look import spawn_demo_look

    return spawn_demo_look(prim_path, cfg, translation, orientation, **kwargs)


@configclass
class DemoLookCfg(SpawnerCfg):
    func: Callable = spawn_demo_look


@configclass
class B2WCameraStudentSceneCfg(B2WFusedStudentSceneCfg):
    front_rgb = front_rgb_cfg()
    look = AssetBaseCfg(prim_path="/World/Demo", spawn=DemoLookCfg())
    """The demo's dressing (declared after the terrain, which it dresses)."""

    def __post_init__(self):
        self.light = None  # the dressing has its own sky and sun
        for ped in self.pedestrians.rigid_objects.values():
            ped.spawn = ped.spawn.replace(func=spawn_hidden_pedestrian)


@configclass
class CameraFeaturesCfg(ObsGroup):
    """Frozen-encoder features of the front color camera's image (class token and patch tokens on a 2 x 3 grid)."""

    camera = ObsTerm(
        func=mdp.camera_features,
        params={
            "sensor_name": "front_rgb",
            "grid": (2, 3),
            "before_render": "g1_patrol.tasks.navigation.config.b2w.camera_look:update",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CameraStudentObservationsCfg(FusedStudentObservationsCfg):
    camera: CameraFeaturesCfg = CameraFeaturesCfg()


@configclass
class B2WLabCameraStudentEnvCfg(B2WLabFusedStudentEnvCfg):
    """The lab fused student with the front color camera (run with ``--enable_cameras``)."""

    scene: B2WCameraStudentSceneCfg = B2WCameraStudentSceneCfg(num_envs=256, env_spacing=15.0)
    observations: CameraStudentObservationsCfg = CameraStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.front_rgb.update_period = self.decimation * self.sim.dt
        self.commands.nyc.debug_vis = False  # (the goal marker would be in every camera's view)
