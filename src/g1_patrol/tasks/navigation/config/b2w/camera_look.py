"""The training environment drawn as in the demo, for the students that see color images (rendering only; requires a
running Kit app with cameras enabled).

Every tile of the terrain is dressed as by the demo (``scripts/demo/nyc_look.py`` and ``b2w_look.py``: textured
ground and structures, SimReady props with their materials, facades, sky and sun; a light over each indoor room) when
the scene is created (:func:`spawn_demo_look`), and every pedestrian of every environment is drawn as an animated Isaac
Sim character (``scripts/demo/people.py``; :func:`update`). The characters of an environment are seen by that
environment's camera only (RTX scene partitions); the capsules that the simulation moves and the lidar and depth cameras
see are not drawn (:func:`spawn_hidden_capsule`), nor are the training meshes. The geometry the robot and its sensors
interact with is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from isaaclab.sim.spawners.shapes.shapes import spawn_capsule
from isaaclab.sim.utils.prims import clone

PROJECT = Path(__file__).resolve().parents[6]
DEMO = PROJECT / "scripts" / "demo"
CLIPS = PROJECT / "src" / "g1_patrol" / "assets" / "data" / "people"
INDOOR = ("makerlab", "civil_lab", "lab", "corridor", "doorway", "office", "warehouse", "lobby")
HALLS = ("makerlab", "civil_lab")
"""Indoor tiles that are sections of a hall continuing beyond the tile's ends (into the same column's other levels)."""
CEILING = {"makerlab": 5.5, "civil_lab": 5.5, "warehouse": 5.0}
"""Height (m) of the indoor tiles' ceilings (others: :data:`ROOM_CEILING`). The demo's rooms are roofless for its
cameras above the walls; seen from the robot, a roofless room shows the sky and the neighbouring tiles above its
walls, which the image-to-image transfer turns into odd ceilings."""
ROOM_CEILING = 3.2
ROOM_LIGHT = 1000.0
"""Intensity of the light under each indoor ceiling."""
CEILING_MATERIALS = {  # (diffuse rgb, roughness, metallic)
    "ceiling_hall": ((0.3, 0.31, 0.33), 0.75, 0.3),  # exposed steel deck
    "ceiling_room": ((0.8, 0.8, 0.78), 0.9, 0.0),  # acoustic tiles
}
CEILING_LIGHT = ((0.9, 0.9, 0.9), (1.0, 0.97, 0.9))  # light fixtures: (diffuse rgb, emissive rgb)


def _demo_modules():
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    import b2w_look
    import nyc_look
    import people

    return nyc_look, b2w_look, people


@clone
def spawn_hidden_capsule(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A capsule (:func:`isaaclab.sim.spawners.shapes.spawn_capsule`) whose geometry is not drawn (purpose ``guide``)."""
    from pxr import Usd, UsdGeom

    prim = spawn_capsule.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    for part in Usd.PrimRange(prim):
        if part.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(part).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    return prim


def spawn_demo_look(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Dress every tile of the terrain as the demo does, with a ceiling and a light under it over each indoor room
    (:func:`_ceiling`), a backdrop, a skyline, sky and sun, and hide the training meshes (spawned as a scene entity after the terrain, before the simulation
    starts; ``prim_path`` must be the dressing's root, ``/World/Demo``)."""
    import omni.usd

    from g1_patrol.tasks.navigation.nyc.terrain import NycTerrainGenerator

    nyc_look, b2w_look, _ = _demo_modules()
    assert prim_path == nyc_look.ROOT, f"the dressing is spawned at {nyc_look.ROOT}"
    stage = omni.usd.get_context().get_stage()
    generator = NycTerrainGenerator.latest
    tiles = sorted(generator.layouts)
    rng = np.random.default_rng(0)
    b2w_look.register_indoor_materials()
    nyc_look.COLORS.update(CEILING_MATERIALS)
    nyc_look.EMISSIVE["ceiling_light"] = CEILING_LIGHT
    nyc_look.create_materials()
    for r, c in tiles:
        lay, center = generator.layouts[(r, c)], nyc_look.tile_center(generator, r, c)
        nyc_look.dress_tile(stage, lay, center, f"r{r}_c{c}", rng, far_side=(r, c - 1) not in generator.layouts)
        if lay.kind in INDOOR:
            height = _ceiling(stage, nyc_look, f"{prim_path}/Tiles/r{r}_c{c}/ceiling", lay.kind, center)
            b2w_look.add_room_light(stage, f"{prim_path}/RoomLights/r{r}_c{c}", center, intensity=ROOM_LIGHT,
                                    height=height - 0.1)
    nyc_look.add_backdrop(stage, generator, set(tiles))
    centers = np.array([nyc_look.tile_center(generator, r, c) for r, c in tiles])
    inner = float(np.ptp(centers, axis=0).max()) / 2 + 35.0  # (the demo's ring is sized for a few filmed tiles)
    nyc_look.add_skyline(stage, centers.mean(axis=0), rng, inner=inner, outer=inner + 85.0, count=140)
    nyc_look.add_sky_and_sun(stage)
    nyc_look.hide_training_visuals(stage)
    print(f"[camera_look] dressed {len(tiles)} tiles", flush=True)
    return stage.GetPrimAtPath(prim_path)


def _ceiling(stage, nyc_look, root: str, kind: str, center) -> float:
    """A ceiling over an indoor tile, the walls continued up to it above the training walls (2.7 m; visual only: the
    lidar and depth cameras see the training meshes) and rows of light fixtures; returns the ceiling's height."""
    import trimesh

    from g1_patrol.tasks.navigation.nyc.indoor import IN_W, WALL_H, WALL_T
    from g1_patrol.tasks.navigation.nyc.tiles import HALF_L, HALF_W

    height = CEILING.get(kind, ROOM_CEILING)
    cx, cy = center
    up, mid = height - WALL_H + 0.02, (WALL_H + height) / 2
    walls = [nyc_look._box((cx, cy + side * (HALF_W - WALL_T / 2), mid), (2 * HALF_L, WALL_T, up)) for side in (-1, 1)]
    if kind not in HALLS:  # rooms are closed at their ends too; halls continue into the next tile
        walls += [nyc_look._box((cx + side * (HALF_L - WALL_T / 2), cy, mid), (WALL_T, 2 * IN_W, up)) for side in (-1, 1)]
    nyc_look.add_mesh(stage, f"{root}/walls", trimesh.util.concatenate(walls), "wall")
    slab = nyc_look._box((cx, cy, height + 0.1), (2 * HALF_L, 2 * HALF_W, 0.2))
    nyc_look.add_mesh(stage, f"{root}/slab", slab, "ceiling_hall" if kind in HALLS else "ceiling_room")
    rows = (-2.4, 0.0, 2.4) if kind in HALLS else (-1.6, 1.6)
    fixtures = [nyc_look._box((cx + x, cy + y, height - 0.03), (1.6, 0.18, 0.06))
                for y in rows for x in np.arange(-HALF_L + 1.6, HALF_L - 1.0, 3.2)]
    nyc_look.add_mesh(stage, f"{root}/lights", trimesh.util.concatenate(fixtures), "ceiling_light")
    return height


class AnimatedPeople:
    """Animated characters for the pedestrians of every environment (``people.AnimatedCrowd``), each environment's
    seen by its own camera only."""

    def __init__(self, env):
        import omni.usd
        from pxr import Sdf, UsdGeom

        nyc_look, _, people = _demo_modules()
        stage = omni.usd.get_context().get_stage()
        root = f"{nyc_look.ROOT}/People"
        scenario = env.command_manager.get_term("nyc")
        self.crowd = people.AnimatedCrowd(scenario, list(range(env.num_envs)), str(CLIPS), root=root)
        for e in range(env.num_envs):
            prim = stage.GetPrimAtPath(f"{root}/env_{e}")
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(f"env_{e}")
        self.step = None
        print(f"[camera_look] {len(self.crowd.people)} people", flush=True)

    def update(self, env):
        """Pose the people for the pedestrians' current state (walk cycles advance with the simulated time)."""
        steps = 0 if self.step is None else env.common_step_counter - self.step
        self.step = env.common_step_counter
        self.crowd.update(max(0, steps) * env.step_dt)


_PEOPLE: dict[int, AnimatedPeople] = {}


def update(env):
    """Create the environment's people on the first call; pose them before every rendered frame."""
    if id(env) not in _PEOPLE:
        _PEOPLE[id(env)] = AnimatedPeople(env)
    _PEOPLE[id(env)].update(env)
