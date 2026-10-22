"""Configuration for the New York street terrain (no simulator imports; see :mod:`.terrain`)."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .tiles import HF_RESOLUTION, KINDS, TILE_SIZE

STRUCTURES_PRIM_PATH = "/World/structures"
_IMPL = "g1_patrol.tasks.navigation.nyc.terrain"


@configclass
class NycTileCfg(SubTerrainBaseCfg):
    """One scenario type; the layout is built by ``indoor.ALL_BUILDERS[kind]`` (street and indoor kinds)."""

    function: object = None
    kind: str = MISSING
    civic: bool = False
    """Add the civic-task features (litter baskets, trash-bag piles, queues, signalized crossings)."""
    route_half_width: float | None = None
    """Keep only the routes that a robot this wide (half width, m) can drive (structures below ``route_overhead``),
    and rebuild the tile from a new seed if there are none; None: all routes (the G1)."""
    route_overhead: float = 1.0
    twin: str | None = None
    """Description (YAML) of the real place a "twin" tile reproduces (:mod:`.twin`)."""


@configclass
class NycTerrainImporterCfg(TerrainImporterCfg):
    class_type: type | str = f"{_IMPL}:NycTerrainImporter"
    structures_visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.42, 0.42, 0.45), metallic=0.4, roughness=0.5
    )
    """Look of the static structures (walls, poles, fences, ...)."""


def nyc_terrain_generator_cfg(
    num_rows: int = 10,
    variants_per_kind: int = 2,
    civic: bool = False,
    kinds: list[str] | None = None,
    route_half_width: float | None = None,
    twin: str | None = None,
) -> TerrainGeneratorCfg:
    """Terrain with ``variants_per_kind`` columns per scenario type (default: the street kinds; see
    :data:`.indoor.ALL_KINDS` for the indoor ones) and ``num_rows`` difficulty levels; with ``route_half_width``, only
    routes a robot that wide can drive."""
    kinds = KINDS if kinds is None else kinds
    sub_terrains = {
        kind: NycTileCfg(kind=kind, proportion=1.0, civic=civic, route_half_width=route_half_width, twin=twin)
        for kind in kinds
    }
    return TerrainGeneratorCfg(
        class_type=f"{_IMPL}:NycTerrainGenerator",
        size=TILE_SIZE,
        border_width=10.0,
        num_rows=num_rows,
        num_cols=len(kinds) * variants_per_kind,
        horizontal_scale=HF_RESOLUTION,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )
