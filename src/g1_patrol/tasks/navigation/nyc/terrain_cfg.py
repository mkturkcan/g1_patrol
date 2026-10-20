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
    """One New York scenario type; the layout is built by ``tiles.BUILDERS[kind]``."""

    function: object = None
    kind: str = MISSING
    civic: bool = False
    """Add the civic-task features (litter baskets, trash-bag piles, queues, signalized crossings)."""


@configclass
class NycTerrainImporterCfg(TerrainImporterCfg):
    class_type: type | str = f"{_IMPL}:NycTerrainImporter"
    structures_visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.42, 0.42, 0.45), metallic=0.4, roughness=0.5
    )
    """Look of the static structures (walls, poles, fences, ...)."""


def nyc_terrain_generator_cfg(num_rows: int = 10, variants_per_kind: int = 2, civic: bool = False) -> TerrainGeneratorCfg:
    """Terrain with ``variants_per_kind`` columns per scenario type and ``num_rows`` difficulty levels."""
    sub_terrains = {kind: NycTileCfg(kind=kind, proportion=1.0, civic=civic) for kind in KINDS}
    return TerrainGeneratorCfg(
        class_type=f"{_IMPL}:NycTerrainGenerator",
        size=TILE_SIZE,
        border_width=10.0,
        num_rows=num_rows,
        num_cols=len(KINDS) * variants_per_kind,
        horizontal_scale=HF_RESOLUTION,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )
