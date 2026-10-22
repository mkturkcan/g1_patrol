"""Terrain generator and importer for New York street tiles (implementation).

Loaded by name from :mod:`.terrain_cfg` when the terrain is instantiated, i.e. after the simulator has started.

The generator builds one :mod:`.tiles` layout per terrain tile (columns: scenario types, rows: difficulty) and keeps
two meshes: the walkable ground (imported as the usual terrain, the only thing the walker's height scan sees) and
the static structures (imported separately at ``STRUCTURES_PRIM_PATH``). The importer also exposes the tile layouts
as padded GPU tensors for the scenario command.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator, TerrainGeneratorCfg, TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh

from .indoor import passable_routes
from .kinds import ALL_BUILDERS, ALL_KINDS
from .terrain_cfg import STRUCTURES_PRIM_PATH, NycTileCfg
from .twin import build_twin, load_twin
from .tiles import (
    HF_RESOLUTION,
    MAX_BASKETS,
    MAX_BOXES,
    MAX_CIRCLES,
    MAX_CROSSINGS,
    MAX_FORBIDDEN,
    MAX_PATHS,
    MAX_PILES,
    MAX_QUEUES,
    MAX_ROUTES,
    MAX_STANDING,
    PATH_ROW,
    TILE_SIZE,
    TileLayout,
)


class NycTerrainGenerator(TerrainGenerator):
    """Terrain generator that also collects per-tile structure meshes and scenario layouts."""

    latest: NycTerrainGenerator | None = None
    """The most recently created generator (read by :class:`NycTerrainImporter`)."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        self.layouts: dict[tuple[int, int], TileLayout] = {}
        self._structures: list[trimesh.Trimesh] = []
        self._pending: TileLayout | None = None
        super().__init__(cfg, device)
        # the base class shifts the terrain so that it is centered on the world origin; shift the structures too
        self.shift = np.array([-cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5, 0.0])
        self.structure_mesh = trimesh.util.concatenate(self._structures) if self._structures else None
        if self.structure_mesh is not None:
            self.structure_mesh.apply_translation(self.shift)
        NycTerrainGenerator.latest = self

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        if not isinstance(cfg, NycTileCfg):
            return super()._get_terrain_mesh(difficulty, cfg)
        seed = int(self.np_rng.integers(1 << 31))
        rng = np.random.default_rng(seed)
        # civic additions draw from their own stream, so that the rest of the tile is the same with or without them
        civic_rng = np.random.default_rng([seed, 1]) if cfg.civic else None
        build = (lambda d, r, _: build_twin(load_twin(cfg.twin), d, r)) if cfg.kind == "twin" else ALL_BUILDERS[cfg.kind]
        self._pending = build(float(difficulty), rng, civic_rng)
        if cfg.route_half_width is not None:  # only routes this robot can drive; rebuild tiles that have none
            for attempt in range(1, 20):
                keep = passable_routes(self._pending, cfg.route_half_width, cfg.route_overhead)
                if keep:
                    lay = self._pending
                    lay.routes = [lay.routes[k] for k in keep]
                    lay.route_headings = [lay.route_headings[k] for k in keep] if lay.route_headings else []
                    break
                rng = np.random.default_rng([seed, 2, attempt])
                civic_rng = np.random.default_rng([seed, 1, attempt]) if cfg.civic else None
                self._pending = build(float(difficulty), rng, civic_rng)
        return self._pending.ground_mesh(), np.zeros(3)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if self._pending is None:
            return
        offset = ((row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1], 0.0)
        structures = self._pending.structure_mesh()
        if structures is not None:
            structures.apply_translation(offset)
            self._structures.append(structures)
        self.layouts[(row, col)] = self._pending
        self._pending = None


def _pad_nan(items, cap):
    out = np.full((cap, 1), np.nan, dtype=np.float32)
    n = min(len(items), cap)
    if n:
        out[:n] = np.asarray(items[:n], dtype=np.float32)
    return out, n


class NycLayout:
    """Tile layouts as padded tensors indexed by (row, col). Positions are tile-local (relative to the tile center)."""

    def __init__(self, generator: NycTerrainGenerator, tile_centers: torch.Tensor, device: str):
        rows, cols = generator.cfg.num_rows, generator.cfg.num_cols
        self.num_rows, self.num_cols = rows, cols
        self.tile_centers = tile_centers[..., :2].to(device)  # (rows, cols, 2) world xy

        def pad(items, width, cap):
            out = np.zeros((cap, width), dtype=np.float32)
            n = min(len(items), cap)
            if n:
                out[:n] = np.asarray(items[:n], dtype=np.float32)
            return out, n

        fields = {
            "circles": (5, MAX_CIRCLES),
            "boxes": (7, MAX_BOXES),
            "routes": (4, MAX_ROUTES),
            "paths": (PATH_ROW, MAX_PATHS),
            "standing": (2, MAX_STANDING),
            "forbidden": (4, MAX_FORBIDDEN),
            "baskets": (4, MAX_BASKETS),
            "piles": (4, MAX_PILES),
            "queues": (5, MAX_QUEUES),
            "crossings": (4, MAX_CROSSINGS),
            "route_headings": (1, MAX_ROUTES),
        }
        data = {name: np.zeros((rows, cols, cap, width), dtype=np.float32) for name, (width, cap) in fields.items()}
        counts = {name: np.zeros((rows, cols), dtype=np.int64) for name in fields}
        sample = next(iter(generator.layouts.values()))
        heights = np.zeros((rows, cols, *sample.heights.shape), dtype=np.float32)
        crowd = np.zeros((rows, cols, 2), dtype=np.float32)
        kind = np.zeros((rows, cols), dtype=np.int64)
        difficulty = np.zeros((rows, cols), dtype=np.float32)
        for (r, c), lay in generator.layouts.items():
            for name, (width, cap) in fields.items():
                if name == "route_headings":  # NaN: start facing the goal
                    heads = list(lay.route_headings) + [float("nan")] * (len(lay.routes) - len(lay.route_headings))
                    data[name][r, c], counts[name][r, c] = _pad_nan([[h] for h in heads], cap)
                    continue
                items = lay.path_rows() if name == "paths" else lay.baskets() if name == "baskets" else getattr(lay, name)
                data[name][r, c], counts[name][r, c] = pad(items, width, cap)
            heights[r, c] = lay.heights
            crowd[r, c] = lay.crowd
            kind[r, c] = ALL_KINDS.index(lay.kind)
            difficulty[r, c] = lay.difficulty
        for name in fields:
            setattr(self, name, torch.from_numpy(data[name]).to(device))
            setattr(self, f"num_{name}", torch.from_numpy(counts[name]).to(device))
        self.heights = torch.from_numpy(heights).to(device)
        self.crowd = torch.from_numpy(crowd).to(device)
        self.kind = torch.from_numpy(kind).to(device)
        self.difficulty = torch.from_numpy(difficulty).to(device)

    def ground_height(self, rows: torch.Tensor, cols: torch.Tensor, xy_local: torch.Tensor) -> torch.Tensor:
        """Ground height at tile-local points. xy_local: (n, ..., 2) with rows/cols (n,)."""
        nx, ny = self.heights.shape[2] - 1, self.heights.shape[3] - 1
        i = ((xy_local[..., 0] + TILE_SIZE[0] / 2) / HF_RESOLUTION).round().long().clamp(0, nx)
        j = ((xy_local[..., 1] + TILE_SIZE[1] / 2) / HF_RESOLUTION).round().long().clamp(0, ny)
        shape = [-1] + [1] * (xy_local.dim() - 2)
        return self.heights[rows.view(shape), cols.view(shape), i, j]


class NycTerrainImporter(TerrainImporter):
    """Terrain importer that additionally imports the static structures and exposes the tile layouts."""

    def __init__(self, cfg):
        super().__init__(cfg)
        generator = NycTerrainGenerator.latest
        if generator is None:
            raise RuntimeError("NycTerrainImporter requires a terrain generator of type NycTerrainGenerator.")
        if generator.structure_mesh is not None:
            create_prim_from_mesh(
                STRUCTURES_PRIM_PATH,
                generator.structure_mesh,
                visual_material=self.cfg.structures_visual_material,
                physics_material=self.cfg.physics_material,
            )
        self.layout = NycLayout(generator, self.terrain_origins, self.device)
