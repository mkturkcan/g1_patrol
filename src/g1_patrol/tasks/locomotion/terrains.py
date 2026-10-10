"""Terrain mixes for the locomotion tasks."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

STREET_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # single curbs: step up out of a pit / step down off a platform (sidewalk curbs are typically 10-20 cm)
        "curb_up": terrain_gen.MeshPitTerrainCfg(proportion=0.15, pit_depth_range=(0.05, 0.22), platform_width=3.0),
        "curb_down": terrain_gen.MeshBoxTerrainCfg(proportion=0.15, box_height_range=(0.05, 0.22), platform_width=3.0),
        # raised strips to step over (curb-to-curb crossings, kerb ramps, door sills)
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1, rail_thickness_range=(0.1, 0.3), rail_height_range=(0.05, 0.18), platform_width=3.0
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.2), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.2), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        # broken pavement, cobblestones, grass
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        # ramps and cambered streets
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Street-like terrain: curbs, sills, stairs, rough pavement and ramps, with difficulty increasing by row."""

CITY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        # sidewalk curbs and building steps
        "curb_up": terrain_gen.MeshPitTerrainCfg(proportion=0.11, pit_depth_range=(0.05, 0.20), platform_width=3.0),
        "curb_down": terrain_gen.MeshBoxTerrainCfg(proportion=0.11, box_height_range=(0.05, 0.20), platform_width=3.0),
        # door sills, hoses, cable protectors, kerb strips
        "sills": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.07, rail_thickness_range=(0.08, 0.3), rail_height_range=(0.03, 0.15), platform_width=3.0
        ),
        # potholes and bumps: cuboids sunk into / raised from the pavement
        "potholes": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.14,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.04, 0.14),
            num_obstacles=30,
            platform_width=1.5,
            border_width=0.25,
        ),
        # broken pavement and cobblestones
        "uneven": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.12, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "cobbles": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.08, grid_width=0.3, grid_height_range=(0.01, 0.06), platform_width=2.0
        ),
        # subway and stoop stairs
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.08, step_height_range=(0.05, 0.18), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.08, step_height_range=(0.05, 0.18), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        # curb cuts, driveways, cambered streets
        "ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        "ramp_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
    },
)
"""New York street surfaces: curbs, sills, potholes and bumps, broken pavement, cobblestones, stairs and curb cuts."""
