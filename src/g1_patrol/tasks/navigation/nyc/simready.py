"""NVIDIA SimReady props (CC-BY 4.0) in the tiles: what is used, where the converted meshes are, and their bounds.

The props are downloaded and converted by ``scripts/assets/fetch_simready.py`` into ``$G1_PATROL_SIMREADY``
(``~/.cache/g1_patrol/simready``); the repository holds only their bounds (``simready_catalog.json``), from which the
tiles place them. A tile refers to a prop by name (:meth:`.tiles.TileLayout.prop`): its bounding box is the collision
primitive and the sensors see the prop's mesh.
"""

import functools
import json
import os

import numpy as np

DATASET = "nvidia/simready-assets"
REVISION = "524747d9b5e1f21b5b18b56041788c01d745f1a1"
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simready_catalog.json")

PROPS = [
    # benches, desks, seats
    "furniture_workbench_modular_a01",
    "furniture_workbench_modular_b01",
    "furniture_workbench_standard_d01",
    "furniture_workbench_standard_e01",
    "desk_a01",
    "chair_office_amf_black_a01",
    "chair_office_fabric_manager_a01",
    "stool_mobile_padded_vinyl_a01",
    # storage
    "storage_shelf_basic_a01",
    "storage_shelf_basic_b01",
    "storage_shelf_basic_c01",
    "rack_bulk_storage_a03",
    "rack_composition_a04",
    "rack_composition_a05",
    "rack_composition_a06",
    "storage_locker_double_bench_b01",
    "storage_toolbox_carry_hardCase_b01",
    "storage_toolbox_standard_hardCase_d01",
    "cabinet_file_a01",
    # carts and trolleys
    "cart_mobile_a01",
    "cart_mobile_b01",
    "cart_mobile_c01",
    "cart_mobile_d01",
    "cart_mobile_e01",
    "cart_mobile_f01",
    "cart_computer_a01",
    "cart_luggage_platform_a01",
    "equipment_tools_trolley_a01",
    "handTruck_convertible_steel_orange_a01",
    "handTruck_convertible_steel_red_a01",
    "vehicleSupport_rolling_dolly_a01",
    # pallets, boxes, crates
    "pallet_a01",
    "box_cardboard_a01",
    "box_cardboard_b02",
    "box_cardboard_c03",
    "crate_blue_a01",
    "crate_black_a02",
    "crate_gray_a08",
    # drums, pails, bins
    "drum_plastic_blue_a01",
    "drum_steel_a01",
    "pail_plastic_green_a02",
    "pail_plastic_red_a02",
    "pail_closedHead_blue_a01",
    "trashCan_office_gray_a03",
    "trashCan_thin_gray_a03",
    "trashCan_wheeled_green_a01",
    "bin_waste_a01",
    # ladders, scaffold
    "ladder_step_aluminium_a06",
    "ladder_extention_fiberglass_a03",
    "ladder_rolling_yellow_a02",
    "scaffold_rolling_steel_a01",
    # computers
    "monitor_a01",
    "computer_monitor_office_a01",
    "computer_desktop_tower_a01",
    # equipment, safety
    "equipment_3d_printer_a01",
    "equipment_airCompressor_a01",
    "equipment_liftTable_scissorMobile_a01",
    "equipment_bench_vise_a01",
    "safety_eyewashStation_stand_selfContained_a01",
    "fireExtinguisher_stand_a01",
    "lighting_taskLight_tripodFlood_a01",
    "machineGuard_a01",
    "structure_wall_column_a01",
    "ppe_helmet_hardHat_a01",
]


def default_root() -> str:
    return os.environ.get("G1_PATROL_SIMREADY", os.path.expanduser("~/.cache/g1_patrol/simready"))


def root_usd(name: str, root: str | None = None) -> str:
    """The prop's USD file (rendering)."""
    return os.path.join(root or default_root(), "nvidia_simready_assets", name, "simready_usd", f"sm_{name}_01.usd")


@functools.cache
def catalog() -> dict:
    """Bounds (``min``, ``max``: corners of the axis-aligned box in the prop's frame [m]) per prop."""
    with open(CATALOG_PATH) as fh:
        return json.load(fh)


@functools.cache
def prop_mesh(name: str):
    """The prop's mesh (trimesh, prop frame) as the sensors see it; its bounding box if the prop is not downloaded and
    ``G1_PATROL_SIMREADY_BOXES=1`` (tests)."""
    import trimesh

    path = os.path.join(default_root(), "processed", f"{name}.npz")
    if not os.path.exists(path):
        if os.environ.get("G1_PATROL_SIMREADY_BOXES") != "1":
            raise FileNotFoundError(f"{path} is missing: run scripts/assets/fetch_simready.py")
        lo, hi = (np.asarray(catalog()[name][k]) for k in ("min", "max"))
        box = trimesh.creation.box(extents=hi - lo)
        box.apply_translation((lo + hi) / 2)
        return box
    data = np.load(path)
    return trimesh.Trimesh(data["vertices"], data["faces"], process=False)
