"""Tests for the litter and trash-bag assets of the civic tasks (no simulator needed)."""

import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRASH_DIR = os.path.join(ROOT, "src", "g1_patrol", "assets", "data", "trash")
CIVIC_CFG = os.path.join(ROOT, "src", "g1_patrol", "tasks", "navigation", "config", "g1", "civic_env_cfg.py")
MAX_HAND_LOAD = 3.0
"""Heaviest load (kg) the reaching walker was trained to carry in one hand."""


def _scene_items() -> tuple[str, ...]:
    """``TRASH_ITEMS`` of the civic scene, read from the config's source (importing it needs the simulator)."""
    tree = ast.parse(open(CIVIC_CFG).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRASH_ITEMS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("TRASH_ITEMS not found")


def test_scene_items_have_assets():
    manifest = json.load(open(os.path.join(TRASH_DIR, "manifest.json")))
    items = _scene_items()
    assert len(items) == len(set(items))
    for name in items:
        entry = manifest[name]
        assert os.path.exists(os.path.join(TRASH_DIR, entry["file"])), name
        assert all(s > 0.0 for s in entry["size"]) and entry["mass"] > 0.0, name
    bags = [name for name in items if manifest[name].get("kind") == "bag"]
    assert bags and len(bags) < len(items)
    assert all(manifest[name]["mass"] <= MAX_HAND_LOAD for name in bags)
    assert all(0.25 <= max(manifest[name]["size"]) <= 0.6 for name in bags)


def test_imported_items_are_credited():
    manifest = json.load(open(os.path.join(TRASH_DIR, "manifest.json")))
    credits = open(os.path.join(TRASH_DIR, "ATTRIBUTION.md")).read()
    imported = [name for name, entry in manifest.items() if entry["file"].endswith(".usdc")]
    assert imported
    for name in imported:
        entry = manifest[name]
        assert entry["license"] in ("CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"), name
        assert f"`{name}`" in credits and entry["source"] in credits, name
