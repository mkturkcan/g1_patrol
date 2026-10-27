"""Tests for the B2-W pedestrians seen as posed characters (``config/b2w/human_targets.py``)."""

import gymnasium as gym
import pytest

import g1_patrol.tasks  # noqa: F401

SENSORS = ("lidar", "front_depth", "rear_depth")


@pytest.mark.parametrize("student", ["Fused", "Lidar"])
def test_people_task_registrations(student):
    spec = gym.spec(f"G1Patrol-B2W-Navigate-Lab-{student}-People")
    assert spec.kwargs["env_cfg_entry_point"] == (
        f"g1_patrol.tasks.navigation.config.b2w.human_targets:B2WLab{student}StudentPeopleEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(f":B2WLab{student}DistillationRunnerCfg")


def _targets(cfg):
    return {name: [getattr(t, "prim_expr", t) for t in getattr(cfg.scene, name).mesh_prim_paths]
            for name in SENSORS if getattr(cfg.scene, name, None) is not None}


def test_people_meshes_replace_only_the_pedestrian_targets():
    from g1_patrol.tasks.navigation.config.b2w import human_targets
    from g1_patrol.tasks.navigation.config.b2w.nav_env_cfg import B2WLabFusedStudentEnvCfg

    capsules, people = B2WLabFusedStudentEnvCfg(), human_targets.B2WLabFusedStudentPeopleEnvCfg()
    before, after = _targets(capsules), _targets(people)
    assert set(before) == set(SENSORS) and set(after) == set(SENSORS)
    for name in SENSORS:
        peds = [b for b in before[name] if b.rsplit("/", 1)[-1].startswith("Ped_")]
        assert peds, f"{name} casts no rays at pedestrians"
        assert after[name] == [f"{b}/person" if b in peds else b for b in before[name]]
    for ped in people.scene.pedestrians.rigid_objects.values():
        assert ped.spawn.func is human_targets.spawn_capsule_with_person


def test_missing_people_meshes_are_reported(tmp_path, monkeypatch):
    from g1_patrol.tasks.navigation.config.b2w import human_targets

    monkeypatch.setenv("G1_PATROL_PEOPLE_MESHES", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="bake_people_meshes"):
        human_targets._mesh_file(0)
