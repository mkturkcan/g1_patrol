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


def test_people_offset_gives_other_meshes(tmp_path, monkeypatch):
    from g1_patrol.tasks.navigation.config.b2w import human_targets

    for k in range(10):
        for j in range(8):
            (tmp_path / f"character{k}_w{j / 16:.2f}.npz").touch()
    monkeypatch.setenv("G1_PATROL_PEOPLE_MESHES", str(tmp_path))
    slots = range(20)
    seen = {human_targets._mesh_file(i) for i in slots}
    monkeypatch.setenv("G1_PATROL_PEOPLE_OFFSET", "40")
    other = {human_targets._mesh_file(i) for i in slots}
    assert len(seen) == len(other) == 20 and not seen & other
    characters = lambda files: {f.rsplit("/", 1)[-1].split("_")[0] for f in files}  # noqa: E731
    assert characters(seen) == characters(other) == {f"character{k}" for k in range(10)}


def test_walking_characters_and_phase_order(tmp_path, monkeypatch):
    from g1_patrol.tasks.navigation.config.b2w import human_targets

    for k in range(10):
        for j in (7, 0, 1):  # created out of order
            (tmp_path / f"character{k}_w{j / 16:.2f}.npz").touch()
    monkeypatch.setenv("G1_PATROL_PEOPLE_MESHES", str(tmp_path))
    assert human_targets.walking_characters("train") == [f"character{k}" for k in range(8)]
    assert human_targets.walking_characters("held_out") == ["character8", "character9"]
    assert len(human_targets.walking_characters("all")) == 10
    poses = human_targets._poses_by_character()["character3"]
    assert [p.rsplit("_w", 1)[1] for p in poses] == ["0.00.npz", "0.06.npz", "0.44.npz"]
