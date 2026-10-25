"""Tests for the velocity safety shield (no simulator needed)."""

import numpy as np
import pytest
import torch

from g1_patrol.b2w import CONTROLLER
from g1_patrol.controller_model import track
from g1_patrol.shield import ShieldCfg, map_points, moving_returns, shield

CFG = ShieldCfg()
FLOOR = CFG.ground_z
FRONT = CFG.half_length


def _column(x, y, top, radius=0.15, n=200, seed=0):
    """Returns on a vertical column (a person when tall, a box when low) centred at (x, y), base frame."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    z = FLOOR + rng.uniform(0.1, top, n)
    return np.c_[x + radius * np.cos(a), y + radius * np.sin(a), z]


def _wall_across(x, y0, y1, top=2.5, n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[np.full(n, x), rng.uniform(y0, y1, n), FLOOR + rng.uniform(0.1, top, n)]


def _run(points, command, velocity=None, moving=None, person=None):
    pts = torch.tensor(np.asarray(points), dtype=torch.float32)[None]
    mask = lambda m: None if m is None else torch.tensor(m, dtype=torch.bool)[None]  # noqa: E731
    vel = None if velocity is None else torch.tensor([velocity], dtype=torch.float32)
    safe, info = shield(pts, torch.tensor([command], dtype=torch.float32), velocity=vel, moving=mask(moving), person=mask(person))
    return safe[0].numpy(), {k: float(v[0]) for k, v in info.items()}


def test_person_ahead_slows_the_approach():
    person = _column(FRONT + 0.15 + 1.0, 0.0, 1.7)  # 1 m ahead of the front
    safe, info = _run(person, [1.5, 0.0, 0.0])
    assert info["min_tall_gap"] == pytest.approx(1.0, abs=0.03)
    assert 0.0 < safe[0] <= 0.5  # closing in on a tall thing: slowly


def test_close_person_is_approached_only_at_a_creep_and_not_at_all_when_detected():
    close = _column(FRONT + 0.15 + 0.4, 0.0, 1.7)
    safe, _ = _run(close, [1.0, 0.0, 0.0])
    assert safe[0] <= CFG.approach_speed + 1e-6  # a tall thing (could be a wall): creep
    safe, _ = _run(close, [1.0, 0.0, 0.0], person=np.ones(len(close), bool))
    assert safe[0] == pytest.approx(0.0, abs=1e-6)  # a detected person within the person margin: no approach
    safe, _ = _run(close, [-0.5, 0.0, 0.0], person=np.ones(len(close), bool))
    assert safe[0] == pytest.approx(-0.5)  # backing away is not limited


def test_low_box_limits_speed_to_stopping_distance():
    box = _column(FRONT + 0.1 + 0.3, 0.0, 0.3, radius=0.1)  # 0.3 m ahead, 0.3 m tall
    safe, _ = _run(box, [1.5, 0.0, 0.0])
    assert 0.0 < safe[0] < 0.4


def test_wall_alongside_allows_driving_on_but_not_turning_into_it():
    xs = np.linspace(-2, 3, 500)
    wall = np.c_[xs, np.full_like(xs, -(CFG.half_width + 0.25)), np.full_like(xs, FLOOR + 0.8)]  # 0.25 m to the right
    safe, _ = _run(wall, [1.2, 0.0, 0.0])
    assert safe[0] > 0.8  # (the drift allowance takes some speed)
    safe, _ = _run(wall, [0.0, 0.0, -1.2])  # turning right swings the front-right corner into the wall
    assert abs(safe[2]) < 1.2


def test_robot_fits_through_a_door_slowly():
    door = np.concatenate([_wall_across(1.0, -3.0, -0.45, seed=1), _wall_across(1.0, 0.45, 3.0, seed=2)])  # 0.9 m door
    safe, _ = _run(door, [0.6, 0.0, 0.0])
    assert safe[0] > 0.2
    inside = np.concatenate([_wall_across(0.0, -3.0, -0.45, seed=1), _wall_across(0.0, 0.45, 3.0, seed=2)])  # in the door
    safe, _ = _run(inside, [0.6, 0.0, 0.0])
    assert 0.05 < safe[0] <= CFG.slow_speed + 0.05


def test_floor_ceiling_and_own_body_are_ignored():
    floor = np.c_[np.random.default_rng(0).uniform(-3, 3, (500, 2)), np.full(500, FLOOR + 0.02)]
    ceiling = np.c_[np.random.default_rng(1).uniform(-3, 3, (500, 2)), np.full(500, FLOOR + 2.6)]
    body = np.c_[np.random.default_rng(2).uniform(-0.4, 0.4, (200, 1)), np.zeros((200, 1)), np.full((200, 1), 0.05)]
    safe, info = _run(np.concatenate([floor, ceiling, body]), [1.5, 0.2, 0.8])
    assert np.allclose(safe, [1.5, 0.2, 0.8]) and info["scale"] == 1.0


def test_drop_ahead_is_kept_clear():
    rng = np.random.default_rng(3)
    pit = np.c_[rng.uniform(0.9, 1.5, (300, 1)), rng.uniform(-0.5, 0.5, (300, 1)), np.full((300, 1), FLOOR - 1.0)]
    safe, _ = _run(pit, [1.5, 0.0, 0.0])
    assert safe[0] < 1.0


def test_map_points_are_obstacle_cells():
    m = torch.full((1, 80, 80), -1.0)
    m[0, 50, 40] = 0.5  # 1 m ahead
    pts = map_points(m, 0.1, FLOOR)
    finite = torch.isfinite(pts).all(dim=-1)
    assert int(finite.sum()) == 1
    assert torch.allclose(pts[finite][0], torch.tensor([1.05, 0.05, FLOOR + 0.5]))


def test_moving_returns_are_where_the_floor_was():
    m = torch.full((1, 3, 80, 80), 0.0)  # floor everywhere, seen by both scans
    m[0, 1, 60, 40] = 1.6  # something tall in the current scan where the previous one saw floor
    m[0, 1:, 50, 30] = 1.6  # something tall in both scans: static
    pts = torch.tensor([[[2.05, 0.05, FLOOR + 1.5], [1.05, -0.95, FLOOR + 1.5]]])
    assert moving_returns(pts, m, 0.1, FLOOR).tolist() == [[True, False]]


def test_flickering_edges_and_stray_cells_are_not_moving():
    """A cell at the edge of something static that held something tall a moment ago does not count as moving, nor
    does a lone cell; several cells of open floor filled at once (a person walking in) do."""
    m = torch.full((1, 3, 80, 80), 0.0)
    m[0, 1, 60, 38:41] = 1.6  # three cells in a row appeared in front of the robot (a person's front)
    m[0, 1, 50, 30] = 1.6  # a lone cell appeared
    m[0, 1, 30, 40] = 1.6  # the edge of a shelf, tall half a second ago
    person = [[2.05, y, FLOOR + 1.5] for y in (-0.15, -0.05, 0.05)]
    pts = torch.tensor([person + [[1.05, -0.95, FLOOR + 1.5], [-0.95, 0.05, FLOOR + 1.5]]])
    tall_age = torch.full((1, 80, 80), 1e3)
    tall_age[0, 30, 40] = 0.5
    assert moving_returns(pts, m, 0.1, FLOOR).tolist() == [[True] * 5]
    assert moving_returns(pts, m, 0.1, FLOOR, tall_age=tall_age, recent=1.0).tolist() == [[True] * 4 + [False]]
    assert moving_returns(pts, m, 0.1, FLOOR, tall_age=tall_age, recent=1.0, min_cells=2).tolist() == [[True] * 3 + [False] * 2]


def test_height_map_remembers_when_cells_were_tall():
    from g1_patrol.heightmap import HeightMap

    hm = HeightMap(1, 8.0, 0.1)
    root, yaw, ground = torch.tensor([[0.0, 0.0, 0.55]]), torch.zeros(1), torch.zeros(1)
    post = torch.tensor([[[1.05, 0.05, 1.0]]])  # something 1 m tall, 1 m ahead
    floor = torch.tensor([[[1.05, 0.05, 0.0]]])
    hm.update(post, root, yaw, ground, 0.1)
    for _ in range(3):
        hm.update(floor, root, yaw, ground, 0.1)
    # before the latest scan, the cell last held something tall 0.2 s earlier
    assert float(hm.tall_age_out[0, 50, 40]) == pytest.approx(0.2, abs=1e-5)
    assert float(hm.tall_age_out[0, 10, 10]) > 100.0  # never


def _controller():
    dt = 0.02
    worst = {"delay": int(round(CONTROLLER["delay_range"][1] / dt)),
             "tau": torch.tensor([[r[1] for r in CONTROLLER["tau_range"]]]),
             "accel": torch.tensor([[r[0] for r in CONTROLLER["accel_range"]]])}
    return dt, worst


@pytest.mark.parametrize("top,person,expected", [(1.7, True, CFG.person_margin), (1.7, False, CFG.margin), (0.4, False, CFG.margin)])
def test_closed_loop_never_breaches_the_margin(top, person, expected):
    """Driving at full speed at a person (detected, or just a tall column) or a box through the controller model with
    its slowest response and weakest braking in the training ranges, the shield in the loop at 10 Hz: the gap stays
    above the margin, and the robot does get there."""
    dt, worst = _controller()
    gain, deadband, drift = torch.tensor([[1.1, 1.0, 1.0]]), torch.zeros(1, 3), torch.zeros(1)
    x_obstacle, x, vel = 6.0, 0.0, torch.zeros(1, 3)
    queue, min_gap = [torch.zeros(1, 3)] * (worst["delay"] + 1), 1e3
    for k in range(300):
        pts = torch.tensor(_column(x_obstacle - x, 0.0, top, seed=k), dtype=torch.float32)[None]
        mask = torch.ones(pts.shape[:2], dtype=torch.bool) if person else None
        safe, _ = shield(pts, torch.tensor([[1.5, 0.0, 0.0]]), velocity=vel, person=mask)
        for _ in range(5):
            queue = [safe] + queue[:-1]
            vel = track(vel, queue[-1], dt, gain, deadband, worst["tau"], worst["accel"], drift)
            x += float(vel[0, 0]) * dt
            min_gap = min(min_gap, x_obstacle - 0.15 - x - FRONT)
    assert min_gap > expected - 0.05, f"gap {min_gap:.2f} m < margin {expected} m"
    assert min_gap < expected + 0.4  # and it does get there (it is not frozen far away)


def test_closed_loop_person_walking_at_the_robot():
    """A person walks head-on at the robot at 1.2 m/s while it is commanded full speed: by the time they meet, the
    robot has (almost) stopped -- the person walks into a standing robot, the robot does not push into them."""
    dt, worst = _controller()
    gain, deadband, drift = torch.tensor([[1.1, 1.0, 1.0]]), torch.zeros(1, 3), torch.zeros(1)
    x_person, x, vel = 8.0, 0.0, torch.zeros(1, 3)
    queue = [torch.zeros(1, 3)] * (worst["delay"] + 1)
    prev = None
    for k in range(200):
        pts = torch.tensor(_column(x_person - x, 0.0, 1.7, seed=k), dtype=torch.float32)[None]
        # what the robot sees: the person's leading edge moved (current vs previous scan)
        moving = torch.ones(pts.shape[:2], dtype=torch.bool) if prev is not None and prev > x_person - x + 0.05 else None
        prev = x_person - x
        safe, _ = shield(pts, torch.tensor([[1.5, 0.0, 0.0]]), velocity=vel, moving=moving)
        for _ in range(5):
            queue = [safe] + queue[:-1]
            vel = track(vel, queue[-1], dt, gain, deadband, worst["tau"], worst["accel"], drift)
            x += float(vel[0, 0]) * dt
            x_person -= 1.2 * dt
        if x_person - 0.15 - x - FRONT <= 0.0:
            break
    assert float(vel[0, 0]) < 0.1, f"robot still at {float(vel[0, 0]):.2f} m/s when the person reached it"


def test_guard_alternatives_see_a_person_among_many_floor_returns():
    """With thousands of floor returns close by, a person 0.5 m ahead still blocks the guard's alternatives."""
    from g1_patrol.shield import ShieldGuard

    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-1.5, 1.5, (6000, 2)), np.full((6000, 1), FLOOR + 0.01)]
    person = _column(FRONT + 0.15 + 0.5, 0.0, 1.7)
    pts = torch.tensor(np.concatenate([floor, person]), dtype=torch.float32)[None]
    moving = torch.zeros(pts.shape[:2], dtype=torch.bool)
    moving[0, len(floor):] = True  # the person walks
    guard = ShieldGuard(1, CFG)
    safe, info = guard(pts, torch.tensor([[0.6, 0.0, 0.3]]), 0.1, torch.zeros(1, 3), moving)
    assert float(safe[0, 0]) <= 0.02  # no step towards the person, whichever alternative


def test_guard_alternatives_only_from_rest():
    """Turning into a wall 5 cm alongside is blocked; from rest the guard drives on, stepping away from the wall; at
    speed it does not send the robot another way (the shield's check assumes the robot follows a new command at
    once)."""
    from g1_patrol.shield import ShieldGuard

    wall = np.c_[np.linspace(-1.5, 1.5, 600), np.full(600, -CFG.half_width - 0.05), FLOOR + np.linspace(0.1, 2.0, 600)]
    pts = torch.tensor(wall, dtype=torch.float32)[None]
    command = torch.tensor([[0.5, 0.0, -0.8]])
    at_rest, info = ShieldGuard(1, CFG)(pts, command, 0.1, torch.zeros(1, 3))
    assert int(info["alternative"][0]) >= 0 and float(at_rest[0, 0]) > 0.1 and float(at_rest[0, 1]) >= 0.0
    _, info = ShieldGuard(1, CFG)(pts, command, 0.1, torch.tensor([[0.5, 0.0, 0.0]]))
    assert int(info["alternative"][0]) == -1


def test_guard_remembers_a_person_until_the_robot_has_moved_away():
    """With crawling on: a person who moved 1 m ahead and then left the scan's view (no returns) still makes the robot
    crawl; once it has backed off out of their reach, or after ``crawl_hold`` s, it no longer does."""
    from g1_patrol.shield import ShieldGuard

    crawl = ShieldCfg(near_moving_gap=1.2)
    person = torch.tensor(_column(FRONT + 1.0, 0.0, 1.7), dtype=torch.float32)[None]
    nothing = torch.full((1, 1, 3), float("inf"))
    forward = torch.tensor([[0.6, 0.0, 0.0]])
    guard = ShieldGuard(1, crawl)
    _, info = guard(person, forward, 0.1, torch.zeros(1, 3), torch.ones(person.shape[:2], dtype=torch.bool))
    assert bool(info["crawling"][0])
    safe, info = guard(nothing, forward, 0.1, torch.zeros(1, 3))  # out of view, still there
    assert bool(info["crawling"][0]) and float(safe[0, 0]) <= CFG.near_moving_speed + 1e-6
    for _ in range(10):  # backing off at 0.5 m/s for 1 s
        _, info = guard(nothing, torch.tensor([[-0.5, 0.0, 0.0]]), 0.1, torch.tensor([[-0.5, 0.0, 0.0]]))
    assert not bool(info["crawling"][0])
    guard = ShieldGuard(1, crawl, crawl_hold=1.0)
    guard(person, forward, 0.1, torch.zeros(1, 3), torch.ones(person.shape[:2], dtype=torch.bool))
    for _ in range(11):  # standing still, 1.1 s later
        _, info = guard(nothing, forward, 0.1, torch.zeros(1, 3))
    assert not bool(info["crawling"][0])
