"""Tests for geometric helpers of the civic tasks (pure torch, no simulator needed)."""

import math

import torch

from g1_patrol.tasks.navigation.mdp.nyc_social import _segment_distance
from g1_patrol.tasks.navigation.mdp.nyc_trash import BAG, LITTER, _in_rects, _rect_point, _rest_quat


def test_segment_distance():
    p = torch.tensor([[[0.0, 1.0]], [[3.0, 0.0]]])  # (n, 1, 2)
    a = torch.tensor([[[-1.0, 0.0]], [[0.0, 0.0]]])
    b = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    d = _segment_distance(p, a, b)
    assert torch.allclose(d, torch.tensor([[1.0], [2.0]]))  # above the middle; beyond the end


def test_rectangles():
    rects = torch.tensor([[[0.0, 2.0, 0.0, 1.0], [5.0, 6.0, 5.0, 6.0]]])  # (n, r, 4): x0, x1, y0, y1
    valid = torch.tensor([[True, False]])
    pts = torch.tensor([[[1.0, 0.5], [5.5, 5.5], [2.1, 0.5]]])
    assert _in_rects(pts, rects, valid).tolist() == [[True, False, False]]  # the second rectangle is not valid
    assert _in_rects(pts, rects, valid, margin=0.2).tolist() == [[True, False, True]]
    closest = _rect_point(torch.tensor([[3.0, 2.0]]), rects)
    assert torch.allclose(closest[0, 0], torch.tensor([2.0, 1.0]))


def test_rest_orientation():
    h = math.sqrt(0.5)
    assert _rest_quat([0.1, 0.1, 0.3], BAG) == (0.0, 0.0, 0.0, 1.0)  # bags stand
    assert _rest_quat([0.1, 0.1, 0.3], LITTER, upright=True) == (0.0, 0.0, 0.0, 1.0)
    assert _rest_quat([0.07, 0.2, 0.07], LITTER) == (0.0, h, 0.0, h)  # a bottle lies on its side
    assert _rest_quat([0.3, 0.2, 0.05], LITTER) == (0.0, 0.0, 0.0, 1.0)  # flat things lie flat
