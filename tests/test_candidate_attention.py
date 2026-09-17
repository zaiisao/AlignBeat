"""Candidate self-attention: does a candidate actually see its neighbours?

The head is otherwise a per-candidate MLP, so candidate j cannot know that j+1 is
claiming the same beat -- and 82% of false fires are within two cells of a real one.
This is the mechanism DETR uses to avoid NMS, on an anchored grid.

Two properties are worth asserting rather than assuming:
  1. with 0 layers a candidate is provably independent of every other one, and with
     attention it is not (otherwise the flag does nothing);
  2. t_hat is unchanged either way, since regression reads the pre-attention features --
     the attention must not be able to move candidates around to fix timing.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.model.head import SubsetSelectionHead, monotonic_times

WINDOW_SECONDS = 30.0   # 1500 frames at 50 fps, the training excerpt these fixtures assume


def head(layers, seed=1):
    torch.manual_seed(seed)
    h = SubsetSelectionHead(feature_size=256, attention_layers=layers).eval()
    # class_head is deliberately zero-initialised, which makes every logit constant and
    # any sensitivity probe vacuously zero. Undo it so the test measures something.
    torch.nn.init.normal_(h.class_head.weight, std=0.05)
    return h


def perturbed_pair(seed=0):
    """A perturbation input_norm can actually see: LayerNorm removes any constant added
    across the feature dim, so shifting all 256 features of a candidate is invisible."""
    torch.manual_seed(seed)
    x = torch.randn(1, 256, 188)
    y = x.clone()
    y[0, :, 100] = torch.randn(256) * 3.0
    return x, y


def sensitivity(h):
    x, y = perturbed_pair()
    with torch.no_grad():
        a, ta = h(x)
        b, tb = h(y)
    return (a - b).abs().mean(-1)[0], float((ta - tb).abs().max())


def test_without_attention_candidates_are_independent():
    d, _ = sensitivity(head(0))
    assert float(d[100]) > 0.0, "the perturbed candidate must change at all"
    assert float(d[[50, 95, 99, 101, 150]].abs().max()) == 0.0


@pytest.mark.parametrize("layers", [1, 2])
def test_attention_makes_neighbours_visible(layers):
    d, _ = sensitivity(head(layers))
    assert float(d[99]) > 0.0, "neighbour must respond once attention is on"
    assert float(d[[50, 95, 99, 101]].abs().max()) > 0.0


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_timing_is_never_affected_by_attention(layers):
    """Regression reads z before the attention pass, so t_hat must be bit-identical."""
    _, dt = sensitivity(head(layers))
    assert dt == 0.0


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_shapes_and_monotonicity(layers):
    h = head(layers)
    cls, t = h(torch.randn(2, 256, 188))
    assert cls.shape == (2, 188, 3) and t.shape == (2, 188)
    assert bool((torch.diff(t, dim=-1) > 0).all())
