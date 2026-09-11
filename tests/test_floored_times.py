"""floored_times must make tolerance-collisions impossible, not merely ordering."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.classes import F_MEASURE_TOLERANCE
from alignbeat.head import MIN_GAP_SECONDS, floored_times, monotonic_times

W = 30.0


@pytest.mark.parametrize("scale", [0.0, 1.0, 20.0, 1e4])
@pytest.mark.parametrize("N", [148, 188, 214])
def test_minimum_gap_holds_for_any_input(N, scale):
    """The floor is architectural: it must hold for every r, not on average."""
    torch.manual_seed(0)
    t = floored_times(torch.randn(8, N) * scale, W)
    gaps = torch.diff(t, dim=-1) * W
    assert float(gaps.min()) >= MIN_GAP_SECONDS   # no epsilon: it must actually hold


@pytest.mark.parametrize("N", [148, 188])
def test_two_detections_cannot_share_a_reference(N):
    """The property the metric needs: no reference beat is within tolerance of two
    candidates. Follows from min gap >= 2 * tolerance, checked directly."""
    torch.manual_seed(1)
    t = floored_times(torch.randn(4, N) * 8, W)[0] * W
    for ref in torch.linspace(0.5, W - 0.5, 400):
        assert int((torch.abs(t - ref) <= F_MEASURE_TOLERANCE).sum()) <= 1


def test_current_parameterisation_does_not_have_this_property():
    """Control. monotonic_times guarantees ordering only, and its floor is
    cell - 2*reach = 16 ms, far inside the 70 ms tolerance -- which is why duplicates
    are possible at all."""
    torch.manual_seed(2)
    gaps = torch.diff(monotonic_times(torch.randn(4, 188) * 8), dim=-1) * W
    assert float(gaps.min()) < 2 * F_MEASURE_TOLERANCE


def test_still_strictly_increasing_and_normalised():
    torch.manual_seed(3)
    t = floored_times(torch.randn(4, 188) * 8, W)
    assert bool((torch.diff(t, dim=-1) > 0).all())
    assert float(t.min()) > 0.0 and abs(float(t.max()) - 1.0) < 1e-5


def test_refuses_an_infeasible_candidate_count():
    with pytest.raises(ValueError):
        floored_times(torch.randn(1, 215), W)
