"""Per-candidate precision: the head emits b_j, and what keeps it honest."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.classes import F_MEASURE_TOLERANCE
from alignbeat.criterion import EPS, SubsetCriterion
from alignbeat.head import SubsetSelectionHead, softplus_inverse


def test_floor_holds_by_construction():
    """b_j = b_min + softplus(u_j) cannot reach zero, for any u_j the head might emit."""
    c = SubsetCriterion(b_min=1e-3)
    for extreme in (-1e4, -100.0, 0.0, 100.0, 1e4):
        b = c.b_min + torch.nn.functional.softplus(torch.full((1, 8), extreme))
        assert torch.isfinite(b).all(), extreme
        assert float(b.min()) >= 1e-3, (extreme, float(b.min()))
    print("ok: b_j is floored at b_min by construction, even at |u_j| = 1e4")


def test_stop_gradient_decouples_the_two_roles():
    """One step cannot both widen b_j and leave t_hat_j unimproved."""
    c = SubsetCriterion()
    residual = torch.tensor([0.30, 0.05], requires_grad=True)
    raw = torch.tensor([0.5, -0.5], requires_grad=True)
    b = c.b_min + torch.nn.functional.softplus(raw)
    c._per_candidate_time_term(residual, b).sum().backward()

    # t_hat's own gradient is exactly 1/b_j: it sees the detached scale, nothing else
    assert torch.allclose(residual.grad, 1.0 / b.detach()), residual.grad
    # and b_j is still trained, through its own separate channel
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0
    print("ok: dL/dt_hat sees only a detached b_j, while b_j keeps its own gradient")


def test_normaliser_gives_b_a_finite_optimum():
    """log(2 eps + 2 b_j) is what stops the head inflating b_j to switch timing off."""
    c = SubsetCriterion()
    residual = torch.tensor([0.004, 0.001, 0.006]).sub(EPS).clamp(min=0.0)

    # Under 4.1.3's split only the precision channel sees b, so that channel is what
    # determines b's optimum: dL/db = 0 at b* = (s + sqrt(s^2 + 4 eps s)) / 2.
    s = float(residual.mean())
    expected = (s + math.sqrt(s * s + 4.0 * EPS * s)) / 2.0

    # b* is optimal for the SUM: each candidate keeps its own residual, so the
    # per-element gradients cancel rather than vanishing individually.
    b = torch.full((3,), expected, requires_grad=True)
    c._per_candidate_time_term(residual, b).sum().backward()
    scale = float(b.grad.abs().max())
    assert abs(float(b.grad.sum())) < 1e-6 * scale, (float(b.grad.sum()), scale)

    # and the gradient genuinely changes sign around it: too small pulls up, too big down
    for factor, sign in ((0.5, -1.0), (2.0, +1.0)):
        b = torch.full((3,), expected * factor, requires_grad=True)
        c._per_candidate_time_term(residual, b).sum().backward()
        assert float(b.grad.mean()) * sign > 0, (factor, float(b.grad.mean()))
    print(f"ok: b_j has a finite optimum at {expected * 30000:.0f} ms, matching the closed form")


def test_gamma_prior_pulls_toward_the_data_informed_default():
    """The MAP term is minimised at beta/(alpha-1), not at 0 or infinity."""
    alpha = 3.0
    c = SubsetCriterion(precision_prior_alpha=alpha, precision_prior_beta=2.0)
    expected_minimum = 2.0 / (alpha - 1.0)

    grid = torch.linspace(0.05, 5.0, 4000)
    values = torch.stack([c._precision_prior(g.reshape(1)) for g in grid])
    assert abs(float(grid[int(values.argmin())]) - expected_minimum) < 0.01

    at_minimum = float(c._precision_prior(torch.tensor([expected_minimum])))
    assert float(c._precision_prior(torch.tensor([50.0]))) > at_minimum   # inflation
    assert float(c._precision_prior(torch.tensor([1e-3]))) > at_minimum   # collapse
    print("ok: Gamma prior is minimised at beta/(alpha-1) and penalises both directions")


def test_head_bias_is_the_tolerance_and_stays_frozen():
    """The head starts by claiming exactly the F-measure tolerance, and cannot drift."""
    for window_seconds in (15.0, 30.0, 60.0):
        head = SubsetSelectionHead(feature_size=16, window_seconds=window_seconds)
        b0 = float(torch.nn.functional.softplus(head.precision_head.bias))
        assert abs(b0 * window_seconds - F_MEASURE_TOLERANCE) < 1e-6, window_seconds
        # only the weights thaw; the bias is frozen for the whole run
        assert head.precision_head.bias.requires_grad is False
        assert head.precision_head.weight.requires_grad is True
    print("ok: the precision bias is 70 ms of whatever window it is given, and frozen")


def test_head_emits_one_scale_per_candidate():
    head = SubsetSelectionHead(feature_size=16)
    class_logits, t_hat, b_hat = head(torch.randn(2, 16, 8))
    assert b_hat.shape == (2, 8), b_hat.shape
    assert float(b_hat.min()) > 0.0, "softplus must keep b_hat positive"
    print("ok: the head emits one positive b_hat per candidate")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
