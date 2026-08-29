"""Section 4.1.3: the mitigations that make per-candidate precision safe to enable.

Section 4.1.2 establishes that predicting b_j per candidate reopens two failure modes
the shared global b closes, in opposite directions: INFLATION on hard candidates
(raising b_j is a cheaper way to cut the loss than localising correctly, and it is
self-reinforcing because the residual gradient scales as 1/b_j) and COLLAPSE on easy
ones (as the residual goes to zero, log(2 b_j) is unbounded below as b_j -> 0).

Each test below pins one mitigation. They are worth having as tests rather than as
comments because every one of them is invisible in a training curve: a run with the
stop-gradient silently removed still trains, still reports a falling loss, and simply
learns worse localisation.

Run: python tests/test_precision.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.subset_head import (  # noqa: E402
    SubsetCriterion, SubsetSelectionHead,
)


def criterion(**kw):
    kw.setdefault('diagnostic_every', 10 ** 9)
    kw.setdefault('precision_warmup', 0)
    return SubsetCriterion(**kw)


def test_floor_holds_by_construction():
    """Mitigation 1: b_j = b_min + softplus(u_j) cannot reach zero, for any u_j."""
    c = criterion(b_min=1e-3)
    for extreme in (-1e4, -100.0, 0.0, 100.0, 1e4):
        raw = torch.full((1, 8), extreme)
        b = c._precision_scales(raw)
        assert torch.isfinite(b).all(), extreme
        assert float(b.min()) >= 1e-3, (extreme, float(b.min()))
    print("ok: b_j is floored at b_min by construction, even at |u_j| = 1e4")


def test_stop_gradient_decouples_the_two_roles():
    """Mitigation 3: one step cannot both widen b_j and leave t_hat_j unimproved.

    t_hat must see b_j only as a constant, while b_j still gets its own gradient --
    dropping either half would be wrong, so both are checked.
    """
    c = criterion()
    residual = torch.tensor([0.30, 0.05], requires_grad=True)
    raw = torch.tensor([0.5, -0.5], requires_grad=True)
    b = c._precision_scales(raw.unsqueeze(0))[0]
    c._per_candidate_time_term(residual, b).sum().backward()

    # t_hat's own gradient is exactly 1/b_j: it sees the detached scale, nothing else
    assert torch.allclose(residual.grad, 1.0 / b.detach()), residual.grad
    # and b_j is still trained, through its own separate term
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0
    print("ok: dL/dt_hat sees only a detached b_j, while b_j keeps its own gradient")


def test_gamma_prior_pulls_toward_the_data_informed_default():
    """Mitigation 2: the MAP term is minimised at beta/(alpha-1), not at 0 or infinity."""
    alpha = 3.0
    c = criterion(precision_prior_alpha=alpha, precision_prior_beta=2.0)
    expected_minimum = 2.0 / (alpha - 1.0)

    grid = torch.linspace(0.05, 5.0, 4000)
    values = torch.stack([c._precision_prior(g.reshape(1)) for g in grid])
    assert abs(float(grid[int(values.argmin())]) - expected_minimum) < 0.01

    # it genuinely damps BOTH failure directions, not just one
    at_minimum = float(c._precision_prior(torch.tensor([expected_minimum])))
    assert float(c._precision_prior(torch.tensor([50.0]))) > at_minimum   # inflation
    assert float(c._precision_prior(torch.tensor([1e-3]))) > at_minimum   # collapse
    print("ok: Gamma prior is minimised at beta/(alpha-1) and penalises both directions")


def test_warmup_defers_the_precision_head():
    """Mitigation 4: localisation is learned before an uncertainty channel exists."""
    c = criterion(precision_warmup=5)
    raw = torch.zeros(1, 4)
    assert c._precision_scales(raw) is None, "precision head active during warm-up"
    c._call_count = 5
    assert c._precision_scales(raw) is not None, "precision head never activates"
    assert c._precision_scales(None) is None, "no head should mean no scales"
    print("ok: the precision head is inert until precision_warmup steps have elapsed")


def test_selection_cost_is_insulated_from_b_j():
    """Section 4.1.3's closing point: the DP's cost must use the SHARED b.

    L_match is not only a loss term, it is the cost the selection minimises, formed
    before any gradient exists. An inflated b_j would make a badly localised candidate
    look artificially cheap and corrupt sigma itself, not merely the loss after it.
    """
    c = criterion(b_scale=0.01)
    log_probabilities = torch.log_softmax(torch.randn(6, 3), dim=-1)
    t_hat = torch.sort(torch.rand(6)).values
    classes = torch.tensor([1, 1])
    times = torch.sort(torch.rand(2)).values

    before = c.build_cost(log_probabilities, t_hat, classes, times)[0]
    # a wildly inflated per-candidate precision must not move the selection cost
    c._precision_scales(torch.full((1, 6), 50.0))
    after = c.build_cost(log_probabilities, t_hat, classes, times)[0]
    assert torch.allclose(before, after)
    assert abs(c.lambda_l1 - 1.0 / float(c.b)) < 1e-9
    print("ok: build_cost uses the shared global b, so b_j cannot corrupt sigma")


def test_head_emits_precision_only_when_asked():
    head_off = SubsetSelectionHead(feature_size=16, num_candidates=8, level_strides=(1,))
    head_on = SubsetSelectionHead(feature_size=16, num_candidates=8, level_strides=(1,),
                                  predict_precision=True)
    features = [torch.randn(2, 16, 8)]
    assert len(head_off(features)) == 2
    out = head_on(features)
    assert len(out) == 3 and out[2].shape == (2, 8)
    print("ok: the head returns u_j only when predict_precision is set")


if __name__ == "__main__":
    test_floor_holds_by_construction()
    test_stop_gradient_decouples_the_two_roles()
    test_gamma_prior_pulls_toward_the_data_informed_default()
    test_warmup_defers_the_precision_head()
    test_selection_cost_is_insulated_from_b_j()
    test_head_emits_precision_only_when_asked()
    print("\nall precision tests passed")
