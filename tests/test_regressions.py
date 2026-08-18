"""Regressions for bugs that have each recurred at least once in this project.

Every one of these was fixed, silently lost to a revert or a partially-applied edit, and
then cost a full training run before anyone noticed. They are cheap; run them before
launching arms.

    python -m pytest tests/test_regressions.py     (or just execute this file)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatfcos.subset_head import (SubsetCriterion, subset_select_dp,
                                  subset_select_dp_meter, subset_select_logsumexp,
                                  subset_posterior_marginals)


def _targets(M=40, seed=0):
    torch.manual_seed(seed)
    N = 160
    logits = torch.tensor([0.10, 0.30, 0.60]).log().repeat(1, N, 1).clone().requires_grad_(True)
    t_hat = torch.linspace(0.005, 1.0, N).unsqueeze(0)
    tg = [{'classes': torch.tensor([0 if i % 4 == 0 else 1 for i in range(M)]),
           'times': torch.linspace(0.01, 0.99, M)}]
    return logits, t_hat, tg


def test_marginal_class_term_has_gradient():
    """-log Z must differentiate into the CLASS logits.

    build_cost is called inside a torch.no_grad() block for the hard path; the marginal
    path must rebuild it. Reusing the detached matrix leaves the class term with
    requires_grad=False, the loss still falls via the background term, and the model
    decodes zero events. Three separate arms died this way.
    """
    logits, t_hat, tg = _targets()
    losses, _ = SubsetCriterion(diagnostic_every=10**9, marginal=True)(logits, t_hat, tg)
    assert losses['class'].requires_grad, "marginal class term is detached"
    g = torch.autograd.grad(losses['class'], logits, retain_graph=True)[0]
    assert float(g.abs().mean()) > 1e-6, "no gradient reaches the class logits"


def test_logsumexp_batched_matches_per_fragment():
    """The batched form must be exact, including ragged batches (55x speed depends on it)."""
    torch.manual_seed(0)
    cost = torch.rand(5, 20, 60).double()
    per = torch.stack([subset_select_logsumexp(cost[b]) for b in range(5)])
    assert torch.allclose(per, subset_select_logsumexp(cost), atol=1e-10)
    lengths = [12, 20, 7]
    mats = [torch.rand(m, 60).double() for m in lengths]
    ref = torch.stack([subset_select_logsumexp(m) for m in mats])
    padded = torch.zeros(3, max(lengths), 60, dtype=torch.float64)
    for b, m in enumerate(mats):
        padded[b, :m.shape[0]] = m
    got = subset_select_logsumexp(padded, lengths=torch.tensor(lengths))
    assert torch.allclose(ref, got, atol=1e-10)


def test_flags_reach_the_criterion():
    """Four flags have silently failed to arrive at the criterion. Check the plumbing."""
    from beatfcos import model_module
    m = model_module.create_beatfcos_model(
        num_classes=2, clusters=torch.tensor([0.4, 0.6, 1.2, 1.9, 3.0]), args=None,
        head_type="subset", dmodel=128, nhead=8, d_hid=512, nlayers=9, attn_len=5,
        dropout=0.1, downbeat_weight=0.6, audio_downsampling_factor=512, centerness=False,
        postprocessing_type="soft_nms", audio_sample_rate=22050, backbone_type="wavebeat",
        num_candidates=160, mu_meter=1e5, lambda_r=200.0, cont_weight=0.5,
        phase_marginal=True, meter_length=4, marginal=True)
    c = m.subset_criterion
    assert c.mu_meter == 1e5 and c.lambda_r == 200.0 and c.cont_weight == 0.5
    assert c.phase_marginal and c.meter_length == 4 and c.marginal


def test_meter_dp_reduces_to_plain_dp_and_is_monotone():
    rng = np.random.default_rng(0)
    cost = rng.random((30, 80)) * 2
    t = np.linspace(0.0, 1.0, 80)
    db = list(range(0, 30, 4))
    assert np.array_equal(subset_select_dp(cost), subset_select_dp_meter(cost, db, t, 4.0, 0.0))
    sigma = subset_select_dp_meter(cost, db, t, 4.0, 1e5)
    assert np.all(np.diff(sigma) > 0), "order-preserving constraint violated"


def test_posterior_matches_brute_force():
    import itertools
    torch.manual_seed(0)
    cost = (torch.rand(3, 6) * 2).double()
    Z, w = 0.0, torch.zeros(3, 6).double()
    for sig in itertools.combinations(range(6), 3):
        p = float(torch.exp(-sum(cost[i, sig[i]] for i in range(3))))
        Z += p
        for i in range(3):
            w[i, sig[i]] += p
    assert torch.allclose(subset_posterior_marginals(cost), w / Z, atol=1e-7)


def test_defaults_unchanged():
    """Every new term must be inert at defaults, or old runs are not reproducible."""
    logits, t_hat, tg = _targets()
    losses, _ = SubsetCriterion(diagnostic_every=10**9)(logits, t_hat, tg)
    for key in ('continuity', 'periodicity'):
        assert float(losses[key]) == 0.0, f"{key} is active at defaults"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("all regression tests passed")
