"""Regressions for bugs that have each recurred at least once in this project."""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.classes import METER_PRIOR
from alignbeat.criterion import SubsetCriterion

DATA_PRIOR = {"downbeat": 0.2853, "beat": 0.7147}   # fold-0 pi_data, measured

def prior_over(candidates):
    """pi_M restricted to `candidates` and renormalised -- what the datamodule's
    get_train_meter_prior returns, built here from the corpus table so the tests do
    not need a dataset."""
    total = sum(METER_PRIOR[L] for L in candidates)
    return {L: METER_PRIOR[L] / total for L in candidates}

from alignbeat.dp import subset_posterior_marginals, subset_select_dp, subset_select_logsumexp


def _targets(M=40, seed=0):
    torch.manual_seed(seed)
    N = 160
    logits = torch.tensor([0.10, 0.30, 0.60]).log().repeat(1, N, 1).clone().requires_grad_(True)
    t_hat = torch.linspace(0.005, 1.0, N).unsqueeze(0)
    tg = [{'classes': torch.tensor([0 if i % 4 == 0 else 1 for i in range(M)]),
           'times': torch.linspace(0.01, 0.99, M)}]
    return logits, t_hat, tg


def test_class_term_has_gradient():
    """The class term must differentiate into the CLASS logits."""
    logits, t_hat, tg = _targets()
    losses, _ = SubsetCriterion(DATA_PRIOR)(logits, t_hat, tg)
    assert losses['class'].requires_grad, "class term is detached"
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
    """Flags have silently failed to arrive at the criterion before. Check the plumbing."""
    from beat_this.model.pl_module import PLBeatThis
    m = PLBeatThis(
        head_type="subset", transformer_dim=64, n_layers=2,
        subset_kwargs=dict(num_candidates=188, data_prior=DATA_PRIOR,
                           meter_prior=prior_over((2, 3, 4, 6)), gamma=0.25))
    c = m.subset_criterion
    assert c.meter_candidates == (2, 3, 4, 6)
    assert c.gamma == 0.25


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


def test_defaults_are_the_three_shipped_terms():
    """v1 ships class + time + background and nothing else; continuity and periodicity
    are retired, so their absence is the property to hold, not their being zero."""
    logits, t_hat, tg = _targets()
    losses, _ = SubsetCriterion(DATA_PRIOR)(logits, t_hat, tg)
    assert set(losses) == {'class', 'time', 'background', 'total'}, sorted(losses)
    assert all(torch.isfinite(v) for v in losses.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("all regression tests passed")
