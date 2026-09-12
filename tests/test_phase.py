"""Brute-force verification of the beat-only E-step (algorithm5 v5-2).

The joint responsibility pi_{omega,L} over (omega, L), the EM surrogate it defines, and
the degenerate case where no meter hypothesis is viable. The joint-phase and meter-
marginal helpers these tests used to share with alignbeat.dp are gone; what remains is
checked directly against enumeration.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.classes import BEAT, DOWNBEAT, METER_PRIOR
from alignbeat.criterion import SubsetCriterion

DATA_PRIOR = {"downbeat": 0.2853, "beat": 0.7147}   # fold-0 pi_data, measured

def prior_over(candidates):
    """pi_M restricted to `candidates` and renormalised -- what the datamodule's
    get_train_meter_prior returns, built here from the corpus table so the tests do
    not need a dataset."""
    total = sum(METER_PRIOR[L] for L in candidates)
    return {L: METER_PRIOR[L] / total for L in candidates}


torch.manual_seed(0)


def test_pi_omega_L_matches_brute_force_and_couples_events():
    """Algorithm 1 line 40's pi_{omega,L}, against enumeration."""
    for L, M in [(2, 6), (3, 7), (4, 9)]:
        crit = SubsetCriterion(DATA_PRIOR, meter_prior={L: 1.0})
        torch.manual_seed(L)
        matched = torch.log_softmax(torch.randn(M, 3, dtype=torch.float64) * 2, dim=-1)
        scores = crit._log_meter_phase_scores(crit._class_log_posterior(matched))
        got = torch.softmax(torch.cat([scores[k] for k in scores]), dim=0)

        # Brute force: section 1.3's P_hat, pi_data divided out before pi_C is applied,
        # then one factor per event under each (omega, L) pattern.
        lc = crit.class_prior.to(matched.dtype).log()
        ld = crit.data_prior.to(matched.dtype).log()
        log_db = matched[:, DOWNBEAT] - ld[DOWNBEAT] + lc[DOWNBEAT]
        log_b = matched[:, BEAT] - ld[BEAT] + lc[BEAT]
        norm = torch.logaddexp(log_db, log_b)
        want = np.array([float(sum((log_db[i] if (w + i) % L == 0 else log_b[i]) - norm[i]
                                   for i in range(M))) for w in range(L)])
        want = np.exp(want - want.max()); want /= want.sum()
        assert np.allclose(want, got.numpy(), atol=1e-12), (L, want, got)

        # The property that makes the joint worth having: one event's evidence moves
        # every other event's responsibility, because they share one (omega, L).
        perturbed = matched.clone()
        perturbed[0] = torch.log_softmax(
            torch.tensor([5.0, -5.0, -5.0], dtype=torch.float64), dim=-1)
        moved = crit._log_meter_phase_scores(crit._class_log_posterior(perturbed))
        moved = torch.softmax(torch.cat([moved[k] for k in moved]), dim=0)
        assert not np.allclose(got.numpy(), moved.numpy(), atol=1e-6), (
            "pi must couple across events")
    print("ok: pi_{omega,L} == brute force, and couples across events")


def test_surrogate_matches_the_direct_marginal_gradient():
    """Note 43's claim: line 52's surrogate under a frozen pi has the same gradient at
    theta = theta_old as the direct marginal log-likelihood. Fisher's identity."""
    crit = SubsetCriterion(DATA_PRIOR, meter_prior=prior_over((2, 3, 4, 5, 6, 8)))
    torch.manual_seed(0)
    logits = (torch.randn(9, 3, dtype=torch.float64) * 2).requires_grad_(True)
    matched = torch.log_softmax(logits, dim=-1)

    def flat(log_p):
        blocks = crit._log_meter_phase_scores(crit._class_log_posterior(log_p))
        return torch.cat([blocks[k] for k in blocks])

    direct = -flat(matched).logsumexp(dim=0)
    g_direct, = torch.autograd.grad(direct, logits, retain_graph=True)

    with torch.no_grad():
        pi = torch.softmax(flat(matched), dim=0)
    g_surrogate, = torch.autograd.grad(crit._beat_only_term(matched, pi), logits)

    assert torch.allclose(g_direct, g_surrogate, atol=1e-12), (g_direct, g_surrogate)
    print("ok: line 52's surrogate and the direct marginal share a gradient (Fisher)")


def test_degenerate_meter_contributes_no_class_term():
    """A fragment with no viable meter hypothesis must NOT be force-fitted to "beat"."""
    torch.manual_seed(0)
    crit = SubsetCriterion(DATA_PRIOR, meter_prior={1: 1.0})
    matched = torch.log_softmax(torch.randn(6, 3), dim=-1)
    assert crit._log_meter_phase_scores(crit._class_log_posterior(matched)) is None, (
        "a degenerate meter must yield no hypothesis")
    term = crit._beat_only_term(matched, None)
    assert torch.allclose(term, torch.zeros_like(term)), (
        "no hypothesis means line 52 has no value, so no class term")
    print("ok: degenerate meter contributes no class term, not a confident beat")


if __name__ == "__main__":
    test_pi_omega_L_matches_brute_force_and_couples_events()
    test_surrogate_matches_the_direct_marginal_gradient()
    test_degenerate_meter_contributes_no_class_term()
    print("\nall phase tests passed")
