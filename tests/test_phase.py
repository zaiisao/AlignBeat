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

from alignbeat.classes import BEAT, DOWNBEAT
from alignbeat.criterion import SubsetCriterion

torch.manual_seed(0)


def test_em_posterior_matches_brute_force_and_couples_events():
    """Equations (12)/(14), and the property that makes them worth having."""

    for L, M in [(2, 6), (3, 7), (4, 9)]:
        criterion = SubsetCriterion(meter_candidates=(L,))
        torch.manual_seed(L)
        matched_log = torch.log_softmax(torch.randn(M, 3, dtype=torch.float64) * 2, dim=-1)
        got, _, _ = criterion._compute_latent_posterior(matched_log)

        # The brute force must score hypotheses the same way the criterion is
        # configured to. pi_C there is a double-count -- c_i(p, L) is deterministic
        # given the hypothesis, and pi_C(DB) = E[1/L] is already applied as log P(L),
        # so including it charges (log pi_C(DB) - log pi_C(B)) = -1.008 nats per
        # claimed downbeat, i.e. -1.008 * M/L, monotone in L. It is still the default
        # because no trained arm has shown the fix helps end-to-end; this test tracks
        # the flag rather than asserting either form is the right one.
        if criterion.hypothesis_class_prior:
            log_prior_c = criterion.log_class_prior.to(matched_log.dtype)
            log_db = matched_log[:, DOWNBEAT] + log_prior_c[DOWNBEAT]
            log_b = matched_log[:, BEAT] + log_prior_c[BEAT]
        else:
            log_db = matched_log[:, DOWNBEAT]
            log_b = matched_log[:, BEAT]
        log_norm = torch.logaddexp(log_db, log_b)
        log_pi = [float(sum((log_db[i0] if (p + i0) % L == 0 else log_b[i0]) - log_norm[i0]
                            for i0 in range(M)))
                  for p in range(L)]
        pi = np.exp(np.array(log_pi) - max(log_pi))
        pi /= pi.sum()
        want = np.array([pi[(-i0) % L] for i0 in range(M)])
        assert np.allclose(want, got.numpy(), atol=1e-12), (L, want, got)

        perturbed = matched_log.clone()
        perturbed[0] = torch.log_softmax(
            torch.tensor([5.0, -5.0, -5.0], dtype=torch.float64), dim=-1)
        moved, _, _ = criterion._compute_latent_posterior(perturbed)
        assert not np.allclose(got[1:].numpy(), moved[1:].numpy(), atol=1e-6), (
            "r_i is not coupled across events -- degenerated to the (9) failure mode")
    print("ok: EM posterior (12)/(14) == brute force, and r_i couples across events")


def test_em_surrogate_matches_the_direct_marginal_gradient():
    """Algorithm 1 note 33's own claim: line 45's surrogate, under pi_{psi,L} frozen at
    theta_old, has the same gradient at theta = theta_old as the direct marginal
    log-likelihood -log sum_h exp(score_h). Fisher's identity. The two values differ --
    the surrogate is a bound -- but only the gradient has to agree, and it does."""

    # Fisher's identity is a property of SOFT EM. It needs the
    # expectation under the FULL posterior, so it holds for the mixture over every
    # (omega, L) -- the default, and what algorithm5_hard1 line 52 specifies for
    # training -- and NOT for meter_mixture=False, which hard-selects argmax_L P(L | x).
    # That is hard EM on the meter axis: exact in its own right, but optimising a
    # MAP-conditioned surrogate rather than the marginal likelihood, so its gradient is
    # deliberately not the marginal's. The second block pins that difference so it stays
    # a choice, not an accident. (hard1 puts its own argmax at INFERENCE, Algorithm 3
    # line 20, a decode path this repo does not have.)
    crit = SubsetCriterion(meter_candidates=(2, 3, 4, 5, 6, 8))
    assert crit.meter_mixture, 'the soft mixture is the default; Fisher needs it'
    torch.manual_seed(0)
    logits = (torch.randn(9, 3, dtype=torch.float64) * 2).requires_grad_(True)
    matched = torch.log_softmax(logits, dim=-1)

    blocks = crit._hypothesis_log_scores(matched)
    direct = -torch.cat([blocks[L] for L in blocks]).logsumexp(dim=0)
    g_direct, = torch.autograd.grad(direct, logits, retain_graph=True)

    with torch.no_grad():
        r, _, prior_term = crit._compute_latent_posterior(matched)
    surrogate = crit._beat_only_term(matched, r).sum() + prior_term
    g_surrogate, = torch.autograd.grad(surrogate, logits, retain_graph=True)

    assert torch.allclose(g_direct, g_surrogate, atol=1e-12), (g_direct, g_surrogate)

    hard = SubsetCriterion(meter_candidates=(2, 3, 4, 5, 6, 8), meter_mixture=False)
    with torch.no_grad():
        r_h, _, prior_h = hard._compute_latent_posterior(matched)
    g_hard, = torch.autograd.grad(
        hard._beat_only_term(matched, r_h).sum() + prior_h, logits)
    assert not torch.allclose(g_direct, g_hard, atol=1e-6), (
        "hard meter selection must NOT reproduce the marginal's gradient; if it does, "
        "the argmax is not actually restricting r_i to one meter")
    print("ok: EM surrogate (45) shares a gradient with the direct marginal under the "
          "soft mixture (Fisher), and deliberately does not under argmax L")


def test_degenerate_meter_falls_back_to_eq9_not_a_confident_beat():
    """A fragment with no viable meter hypothesis must NOT be force-fitted to "beat"."""
    torch.manual_seed(0)
    crit = SubsetCriterion(meter_candidates=(1,))
    matched_log = torch.log_softmax(torch.randn(6, 3), dim=-1)
    assert crit._compute_latent_posterior(matched_log) is None, (
        "a degenerate meter must yield no posterior")

    term = crit._beat_only_term(matched_log, None)
    eq9 = -torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)
    assert torch.allclose(term, eq9), "a fragment with no meter must use eq. (9)"
    print("ok: degenerate meter falls back to eq (9), not a confident beat")


if __name__ == "__main__":
    test_em_posterior_matches_brute_force_and_couples_events()
    test_em_surrogate_matches_the_direct_marginal_gradient()
    test_degenerate_meter_falls_back_to_eq9_not_a_confident_beat()
    print("\nall phase tests passed")
