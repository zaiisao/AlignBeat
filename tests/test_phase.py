"""Brute-force verification of the phase constructions (sections 8, 8.1, 8.3, 8.5, 8.6).

Run: python tests/test_phase.py

The paper states that eq. (19), (25), (26)-(29) and (32)-(34) were each checked
against brute-force enumeration on a minimal case. These are those checks. They are
the load-bearing tests for this file: every one of the constructions below is a
claim that some cheap recursion equals an exponentially large sum or minimum, and
enumeration is the only way to confirm that rather than assume it.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.subset_head import (  # noqa: E402
    BEAT, DOWNBEAT, SubsetCriterion,
    downbeat_marginal_from_phase_posterior,
    downbeat_marginal_over_meters,
    event_is_downbeat_under,
    joint_phase_log_partition,
    joint_phase_matching_marginals,
    joint_phase_posterior,
    meter_joint_log_partition,
    meter_posterior,
    phase_class_nll,
    phase_star,
    subset_select_dp_joint_phase,
    subset_select_dp_phase_segments,
)

torch.manual_seed(0)


def injections(M, N):
    """Every order-preserving injection from {1..M} into {0..N-1}."""
    return list(itertools.combinations(range(N), M))


def phase_costs(log_probabilities, event_times, t_hat, L, lambda_l1, gamma):
    """L'^p_match (section 8.5): phase class term + timing - gamma * background."""
    M = event_times.shape[0]
    background = -log_probabilities[:, 2]
    timing = lambda_l1 * (event_times[:, None] - t_hat[None, :]).abs()
    return torch.stack([
        phase_class_nll(log_probabilities, M, L, p) + timing - gamma * background[None, :]
        for p in range(L)
    ])


def random_case(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    log_probabilities = torch.log_softmax(torch.randn(N, 3, generator=g, dtype=torch.float64), dim=-1)
    t_hat = torch.sort(torch.rand(N, generator=g, dtype=torch.float64)).values
    event_times = torch.sort(torch.rand(M, generator=g, dtype=torch.float64)).values
    return log_probabilities, t_hat, event_times


# --- phase bookkeeping -----------------------------------------------------

def test_phase_star_is_the_unique_downbeat_hypothesis():
    for L in range(2, 8):
        for M in range(1, 12):
            stars = phase_star(M, L)
            for i0 in range(M):
                matching = [p for p in range(L)
                            if bool(event_is_downbeat_under(M, L, p)[i0])]
                assert matching == [int(stars[i0])], (L, M, i0, matching, stars)
    print("ok: p*_i is a singleton and matches the (p + i - 1) mod L = 0 condition")


def test_phase_class_nll_reads_the_right_class():
    log_probabilities = torch.log_softmax(torch.randn(5, 3, dtype=torch.float64), dim=-1)
    M, L, p = 4, 3, 1
    got = phase_class_nll(log_probabilities, M, L, p)
    for i0 in range(M):
        want_db = ((p + i0) % L) == 0
        column = -log_probabilities[:, DOWNBEAT if want_db else BEAT]
        assert torch.allclose(got[i0], column)
    print("ok: phase-valued query reads DB at phase 0 and B at every nonzero phase")


# --- section 8.3, equation (19) --------------------------------------------

def test_joint_map_matches_brute_force():
    """Equation (19) against enumeration of every (sigma, phi_0) pair."""
    differed = 0
    for seed in range(40):
        M, N, L = 2, 3, 2
        log_probabilities, t_hat, event_times = random_case(M, N, seed)
        costs = phase_costs(log_probabilities, event_times, t_hat, L, 5.0, 0.5)

        best = min(
            ((p, sigma) for p in range(L) for sigma in injections(M, N)),
            key=lambda ps: float(sum(costs[ps[0], i, ps[1][i]] for i in range(M))))
        want_p, want_sigma = best

        got_sigma, got_p = subset_select_dp_joint_phase(costs.numpy())
        want_cost = float(sum(costs[want_p, i, want_sigma[i]] for i in range(M)))
        got_cost = float(sum(costs[got_p, i, got_sigma[i]] for i in range(M)))
        assert abs(want_cost - got_cost) < 1e-9, (seed, want_cost, got_cost)

        # per-hypothesis optima genuinely differ, or the coupling would be vacuous
        per_p = [min(injections(M, N),
                     key=lambda s: float(sum(costs[p, i, s[i]] for i in range(M))))
                 for p in range(L)]
        if per_p[0] != per_p[1]:
            differed += 1
    assert differed > 0, "sigma never differed across phase hypotheses; coupling is vacuous"
    print(f"ok: joint MAP (19) == brute force; sigma differed across phases in "
          f"{differed}/40 cases")


# --- section 8.3, equation (20), mixed meter -------------------------------

def test_mixed_meter_dp_matches_brute_force():
    """Equation (20) against enumeration over (sigma, per-segment phases)."""
    for seed in range(20):
        M, N = 4, 6
        segments = [(0, 2), (2, 3)]          # events 0-1 in 2/4, events 2-3 in 3/4
        log_probabilities, t_hat, event_times = random_case(M, N, seed)
        lambda_l1, gamma = 4.0, 0.5
        background = -log_probabilities[:, 2]

        def cost_fn(i0, phi):
            column = DOWNBEAT if phi == 0 else BEAT
            klass = -log_probabilities[:, column]
            timing = lambda_l1 * (event_times[i0] - t_hat).abs()
            return (klass + timing - gamma * background).numpy()

        got = subset_select_dp_phase_segments(cost_fn, M, N, segments)
        got_cost = min(
            sum(cost_fn(i0, phi_for(i0, p1, p2, segments))[got[i0]] for i0 in range(M))
            for p1 in range(2) for p2 in range(3))

        want = min(
            (sum(cost_fn(i0, phi_for(i0, p1, p2, segments))[sigma[i0]] for i0 in range(M))
             for sigma in injections(M, N) for p1 in range(2) for p2 in range(3)))
        assert abs(want - got_cost) < 1e-9, (seed, want, got_cost)
    print("ok: mixed-meter augmented DP (20) == brute force over (sigma, phases)")


def phi_for(i0, p1, p2, segments):
    """Phase of event i0 given each segment's own origin hypothesis."""
    if i0 < segments[1][0]:
        return (p1 + i0 - segments[0][0]) % segments[0][1]
    return (p2 + i0 - segments[1][0]) % segments[1][1]


# --- section 8.5, equations (25)-(29) --------------------------------------

def test_joint_partition_matches_brute_force():
    """Equation (25) against enumeration, and eq. (26)-(27) posteriors with it."""
    for seed in range(30):
        M, N, L = 2, 3, 2
        log_probabilities, t_hat, event_times = random_case(M, N, seed)
        costs = phase_costs(log_probabilities, event_times, t_hat, L, 5.0, 0.5)

        want = torch.logsumexp(torch.stack([
            -sum(costs[p, i, sigma[i]] for i in range(M))
            for p in range(L) for sigma in injections(M, N)]), dim=0)
        got, per_phase = joint_phase_log_partition(costs)
        assert torch.allclose(want, got, atol=1e-10), (seed, float(want), float(got))

        # eq. (26): the phase posterior
        want_p = torch.stack([
            torch.logsumexp(torch.stack([
                -sum(costs[p, i, sigma[i]] for i in range(M))
                for sigma in injections(M, N)]), dim=0) - want
            for p in range(L)])
        got_p = joint_phase_posterior(per_phase)
        assert torch.allclose(want_p, got_p, atol=1e-10)

        # eq. (27): r_i is a lookup, and must equal the honest marginal
        got_r = downbeat_marginal_from_phase_posterior(got_p, M, L)
        want_r = torch.stack([
            sum(got_p[p].exp() for p in range(L)
                if bool(event_is_downbeat_under(M, L, p)[i0]))
            for i0 in range(M)])
        assert torch.allclose(want_r, got_r, atol=1e-10)
    print("ok: joint partition (25), phase posterior (26) and r_i (27) == brute force")


def test_joint_matching_marginal_matches_brute_force():
    """Equation (29) against enumeration of the full joint distribution."""
    for seed in range(20):
        M, N, L = 2, 4, 2
        log_probabilities, t_hat, event_times = random_case(M, N, seed)
        costs = phase_costs(log_probabilities, event_times, t_hat, L, 5.0, 0.5)

        weights = {}
        total = 0.0
        for p in range(L):
            for sigma in injections(M, N):
                w = float(torch.exp(-sum(costs[p, i, sigma[i]] for i in range(M))))
                total += w
                for i in range(M):
                    weights[(i, sigma[i])] = weights.get((i, sigma[i]), 0.0) + w
        want = torch.zeros(M, N, dtype=torch.float64)
        for (i, j), w in weights.items():
            want[i, j] = w / total

        _, per_phase = joint_phase_log_partition(costs)
        got = joint_phase_matching_marginals(costs, joint_phase_posterior(per_phase))
        assert torch.allclose(want, got, atol=1e-9), (seed, want, got)
        assert torch.allclose(got.sum(dim=1), torch.ones(M, dtype=torch.float64), atol=1e-9)
    print("ok: joint matching marginal (29) == brute force, rows sum to 1")


# --- section 8.6, equations (32)-(34) --------------------------------------

def test_meter_marginalization_matches_brute_force():
    """Equations (32)-(34) against enumeration over every (L, p, sigma) triple."""
    meters = [2, 3]
    for seed in range(30):
        M, N = 2, 3
        log_probabilities, t_hat, event_times = random_case(M, N, seed)
        lambda_l1, gamma = 5.0, 0.5
        background = -log_probabilities[:, 2]
        timing = lambda_l1 * (event_times[:, None] - t_hat[None, :]).abs()

        def cost_builder(L, p):
            return (phase_class_nll(log_probabilities, M, L, p)
                    + timing - gamma * background[None, :])

        terms, by_meter = [], {L: [] for L in meters}
        for L in meters:
            for p in range(L):
                c = cost_builder(L, p)
                for sigma in injections(M, N):
                    value = -sum(c[i, sigma[i]] for i in range(M))
                    terms.append(value)
                    by_meter[L].append(value)
        want = torch.logsumexp(torch.stack(terms), dim=0)

        got, hypotheses = meter_joint_log_partition(cost_builder, meters)
        assert torch.allclose(want, got, atol=1e-10), (seed, float(want), float(got))

        # eq. (33): the meter posterior
        got_meter = meter_posterior(hypotheses, got)
        for L in meters:
            want_L = (torch.logsumexp(torch.stack(by_meter[L]), dim=0) - want).exp()
            assert torch.allclose(want_L, got_meter[L], atol=1e-10)
        assert abs(float(sum(got_meter.values())) - 1.0) < 1e-9

        # eq. (34): r_i summed over candidate meters
        got_r = downbeat_marginal_over_meters(hypotheses, got, M)
        want_r = torch.zeros(M, dtype=torch.float64)
        for L in meters:
            for p in range(L):
                c = cost_builder(L, p)
                mass = torch.logsumexp(torch.stack([
                    -sum(c[i, sigma[i]] for i in range(M))
                    for sigma in injections(M, N)]), dim=0)
                for i0 in range(M):
                    if bool(event_is_downbeat_under(M, L, p)[i0]):
                        want_r[i0] += (mass - want).exp()
        assert torch.allclose(want_r, got_r, atol=1e-9), (seed, want_r, got_r)
    print("ok: meter marginalization (32), meter posterior (33) and r_i (34) == brute force")


def test_em_posterior_matches_brute_force_and_couples_events():
    """Equations (12)/(14), and the property that makes them worth having.

    The single-event marginal (9) has exactly zero gradient along a B-versus-DB
    redistribution, so the only thing that can teach the distinction is coupling
    across the fragment. It is not enough that r_i be computed correctly: perturbing
    ONE event's prediction must move the OTHER events' r_i, or the construction has
    silently degenerated back into the per-candidate estimator (9) already was.
    """
    from alignbeat.subset_head import SubsetCriterion

    for L, M in [(2, 6), (3, 7), (4, 9)]:
        criterion = SubsetCriterion(meter_length=L, diagnostic_every=10 ** 9,
                                    beat_only_warmup=0)
        torch.manual_seed(L)
        matched_log = torch.log_softmax(torch.randn(M, 3, dtype=torch.float64) * 2, dim=-1)
        got, got_valid = criterion._phase_posterior_marginal(matched_log)
        assert bool(got_valid.all()), "a well-formed single-meter fragment must be valid everywhere"

        log_pi = [float(sum(matched_log[i0, DOWNBEAT if (p + i0) % L == 0 else BEAT]
                            for i0 in range(M)))
                  for p in range(L)]
        pi = np.exp(np.array(log_pi) - max(log_pi))
        pi /= pi.sum()
        want = np.array([pi[(-i0) % L] for i0 in range(M)])
        assert np.allclose(want, got.numpy(), atol=1e-12), (L, want, got)

        perturbed = matched_log.clone()
        perturbed[0] = torch.log_softmax(
            torch.tensor([5.0, -5.0, -5.0], dtype=torch.float64), dim=-1)
        moved, _ = criterion._phase_posterior_marginal(perturbed)
        assert not np.allclose(got[1:].numpy(), moved[1:].numpy(), atol=1e-6), (
            "r_i is not coupled across events -- degenerated to the (9) failure mode")
    print("ok: EM posterior (12)/(14) == brute force, and r_i couples across events")


def test_segment_posteriors_use_only_their_own_events():
    """Equation (17): each segment's posterior sees its own events and nothing else."""
    from alignbeat.subset_head import SubsetCriterion

    def criterion(meter):
        return SubsetCriterion(meter_length=meter, diagnostic_every=10 ** 9,
                               beat_only_warmup=0)

    torch.manual_seed(0)
    matched_log = torch.log_softmax(torch.randn(7, 3, dtype=torch.float64) * 2, dim=-1)
    mixed, mixed_valid = criterion(0)._phase_posterior_marginal(matched_log, segments=[(0, 2), (3, 3)])
    assert bool(mixed_valid.all())
    assert torch.allclose(mixed[:3], criterion(2)._phase_posterior_marginal(matched_log[:3])[0])
    assert torch.allclose(mixed[3:], criterion(3)._phase_posterior_marginal(matched_log[3:])[0])
    print("ok: mixed-meter segment posteriors (17) factor across segments")


def test_joint_loss_is_differentiable():
    """Equation (30) requires no stop-gradient: gradient must reach the logits."""
    M, N, L = 3, 8, 4
    logits = torch.randn(N, 3, dtype=torch.float64, requires_grad=True)
    t_hat = torch.sort(torch.rand(N, dtype=torch.float64)).values
    event_times = torch.sort(torch.rand(M, dtype=torch.float64)).values
    log_probabilities = torch.log_softmax(logits, dim=-1)
    costs = phase_costs(log_probabilities, event_times, t_hat, L, 5.0, 0.5)
    loss = -joint_phase_log_partition(costs)[0]
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0
    print("ok: -log Z_joint (30) is differentiable end to end, no stop-gradient")


if __name__ == "__main__":
    test_phase_star_is_the_unique_downbeat_hypothesis()
    test_phase_class_nll_reads_the_right_class()
    test_joint_map_matches_brute_force()
    test_mixed_meter_dp_matches_brute_force()
    test_joint_partition_matches_brute_force()
    test_joint_matching_marginal_matches_brute_force()
    test_meter_marginalization_matches_brute_force()
    test_em_posterior_matches_brute_force_and_couples_events()
    test_segment_posteriors_use_only_their_own_events()
    test_joint_loss_is_differentiable()
    print("\nall phase tests passed")


def test_degenerate_segment_falls_back_to_eq9_not_a_confident_beat():
    """A segment with no usable meter must NOT be force-fitted to "beat".

    Regression: the segment loop skipped meter <= 1, leaving r = 0 for those events.
    The confidence gate reads max(r, 1-r) = 1 >= threshold, so r = 0 was indistinguishable
    from a CONFIDENT downbeat-probability-zero pseudo-label, asserting exactly what an
    absent meter means we do not know. Those events must fall back to eq. (9) instead.
    """
    torch.manual_seed(0)
    crit = SubsetCriterion(meter_length=4, beat_only_warmup=0, beat_only_confidence=0.7,
                           diagnostic_every=0)
    matched_log = torch.log_softmax(torch.randn(6, 3), dim=-1)
    # segment 0 has a degenerate meter (1), segment 1 is a normal 3
    r, valid = crit._phase_posterior_marginal(matched_log, segments=[(0, 1), (3, 3)])
    assert not bool(valid[:3].any()), "degenerate segment must be marked invalid"
    assert bool(valid[3:].all()), "well-formed segment must stay valid"

    term, _ = crit._beat_only_term(matched_log, segments=[(0, 1), (3, 3)])
    eq9 = -torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)
    assert torch.allclose(term[:3], eq9[:3]), "degenerate segment must use eq. (9)"
    print("ok: degenerate segment meter falls back to eq (9), not a confident beat")


test_degenerate_segment_falls_back_to_eq9_not_a_confident_beat()
