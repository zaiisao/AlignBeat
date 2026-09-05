"""Correctness tests for the continuous-phase arm (--head_type phase).

Several of these pin a DIAGNOSED DEFECT rather than a guarantee. As ported, this arm
scored dbF exactly 0.0000 on every dataset of fold 0, with a trained phase head measuring
WORSE than a fixed constant (0.2488 against 0.2484, and 0.2434 for the best constant).

The cause is not the phase objective itself. A circular-L1 head learns phase perfectly
(error 0.0001) when its input carries phase, so the loss is learnable. The cause is that
its input does not: a probe trained on the arm's own frozen candidate features recovers
nothing (0.2519, at chance), because the SHARED trunk was reallocated entirely to timing.

  b has no floor without the guards, so it tracks the raw residual toward zero
      -> the timing term's 1/b weight grows without bound
      -> gradient into the shared trunk went 22:1 in phase's favour at epoch 4 to
         237:1 in timing's favour at epoch 9, as b fell 0.589 -> 8.5e-4
      -> the trunk stops carrying phase, and phi collapses to a constant
         (its spread across candidates fell 0.0283 -> 0.0072 over those same epochs)

The subset arm is the control: its Gamma prior pins b at 61.3 ms after 100 epochs, and
it reaches dbF 0.866 from the same encoder and the same Downsample.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.dp import subset_select_dp
from alignbeat.phase_criterion import PhaseCriterion, circ_dist, phases_from_downbeats
from alignbeat.phase_decode import decode_events, infer_meter
from alignbeat.phase_head import PhaseHead, PhaseSelectionHead, PhaseTimeHead


def test_head_emits_a_valid_phase_and_a_monotone_time():
    head = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8)
    phi, t, b = head(torch.randn(2, 16, 12))
    assert phi.shape == (2, 12) and t.shape == (2, 12) and b.shape == (2,), (
        phi.shape, t.shape, b.shape)
    assert float(phi.min()) >= 0.0 and float(phi.max()) < 1.0, "phase must lie in [0, 1)"
    assert float((t[:, 1:] - t[:, :-1]).min()) > 0.0, "t_hat must be strictly increasing"
    assert float(b.min()) > 0.0, "the Laplace scale must be positive"
    print("ok: the phase head emits phi in [0,1), a strictly increasing t_hat, and b > 0")


def test_time_starts_as_the_uniform_grid():
    """Zero-initialised regression weights, exactly as SubsetSelectionHead does it."""
    head = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8)
    _, t, _ = head(torch.randn(3, 16, 20))
    gaps = t[:, 1:] - t[:, :-1]
    assert float(gaps.std()) < 1e-6, f"expected a uniform grid, gaps vary by {float(gaps.std())}"
    print("ok: t_hat begins as the uniform grid, claiming no timing before training")


def test_warm_start_pins_the_scale_then_releases_it():
    head = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8,
                              warmup_epochs=5)
    _, _, early = head(torch.randn(2, 16, 12), epoch=0)
    _, _, late = head(torch.randn(2, 16, 12), epoch=6)
    assert torch.allclose(early, torch.full_like(early, head.b_0)), early
    assert not torch.allclose(late, torch.full_like(late, head.b_0)), late
    print("ok: b_e is pinned at b_0 through the warm start and learned afterwards")


def test_skip_cost_is_a_column_subtraction():
    """The phase DP's per-candidate skip cost needs no DP of its own.

    D[i][j] = min(D[i][j-1] + skip[j-1], D[i-1][j-1] + cost[i-1][j-1]) is the existing
    recursion run on cost - skip, by a prefix-sum argument. This is the same structure
    section 8.4's background correction already uses, so the arm reuses alignbeat.dp
    rather than carrying a second implementation that could drift from it.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        M, N = int(rng.integers(2, 10)), int(rng.integers(10, 30))
        cost, skip = rng.normal(size=(M, N)), rng.random(N) * 2

        sigma = subset_select_dp(cost - skip[None, :])
        # The recursion, written out, as the thing being claimed equal.
        INF = float("inf")
        D = np.full((M + 1, N + 1), INF)
        D[0, 0] = 0.0
        for j in range(1, N + 1):
            D[0, j] = D[0, j - 1] + skip[j - 1]
        back = np.zeros((M + 1, N + 1), dtype=np.int64)
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                a, b = D[i, j - 1] + skip[j - 1], D[i - 1, j - 1] + cost[i - 1, j - 1]
                D[i, j], back[i, j] = (a, back[i, j - 1]) if a < b else (b, j)
        want, j = np.empty(M, dtype=np.int64), N
        for i in range(M, 0, -1):
            want[i - 1] = back[i, j] - 1
            j = back[i, j] - 1
        assert np.array_equal(sigma, want), (sigma, want)
    print("ok: the skip cost is exactly a column subtraction through the existing DP")


def test_phases_from_downbeats_reads_the_bar():
    phi = phases_from_downbeats(8, np.array([0, 4]))
    assert np.allclose(phi, [0, .25, .5, .75, 0, .25, .5, .75]), phi
    # A 3/4 fragment: the trailing bar has no following downbeat to measure against,
    # so it reuses the previous bar's length rather than guessing the fallback meter.
    phi = phases_from_downbeats(7, np.array([0, 3]))
    assert np.allclose(phi, [0, 1 / 3, 2 / 3, 0, 1 / 3, 2 / 3, 0]), phi
    # A single downbeat carries no spacing at all, so the fallback meter applies.
    assert np.allclose(phases_from_downbeats(4, np.array([0])), [0, .25, .5, .75])
    print("ok: bar phase is read per bar, so a meter change needs no annotation")


def test_circular_l1_learns_phase_when_the_input_carries_it():
    """The objective is NOT the defect, which is why the fix is not to replace it.

    A constant output is a stationary point -- the 4/4 target set {0, 1/4, 1/2, 3/4} is
    invariant under rotation by 1/4, so the gradient there vanishes -- but it is not an
    attractor once the input is informative. This test is what rules the loss out.
    """
    torch.manual_seed(0)
    D, N, METER = 32, 64, 4
    basis = torch.randn(METER, D)

    def batch(B=8):
        pos = torch.randint(0, METER, (B, N))
        return basis[pos] + 0.1 * torch.randn(B, N, D), pos.float() / METER

    head = PhaseHead(D, hidden=64)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(1500):
        z, target = batch()
        loss = circ_dist(target, head(z)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    z, target = batch(32)
    with torch.no_grad():
        err = float(circ_dist(target, head(z)).mean())
    assert err < 0.05, f"circular L1 failed to learn a perfectly predictable phase: {err}"

    # ...while the landscape over CONSTANTS really is flat, which is the trap it falls
    # into once the trunk has stopped carrying phase.
    targets = torch.tensor([0.0, .25, .5, .75] * 8)
    costs = torch.tensor([float(circ_dist(targets, torch.full_like(targets, c / 400)).mean())
                          for c in range(400)])
    assert float(costs.max() - costs.min()) < 1e-6, float(costs.max() - costs.min())
    print(f"ok: circular L1 learns phase from informative features ({err:.4f}), so the "
          f"loss is not the defect")


def test_the_guards_put_a_floor_under_the_timing_scale():
    """THE DEFECT, and the fix. Without the guards b* is the residual, so it -> 0.

    All three terms are per-event, as SubsetCriterion normalises them: the eps-insensitive
    residual, log(2 eps + 2 b) rather than log(2 b), and the Gamma prior on 1/b.
    """
    from alignbeat.phase_criterion import EPS, PRECISION_PRIOR_ALPHA

    def b_star(residual, guarded):
        b = np.logspace(-7, -1, 20000)
        if guarded:
            r = max(residual - EPS, 0.0)
            loss = r / b + np.log(2 * EPS + 2 * b) + (np.log(b) + EPS / b)
        else:
            loss = residual / b + np.log(2 * b)
        return b[loss.argmin()]

    tiny = 1e-4 * EPS                                  # timing far inside tolerance
    assert b_star(tiny, guarded=False) < 0.02 * EPS, "unguarded b* must chase the residual"
    floored = b_star(tiny, guarded=True)
    assert 0.5 * EPS < floored < 1.5 * EPS, f"guarded b* should sit near eps, got {floored}"
    # ...and the floor holds however good timing gets, which is what bounds 1/b.
    assert abs(b_star(tiny, True) - b_star(1e-8 * EPS, True)) < 1e-9
    print(f"ok: guarded b* floors at {floored / EPS:.2f} eps; unguarded b* tracks the "
          f"residual to zero")


def test_guards_bound_the_timing_terms_weight_on_the_shared_trunk():
    """Why the floor matters: 1/b IS the timing term's weight on the shared features.

    Unguarded, b* is the mean residual, so as timing improves 1/b grows without bound and
    the trunk is reallocated to timing -- measured 22:1 for phase at epoch 4 against
    237:1 for timing at epoch 9. Guarded, 1/b* stops at a constant however good timing
    gets, so the phase branch keeps a fixed share.
    """
    from alignbeat.phase_criterion import EPS

    def timing_weight(residual, guarded):
        b = np.logspace(-8, -1, 20000)
        if guarded:
            r = max(residual - EPS, 0.0)
            loss = r / b + np.log(2 * EPS + 2 * b) + (np.log(b) + EPS / b)
        else:
            loss = residual / b + np.log(2 * b)
        return 1.0 / b[loss.argmin()]

    shrinking = [EPS * f for f in (10.0, 1.0, 1e-2, 1e-4, 1e-6)]
    unguarded = [timing_weight(r, False) for r in shrinking]
    guarded = [timing_weight(r, True) for r in shrinking]

    assert unguarded[-1] / unguarded[0] > 1e5, "unguarded 1/b must diverge"
    assert max(guarded) / min(guarded) < 10.0, f"guarded 1/b must stay bounded: {guarded}"
    assert unguarded[-1] > 100 * max(guarded), (unguarded[-1], max(guarded))
    print(f"ok: as the residual shrinks 1e7x, the timing weight goes "
          f"{unguarded[0]:.0f} -> {unguarded[-1]:.0f} unguarded but stays under "
          f"{max(guarded):.0f} guarded")


def test_criterion_runs_on_labelled_and_beat_only_fragments():
    torch.manual_seed(0)
    crit = PhaseCriterion()
    B, N, M = 2, 32, 8
    phi = torch.rand(B, N, requires_grad=True)
    t = torch.sort(torch.rand(B, N), dim=-1).values
    b = torch.full((B,), 0.002)

    t_true = torch.sort(torch.rand(M)).values
    targets = [
        {"t_true": t_true, "phi_true": torch.tensor([0., .25, .5, .75] * 2)},
        {"t_true": t_true, "phi_true": None},            # beat-only: phase unobserved
    ]
    losses, stats = crit(phi, t, b, targets)
    assert stats["used"] == 2 and stats["skipped"] == 0, stats
    assert torch.isfinite(losses["total"]), losses
    losses["total"].backward()
    assert phi.grad is not None and torch.isfinite(phi.grad).all()
    assert float(phi.grad.abs().sum()) > 0.0, "the phase branch must receive gradient"
    print("ok: the criterion runs and is differentiable on both annotation types")


def test_criterion_skips_infeasible_fragments():
    crit = PhaseCriterion()
    phi, t, b = torch.rand(1, 4), torch.sort(torch.rand(1, 4), dim=-1).values, torch.full((1,), 0.002)
    losses, stats = crit(phi, t, b, [{"t_true": torch.rand(9).sort().values,
                                      "phi_true": None}])
    assert stats["skipped"] == 1 and stats["used"] == 0, stats
    assert torch.isfinite(losses["total"]) and float(losses["total"]) == 0.0
    print("ok: a fragment with more events than candidates is skipped, not fitted")


def test_decode_infers_the_meter_and_nests_downbeats_in_beats():
    # A clean 4/4 phase sequence: the meter is recoverable from phi alone.
    phi = torch.tensor([0.0, .25, .5, .75] * 8)
    t = torch.linspace(0.01, 0.99, 32)
    assert infer_meter(phi) == 4, infer_meter(phi)

    beats, downbeats, meter = decode_events(phi, t)
    assert meter == 4
    assert set(downbeats) <= set(beats), "downbeats must be a subset of beats"
    assert len(downbeats) == 8, downbeats
    print("ok: decode recovers the meter and emits downbeats as a subset of beats")


def test_end_to_end_head_shapes():
    head = PhaseTimeHead(16, num_candidates=20, train_length=160, downsample_stages=3)
    out = head(torch.randn(2, 160, 16), epoch=0)
    assert out["phi_hat"].shape == (2, 20), out["phi_hat"].shape
    assert out["t_hat"].shape == (2, 20) and out["b_e"].shape == (2,)
    assert out["beat"] is None and out["downbeat"] is None
    print("ok: PhaseTimeHead maps a window to N candidates with the shared Downsample")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nall tests passed" if failures == 0 else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
