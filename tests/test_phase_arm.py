"""Correctness tests for the continuous-phase arm (--head_type phase).

Several of these pin a KNOWN DEFECT rather than a guarantee. The phase objective as
ported cannot learn: its target distribution is rotation-symmetric and its loss is an L1
distance on the circle, so every constant output is a stationary point. Measured on the
standalone arm at epoch 9 of fold 0, the trained head scored a mean circular phase error
of 0.2488 against 0.2484 for a single fixed constant and 0.2434 for the best constant --
worse than a constant, and dbF was exactly 0.0000 on every dataset.

The tests below encode that, so a future change to the objective is measured against it
instead of rediscovering it after another eight-fold run.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.dp import subset_select_dp
from alignbeat.phase_criterion import PhaseCriterion, circ_dist, phases_from_downbeats
from alignbeat.phase_decode import decode_events, infer_meter
from alignbeat.phase_head import PhaseSelectionHead, PhaseTimeHead


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


def test_the_phase_loss_is_flat_over_constant_predictions():
    """THE DEFECT. Any constant phi_hat is a stationary point of the phase term.

    A 4/4 corpus puts the targets uniformly on {0, 1/4, 1/2, 3/4}, a set invariant under
    rotation by 1/4. circ_dist is an L1 distance, so E[d(c, phi)] is the SAME for every
    constant c and its gradient vanishes. A head that has collapsed to a constant has no
    gradient out of it, which is why the ported arm scored dbF = 0.
    """
    targets = torch.tensor([0.0, .25, .5, .75] * 8)
    costs = torch.tensor([float(circ_dist(targets, torch.full_like(targets, c / 400)).mean())
                          for c in range(400)])
    assert float(costs.max() - costs.min()) < 1e-6, (
        f"the landscape over constants varies by {float(costs.max() - costs.min())}; if "
        f"this now FAILS the objective has been changed and this test should be replaced "
        f"by one asserting the trained head beats the best constant")

    c = torch.tensor(0.31, requires_grad=True)
    circ_dist(targets, c.expand_as(targets)).mean().backward()
    assert abs(float(c.grad)) < 1e-6, f"gradient at a constant is {float(c.grad)}, expected 0"
    print("ok: the phase loss is provably flat over constants -- the collapse is structural")


def test_the_timing_scale_has_no_floor():
    """THE SECOND DEFECT. This arm dropped three guards SubsetCriterion has.

    SubsetCriterion's normaliser is log(2 eps + 2 b), whose 2 eps bounds it below; this
    arm's is log(2 b), which is not bounded, and there is no Gamma prior on 1/b either.
    The loss can therefore improve without limit by shrinking b, which is what makes its
    training curve uninformative about whether anything is being learned.
    """
    from alignbeat.criterion import EPS

    b = torch.logspace(-8, -2, 40)
    assert float((2 * b).log().min()) < np.log(2 * EPS) - 5.0, (
        "log(2b) should run far below the floor eps would have imposed")
    guarded = (2 * EPS + 2 * b).log()
    assert float(guarded.min()) > float(np.log(2 * EPS)) - 1e-6, "eps must floor it"
    print(f"ok: log(2b) is unbounded below, while log(2 eps + 2b) floors at "
          f"{float(np.log(2 * EPS)):.2f}")


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
