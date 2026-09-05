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
from alignbeat.phase_decode import decode, decode_events, decode_fragment, grid_tolerance, infer_meter
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
    # Pinned: a constant, nothing to train. Released: the same VALUE (continuity is the
    # point) but now on the graph, so the scale head can move from here.
    assert not early.requires_grad
    assert late.requires_grad and late.grad_fn is not None
    late.sum().backward()
    assert head.precision_head.net[-1].bias.grad is not None
    print("ok: b_e is pinned at b_0 through the warm start and trainable afterwards")


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


def test_l_agree_interpolates_around_the_circle_not_across_it():
    """The bug that actually corrupted the supervision.

    An unmatched candidate is taught the phase interpolated between the two annotated
    events bracketing it. Phase advances FORWARD through a bar, so the step from one
    event to the next is (phi_i1 - phi_i) mod 1: at a bar line, 0.75 -> 0.00 is a forward
    step of 0.25, through 0.875. Interpolating the raw difference runs backwards through
    0.375 instead -- a circular error of 0.5, the largest possible -- on every wrapping
    interval, which is 22% of them at L=4 and 48% at L=2.

    Those candidates are N-M of N, the majority of the phase supervision, and they
    contradict what the matched events are taught one candidate away.
    """
    t_true = torch.linspace(0.05, 0.95, 12)
    phi_true = torch.tensor([0., .25, .5, .75] * 3)
    # A candidate at the midpoint of a wrapping interval: between phi=0.75 and phi=0.00.
    wrap = [k for k in range(11) if float(phi_true[k + 1]) < float(phi_true[k])]
    assert wrap, "fixture must contain a bar line"
    k = wrap[0]
    midpoint = (t_true[k] + t_true[k + 1]) / 2

    zero = torch.zeros(1)
    d = float(PhaseCriterion.l_agree(zero, t_true, phi_true, midpoint.reshape(1)))
    # l_agree returns the distance from a prediction of 0.0, so the target it used is
    # whichever of d / 1-d lies in the interval; either way it must be 0.875, not 0.375.
    assert abs(d - 0.125) < 1e-5, (
        f"target at a bar line's midpoint is {1 - d if d > 0.5 else d:.3f} away from 0; "
        f"expected 0.875 (distance 0.125), got distance {d:.3f}")

    # ...and away from a bar line, plain interpolation is unchanged.
    flat = [k for k in range(11) if float(phi_true[k + 1]) > float(phi_true[k])][0]
    mid2 = (t_true[flat] + t_true[flat + 1]) / 2
    want = (float(phi_true[flat]) + float(phi_true[flat + 1])) / 2
    got = float(PhaseCriterion.l_agree(torch.tensor([want]), t_true, phi_true,
                                       mid2.reshape(1)))
    assert got < 1e-5, f"non-wrapping interval should interpolate linearly, off by {got}"
    print("ok: l_agree takes the circular short path, so bar lines are not taught "
          "antipodal targets")


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


def test_warm_start_release_is_continuous():
    """The reference released b_e at softplus(random init) ~ 0.69, 280x b_0."""
    torch.manual_seed(0)
    head = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8, warmup_epochs=5)
    x = torch.randn(3, 16, 20)
    _, _, pinned = head(x, epoch=5)
    _, _, released = head(x, epoch=6)
    assert torch.allclose(pinned, released, atol=1e-5), (pinned, released)
    assert abs(float(released[0]) - head.b_0) < 1e-5, (float(released[0]), head.b_0)
    print("ok: b_e is identical the epoch before and after the warm start releases it")


def test_head_widths_match_the_reference():
    head = PhaseSelectionHead(feature_size=16, reduced_dim=8)
    for branch in (head.phase_head, head.regression_head, head.precision_head):
        assert branch.net[0].out_features == 256, branch.net[0].out_features
    print("ok: every head branch is 256 wide, as in the reference and SubsetSelectionHead")


def test_positional_encoding_lets_the_phase_branch_count():
    """Identical candidate features must still yield DIFFERENT phases under index pos."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1).expand(1, 16, 12).contiguous()      # 12 identical candidates
    blind = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8,
                               phase_attention_pos="none")
    torch.manual_seed(0)
    indexed = PhaseSelectionHead(feature_size=16, reduced_dim=8, hidden_size=8,
                                 phase_attention_pos="index")
    phi_blind, _, _ = blind(x)
    phi_index, _, _ = indexed(x)
    assert float(phi_blind.std()) < 1e-5, "with no position, identical inputs give identical phase"
    assert float(phi_index.std()) > 1e-3, "with index position, identical inputs must differ"
    print("ok: permutation-equivariant attention cannot count; index encoding can")


def test_prior_is_per_event_so_the_floor_does_not_drift_with_M():
    """Added once per fragment the floor was eps/sqrt(M): 9 ms at M=60, not 50 ms."""
    from alignbeat.phase_criterion import EPS
    crit = PhaseCriterion(timing_guards=True)
    floors = {}
    for M in (2, 10, 60):
        t_true = torch.linspace(0.1, 0.9, M)
        t_hat = t_true + 1e-5 * EPS                              # inside the dead zone
        sigma = torch.arange(M)
        phi = torch.zeros(M)
        best_b, best = None, float("inf")
        for b in torch.logspace(-6, -1, 2000):
            terms = crit._m_step(sigma, phi, t_true, t_hat, phi, b, t_hat.new_zeros(0))
            v = float(terms["time"] + terms["scale"])
            if v < best:
                best, best_b = v, float(b)
        floors[M] = best_b / EPS
    for M, f in floors.items():
        assert abs(f - 0.707) < 0.05, f"M={M}: b*/eps = {f:.3f}, expected ~0.707"
    print(f"ok: guarded b*/eps = {floors} -- independent of M")


def test_decode_threshold_rejects_something_at_every_meter():
    """tau=0.2 absolute exceeded 1/(2L) for L>=3: every candidate fired, always."""
    torch.manual_seed(0)
    phi = torch.rand(10000)
    t = torch.linspace(0.0001, 1.0, 10000)
    for L in (2, 3, 4, 6):
        _, _, accepted = decode(phi, t, L)
        frac = len(accepted) / 10000
        assert 0.0 < frac < 1.0, f"L={L}: accepted fraction {frac}"
        # tau is a fraction of the half-spacing, so the accepted fraction IS tau
        assert abs(frac - 0.4) < 0.03, f"L={L}: accepted {frac:.3f}, expected ~0.40"
        on_grid = torch.arange(L, dtype=torch.float) / L
        _, _, all_in = decode(on_grid, torch.linspace(0.1, 0.9, L), L)
        assert len(all_in) == L, "phases exactly on the grid must all be accepted"
    assert grid_tolerance(1.0, 4) == 0.125, grid_tolerance(1.0, 4)
    print("ok: the decode threshold now rejects 60% of uniform phase at every meter")


def test_decode_fragment_matches_decode_events():
    phi = torch.tensor([0.0, .25, .5, .75] * 8) + 0.01 * torch.randn(32)
    t = torch.linspace(0.01, 0.99, 32)
    beats, downbeats, _ = decode_events(phi % 1.0, t)
    classes, times, scores = decode_fragment(phi % 1.0, t)
    assert times.tolist() == beats, "same accepted times in the same order"
    assert sorted(times[classes == 0].tolist()) == sorted(downbeats)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    print("ok: the stitcher's per-fragment decode agrees with the excerpt decode")


def test_beat_only_phase_blind_mode_sends_no_phase_gradient():
    """A perfect 3/4 prediction on a beat-only fragment: lattice mode still penalises it."""
    torch.manual_seed(0)
    M, N = 12, 40
    t_true = torch.linspace(0.05, 0.95, M)
    t_hat = t_true.clone()
    phi_hat = torch.tensor([(k % 3) / 3 for k in range(M)]).repeat_interleave(1)
    full_phi = torch.zeros(N); full_phi[torch.arange(M)] = phi_hat
    full_phi = full_phi.requires_grad_(True)
    full_t = torch.sort(torch.cat([t_hat, torch.rand(N - M) * 0.98 + 0.01])).values
    b = torch.full((1,), 0.002)
    grads = {}
    for mode in ("lattice", "phase_blind"):
        crit = PhaseCriterion(beat_only_mode=mode)
        losses, _ = crit(full_phi.unsqueeze(0), full_t.unsqueeze(0), b,
                         [{"t_true": t_true, "phi_true": None}])
        if not losses["total"].requires_grad:      # nothing in the loss touches phi
            grads[mode] = 0.0
            continue
        g, = torch.autograd.grad(losses["total"], full_phi, allow_unused=True)
        grads[mode] = 0.0 if g is None else float(g.abs().sum())
    assert grads["phase_blind"] == 0.0, grads
    assert grads["lattice"] > 0.0, grads
    print(f"ok: beat-only gradient on phi -- lattice {grads['lattice']:.2f}, phase_blind 0")


def test_phase_in_b_puts_phase_and_timing_on_one_scale():
    crit = PhaseCriterion(phase_in_b=True)
    ref = PhaseCriterion(phase_in_b=False)
    t_true = torch.tensor([0.3, 0.6]); phi_true = torch.tensor([0.0, 0.5])
    t_hat = torch.linspace(0.1, 0.9, 9); phi_hat = torch.rand(9)
    b = torch.tensor(0.002)
    a = crit.match_cost(t_true, phi_true, t_hat, phi_hat, b, phase_blind=False)
    r = ref.match_cost(t_true, phi_true, t_hat, phi_hat, b, phase_blind=False)
    from alignbeat.phase_criterion import EPS
    residual = (t_true[:, None] - t_hat[None, :]).abs().sub(EPS).clamp(min=0)
    phase = 3.0 * circ_dist(phi_true[:, None], phi_hat[None, :])
    assert torch.allclose(a, (residual + phase) / b)
    assert torch.allclose(r, residual / b + phase)
    print("ok: phase_in_b divides the phase distance by b in the matching cost")


def test_quantize_targets_keeps_downbeats():
    """Rounding beats but not downbeats made np.isin miss every one of them."""
    import numpy as np
    from beat_this.model.pl_module import PLBeatThis
    m = PLBeatThis(head_type="phase", num_candidates=188, transformer_dim=64, n_layers=2,
                   downsample_stages=3, quantize_targets=True, max_epochs=1)
    beats = 0.5 + 0.513 * np.arange(40)                       # off the 50 fps grid
    batch = {"spect": torch.zeros(1, 1500, 128), "truth_beat": torch.zeros(1, 1500),
             "truth_orig_beat": [beats.tobytes()],
             "truth_orig_downbeat": [beats[::4].tobytes()],
             "downbeat_mask": torch.tensor([True])}
    phi = m._phase_targets(batch)[0]["phi_true"]
    assert phi is not None
    assert int((phi == 0).sum()) == 10, f"expected 10 downbeats, got {int((phi == 0).sum())}"
    assert len(torch.unique(phi)) == 4, torch.unique(phi)
    print("ok: --quantize_targets moves downbeats to the same grid as beats")


def test_predict_step_routes_the_phase_arm():
    from beat_this.model.pl_module import PLBeatThis
    # 3 halvings of a 400-frame window: 400 -> 200 -> 100 -> 50, so N must be 50.
    m = PLBeatThis(head_type="phase", num_candidates=50, transformer_dim=64, n_layers=2,
                   downsample_stages=3, train_length=400, max_epochs=1)
    m.eval()
    T = 900
    beats = np.arange(0.5, 17.5, 0.5)
    batch = {"spect": torch.randn(1, T, 128), "padding_mask": torch.ones(1, T, dtype=torch.bool),
             "truth_beat": torch.zeros(1, T), "truth_downbeat": torch.zeros(1, T),
             "downbeat_mask": torch.tensor([True]),
             "truth_orig_beat": [beats.tobytes()], "truth_orig_downbeat": [beats[::4].tobytes()],
             "dataset": ["synthetic"], "spect_path": ["synthetic"]}
    metrics, _, _, _ = m.predict_step(batch, 0, chunk_size=400)
    assert isinstance(metrics, dict) and metrics, metrics
    print("ok: predict_step reaches the phase decoder through the shared stitcher")


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
