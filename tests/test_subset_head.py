"""Correctness tests for the order-constrained subset selection head.

Run: python -m pytest tests/test_subset_head.py -q
(or plain `python tests/test_subset_head.py` for a pytest-free run)

The load-bearing test is test_dp_matches_brute_force: Algorithm 1 is checked against
exhaustive enumeration of every order-preserving injection, which is what Proposition
1 claims it computes.
"""
import itertools
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatfcos.subset_head import (  # noqa: E402
    BACKGROUND, BEAT, DOWNBEAT,
    SubsetCriterion, SubsetSelectionHead,
    decode_events, intervals_to_events, monotonic_times,
    subset_select_dp, subset_select_logsumexp, targets_to_events,
)


def brute_force_select(cost):
    """Exhaustive minimisation over every order-preserving injection (equation 4)."""
    M, N = cost.shape
    best_cost, best_sigma = np.inf, None
    for sigma in itertools.combinations(range(N), M):
        total = sum(cost[i, sigma[i]] for i in range(M))
        if total < best_cost:
            best_cost, best_sigma = total, np.array(sigma, dtype=np.int64)
    return best_cost, best_sigma


def test_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(300):
        N = int(rng.integers(1, 10))
        M = int(rng.integers(0, N + 1))
        cost = rng.normal(size=(M, N))
        sigma = subset_select_dp(cost)
        assert len(sigma) == M
        assert np.all(np.diff(sigma) > 0), "sigma must be strictly increasing"
        assert np.all(sigma >= 0) and np.all(sigma < N)
        dp_cost = sum(cost[i, sigma[i]] for i in range(M))
        expected, _ = brute_force_select(cost)
        assert np.isclose(dp_cost, expected), (
            f"trial {trial}: DP {dp_cost} != brute force {expected}")
    print("ok: DP == brute force over 300 random instances")


def test_dp_prefers_obvious_diagonal():
    """A cost matrix with a clear cheap diagonal must recover exactly that diagonal."""
    N, M = 12, 4
    cost = np.ones((M, N)) * 10.0
    truth = [1, 4, 7, 10]
    for i, j in enumerate(truth):
        cost[i, j] = 0.0
    assert list(subset_select_dp(cost)) == truth
    print("ok: DP recovers a planted diagonal")


def test_background_correction_is_selection_invariant_under_uniform_shift():
    """Adding the same constant to every g_j must not change sigma.

    L' = L - gamma*g. Shifting all g by c changes every candidate's corrected cost by
    -gamma*c, and since exactly M pairs are always selected the total shifts by
    -gamma*c*M for *every* sigma. The argmin is therefore unchanged - a sanity check
    that the correction is applied per candidate and not, say, per pair index.
    """
    rng = np.random.default_rng(1)
    M, N = 5, 14
    base = rng.normal(size=(M, N))
    g = rng.normal(size=N)
    gamma = 0.5
    a = subset_select_dp(base - gamma * g[None, :])
    b = subset_select_dp(base - gamma * (g + 3.7)[None, :])
    assert list(a) == list(b)
    print("ok: corrected cost is invariant to a uniform background shift")


def test_background_correction_actually_changes_selection():
    """...but a non-uniform g must be able to change sigma, else it is a no-op."""
    M, N = 1, 3
    base = np.array([[1.0, 0.9, 1.0]])
    without = subset_select_dp(base)
    g = np.array([0.0, 0.0, 10.0])
    with_correction = subset_select_dp(base - 0.5 * g[None, :])
    assert list(without) == [1]
    assert list(with_correction) == [2]
    print("ok: the background correction can change the selection")


def test_infeasible_raises():
    try:
        subset_select_dp(np.zeros((5, 4)))
    except ValueError as exc:
        assert "infeasible" in str(exc)
        print("ok: M > N raises")
        return
    raise AssertionError("expected ValueError for M > N")


def test_monotonicity_invariant():
    """Equation (1) must be strictly increasing at any realistic parameter scale, and
    non-decreasing (= non-crossing, which is what the no-NMS argument needs) even at
    pathological ones where the float32 cumulative sum loses tiny increments."""
    torch.manual_seed(0)
    # The relative floor makes strictness hold even at absurd parameter scales, where
    # a plain softplus cumsum ties in float32 (measured: 46/159 ties at scale 20).
    for scale in (1e-3, 0.1, 1.0, 5.0, 20.0, 200.0, 1000.0):
        r = torch.randn(8, 160) * scale
        t = monotonic_times(r)
        gaps = t[:, 1:] - t[:, :-1]
        assert torch.all(gaps > 0), (
            f"not strictly increasing at scale {scale} (min gap {gaps.min():.3e})")
        assert torch.all(t > 0) and torch.all(t <= 1.0 + 1e-6)
        assert torch.isfinite(t).all()
        assert torch.allclose(t[:, -1], torch.ones(8), atol=1e-6), "t_N must equal 1"

    # r == 0 gives the uniform grid the head is initialised to
    t0 = monotonic_times(torch.zeros(1, 8))
    assert torch.allclose(t0[0], torch.arange(1, 9, dtype=torch.float32) / 8.0, atol=1e-5)
    # all-underflowing r degrades gracefully to that same uniform grid
    t_degenerate = monotonic_times(torch.full((1, 8), -500.0))
    assert torch.allclose(t_degenerate[0], torch.arange(1, 9, dtype=torch.float32) / 8.0,
                          atol=1e-5), "epsilon floor should give the uniform grid, not 0/0"
    assert torch.isfinite(t_degenerate).all()
    print("ok: equation (1) strict at realistic scales, non-crossing everywhere, "
          "graceful when fully underflowed")


def test_logsumexp_dp_bounds_the_min():
    """log Z >= -min_cost: the marginalised objective must upper-bound the single best
    injection's probability mass (equation 13 vs equation 7)."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        N = int(rng.integers(2, 8))
        M = int(rng.integers(1, N + 1))
        cost = rng.normal(size=(M, N))
        sigma = subset_select_dp(cost)
        best = sum(cost[i, sigma[i]] for i in range(M))
        log_z = float(subset_select_logsumexp(torch.tensor(cost)))
        assert log_z >= -best - 1e-6, f"log Z {log_z} < -min cost {-best}"
        # and it must equal brute-force logsumexp exactly
        totals = [sum(cost[i, s[i]] for i in range(M))
                  for s in itertools.combinations(range(N), M)]
        expected = float(torch.logsumexp(-torch.tensor(totals), dim=0))
        assert np.isclose(log_z, expected, atol=1e-6), f"{log_z} != {expected}"
    print("ok: logsumexp DP == brute-force marginal")


def test_targets_to_events_class_exclusivity():
    """Every downbeat is set in both channels by the dataloader; downbeat must win."""
    T = 100
    target = torch.zeros(2, T)
    beats = [10, 20, 30, 40]
    downbeats = [10, 30]
    target[0, beats] = 1
    target[1, downbeats] = 1
    events = targets_to_events(target)
    assert list(events['classes']) == [DOWNBEAT, BEAT, DOWNBEAT, BEAT]
    assert torch.allclose(events['times'], torch.tensor([0.10, 0.20, 0.30, 0.40]))
    assert torch.all(events['times'][1:] > events['times'][:-1]), "times must be sorted"
    empty = targets_to_events(torch.zeros(2, T))
    assert empty['classes'].numel() == 0
    print("ok: target conversion is exclusive, sorted and normalised")


def test_intervals_to_events_is_the_real_data_path():
    """BeatDataset returns make_intervals() output, not the frame grid: (M, 3) rows of
    [start, end, class] with 0 = downbeat-to-downbeat and 1 = beat-to-beat, padded with
    -1 by collater. Events must be recovered exactly, endpoints included."""
    T = 1280
    # downbeats at 100/200/300, beats at 100/140/180/220 (every downbeat is a beat too)
    rows = [[100., 200., 0.], [200., 300., 0.],
            [100., 140., 1.], [140., 180., 1.], [180., 220., 1.]]
    annot = torch.ones(1, 8, 3) * -1
    annot[0, :len(rows)] = torch.tensor(rows)

    events = intervals_to_events(annot, T)[0]
    assert list(events['classes']) == [DOWNBEAT, BEAT, BEAT, DOWNBEAT, BEAT, DOWNBEAT], (
        "downbeat must win where a frame is both; the final interval END must appear")
    expected = torch.tensor([100., 140., 180., 200., 220., 300.]) / T
    assert torch.allclose(events['times'], expected)
    assert torch.all(events['times'][1:] > events['times'][:-1])

    # fully padded fragment -> no events, not a crash
    empty = intervals_to_events(torch.ones(1, 4, 3) * -1, T)[0]
    assert empty['classes'].numel() == 0 and empty['times'].numel() == 0

    # batch form returns one dict per fragment
    batch = intervals_to_events(torch.cat([annot, torch.ones(1, 8, 3) * -1]), T)
    assert len(batch) == 2 and batch[0]['classes'].numel() == 6 and batch[1]['classes'].numel() == 0
    print("ok: interval annotations round-trip to events, endpoints preserved")


def test_intervals_to_events_matches_grid_conversion():
    """The interval path and the grid path must agree on the same underlying data."""
    T = 400
    beat_frames = list(range(20, 380, 17))
    downbeat_frames = beat_frames[::4]
    grid = torch.zeros(2, T)
    grid[0, beat_frames] = 1
    grid[1, downbeat_frames] = 1

    rows = [[float(a), float(b), 0.] for a, b in zip(downbeat_frames, downbeat_frames[1:])]
    rows += [[float(a), float(b), 1.] for a, b in zip(beat_frames, beat_frames[1:])]
    annot = torch.tensor(rows).unsqueeze(0)

    from_grid = targets_to_events(grid, num_frames=T)
    from_intervals = intervals_to_events(annot, T)[0]
    assert torch.equal(from_grid['classes'], from_intervals['classes'])
    assert torch.allclose(from_grid['times'], from_intervals['times'])
    print("ok: interval and grid conversions agree")


def test_criterion_rewards_a_perfect_prediction():
    """A model that puts a confident, exactly-located candidate on every event and
    calls everything else background must score far below a random one."""
    torch.manual_seed(0)
    N = 32
    # times chosen to land exactly on the uniform candidate grid t_hat[k] = (k+1)/N,
    # so a perfect prediction really does have zero residual
    times = torch.tensor([4.0, 8.0, 12.0, 16.0]) / N
    classes = torch.tensor([DOWNBEAT, BEAT, BEAT, BEAT])
    targets = [{'classes': classes, 'times': times}]

    criterion = SubsetCriterion(b_scale=0.05, gamma=0.5)

    # Perfect: uniform grid t_hat = j/N puts candidates exactly on 0.1/0.2/0.3/0.4
    # (indices 2, 5, 8, 11 with N=32 -> 3/32... so instead force r to place them).
    t_hat = torch.linspace(1.0 / N, 1.0, N).unsqueeze(0)
    matched = [int(round(float(t) * N)) - 1 for t in times]
    logits = torch.full((1, N, 3), -6.0)
    logits[0, :, BACKGROUND] = 6.0
    for slot, cls in zip(matched, classes):
        logits[0, slot, BACKGROUND] = -6.0
        logits[0, slot, int(cls)] = 6.0

    good, good_stats = criterion(logits, t_hat, targets)
    bad, _ = criterion(torch.zeros(1, N, 3), t_hat, targets)
    good_loss, bad_loss = float(good['total']), float(bad['total'])
    assert good_loss < bad_loss, (good_loss, bad_loss)
    assert torch.allclose(good['total'], good['class'] + good['time'] + good['background'])
    assert good_stats['residual_mean'] < 1e-3
    assert good_stats['infeasible'] == 0
    print(f"ok: perfect={good_loss:.4f} < uniform={bad_loss:.4f}, components sum to total")


def test_criterion_handles_empty_and_infeasible_fragments():
    N = 16
    logits = torch.zeros(2, N, 3)
    t_hat = torch.linspace(1.0 / N, 1.0, N).unsqueeze(0).repeat(2, 1)
    targets = [
        {'classes': torch.zeros(0, dtype=torch.long), 'times': torch.zeros(0)},
        {'classes': torch.full((N + 5,), BEAT, dtype=torch.long),
         'times': torch.linspace(0, 1, N + 5)},
    ]
    criterion = SubsetCriterion()
    losses, stats = criterion(logits, t_hat, targets)
    assert torch.isfinite(losses['total'])
    assert stats['infeasible'] == 1
    print("ok: empty and over-dense fragments do not crash")


def test_criterion_gradients_flow_only_where_expected():
    """sigma is detached, so gradient must reach the class logits of every candidate
    (matched via the class term, unmatched via the background term) and the regression
    outputs of matched candidates only."""
    N = 16
    torch.manual_seed(0)
    r = torch.randn(1, N, requires_grad=True)
    logits = torch.randn(1, N, 3, requires_grad=True)
    t_hat = monotonic_times(r)
    targets = [{'classes': torch.tensor([BEAT, DOWNBEAT]),
                'times': torch.tensor([0.25, 0.75])}]
    losses, _ = SubsetCriterion(b_scale=0.05)(logits, t_hat, targets)
    losses['total'].backward()
    assert logits.grad is not None and torch.any(logits.grad != 0)
    assert r.grad is not None and torch.any(r.grad != 0)
    assert torch.isfinite(logits.grad).all() and torch.isfinite(r.grad).all()
    print("ok: gradients flow and are finite")


def test_head_shapes_and_monotonicity_end_to_end():
    head = SubsetSelectionHead(feature_size=32, num_candidates=160, level_strides=(8, 4, 2))
    features = [torch.randn(2, 32, 1280), torch.randn(2, 32, 640), torch.randn(2, 32, 320)]
    logits, t_hat = head(features)
    assert logits.shape == (2, 160, 3)
    assert t_hat.shape == (2, 160)
    assert torch.all(t_hat[:, 1:] > t_hat[:, :-1])
    # at initialization the regression head is zeroed -> uniform candidate grid
    assert torch.allclose(t_hat[0], torch.arange(1, 161, dtype=torch.float32) / 160.0, atol=1e-5)
    print("ok: head shapes correct, uniform grid at init")


def test_head_rejects_wrong_length():
    """A strided conv on 1281 frames still emits 160 candidates while silently
    dropping the last frame, so the head must validate the INPUT length."""
    head = SubsetSelectionHead(feature_size=8, num_candidates=160, level_strides=(8, 4, 2))
    try:
        head([torch.randn(1, 8, 1281), torch.randn(1, 8, 641), torch.randn(1, 8, 321)])
    except ValueError as exc:
        assert "truncate" in str(exc) or "expected" in str(exc)
        print("ok: silently-truncating input length is rejected loudly")
        return
    raise AssertionError("expected ValueError for a non-divisible input length")


def test_decode_no_duplicates_and_sorted():
    N = 64
    torch.manual_seed(3)
    logits = torch.randn(N, 3) * 3
    t_hat = monotonic_times(torch.randn(N))
    classes, times, scores = decode_events(logits, t_hat, 0.2, 0.2)
    assert torch.all(times[1:] > times[:-1]), "decoded times must be strictly increasing"
    assert torch.all(classes != BACKGROUND)
    assert len(torch.unique(times)) == len(times), "no duplicate times possible"
    print(f"ok: decode gives {len(times)} strictly ordered, unique detections")


# --------------------------------------------------------------------------------
# Section 7.1: beat-only datasets (SMC), where the B/DB distinction is not annotated
# --------------------------------------------------------------------------------

def test_beat_only_events_use_the_marginal_not_a_fabricated_label():
    """A CLASS_UNKNOWN event must be scored by -log(p(B)+p(DB)), never by pretending
    it is a downbeat. The dataloader hardcodes beat=1 for SMC, which would otherwise
    make every SMC beat a downbeat - actively wrong supervision."""
    from beatfcos.subset_head import CLASS_UNKNOWN
    N = 8
    log_p = torch.log(torch.tensor([[0.25, 0.6, 0.15]]).repeat(N, 1))
    crit = SubsetCriterion(diagnostic_every=0)
    unknown = crit.class_nll(log_p, torch.tensor([CLASS_UNKNOWN]))
    expected = -np.log(0.25 + 0.6)
    assert np.isclose(float(unknown[0, 0]), expected, atol=1e-5), (float(unknown[0, 0]), expected)
    # and a known label still uses its own class
    known = crit.class_nll(log_p, torch.tensor([DOWNBEAT]))
    assert np.isclose(float(known[0, 0]), -np.log(0.25), atol=1e-5)
    print("ok: beat-only events scored by the marginal (eq. 9), labelled ones unchanged")


def test_marginal_is_invariant_to_b_db_split():
    """Equation (9) depends only on p(B)+p(DB) - the paper's stated zero-gradient
    property. Two distributions with the same active mass must score identically."""
    from beatfcos.subset_head import CLASS_UNKNOWN
    crit = SubsetCriterion(diagnostic_every=0)
    a = torch.log(torch.tensor([[0.10, 0.75, 0.15]]))
    b = torch.log(torch.tensor([[0.75, 0.10, 0.15]]))
    ca = crit.class_nll(a, torch.tensor([CLASS_UNKNOWN]))
    cb = crit.class_nll(b, torch.tensor([CLASS_UNKNOWN]))
    assert np.isclose(float(ca), float(cb), atol=1e-6)
    print("ok: marginal is invariant to how active mass splits between B and DB")


def test_beat_only_end_to_end_trains_without_crashing():
    from beatfcos.subset_head import CLASS_UNKNOWN
    N = 32
    torch.manual_seed(0)
    logits = torch.randn(1, N, 3, requires_grad=True)
    r = torch.randn(1, N, requires_grad=True)
    targets = [{'classes': torch.full((6,), CLASS_UNKNOWN, dtype=torch.long),
                'times': torch.linspace(0.1, 0.9, 6)}]
    crit = SubsetCriterion(b_scale=0.05, diagnostic_every=0, beat_only_warmup=0)
    losses, stats = crit(logits, monotonic_times(r), targets)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])
    assert stats['unlabelled_events'] == 6
    assert torch.isfinite(logits.grad).all() and torch.any(logits.grad != 0)
    print(f"ok: beat-only fragment trains end-to-end, {stats['unlabelled_events']} unlabelled events")


def test_log_prob_floor_keeps_dp_cost_finite():
    """A confident model underflows p(background) to 0; the section 7.2 correction then
    subtracts gamma*inf and the DP cannot backtrack. Observed live 2x in 11,200 iters."""
    N = 16
    logits = torch.zeros(1, N, 3)
    logits[0, :, BEAT] = 400.0          # p(background) underflows to exactly 0
    logits[0, :, BACKGROUND] = -400.0
    t_hat = monotonic_times(torch.zeros(1, N))
    targets = [{'classes': torch.tensor([BEAT, BEAT]), 'times': torch.tensor([0.25, 0.75])}]
    losses, stats = SubsetCriterion(diagnostic_every=0)(logits, t_hat, targets)
    assert torch.isfinite(losses['total']), "clamping must keep the loss finite"
    assert stats['infeasible'] == 0, "must not lose the batch"
    print("ok: extreme logits no longer produce a non-finite DP cost")



def test_monotonic_times_survives_overflow_scale_r():
    """softplus(r) ~ r, so an unbounded r overflows the cumulative sum to inf and the
    normalisation returns inf/inf = NaN. That NaN reaches the DP cost through the time
    term (which LOG_PROB_FLOOR does not cover) and killed a real run: training stopped
    at epoch ~74 after 9,749 consecutive skipped batches."""
    for v in (1e30, -1e30, 1e38):
        t = monotonic_times(torch.full((1, 160), v))
        assert torch.isfinite(t).all(), f"non-finite t_hat at r={v}"
        assert torch.all(t[:, 1:] > t[:, :-1]), f"ordering lost at r={v}"
    mixed = torch.zeros(1, 160); mixed[0, 0] = 1e38
    t = monotonic_times(mixed)
    assert torch.isfinite(t).all() and torch.all(t[:, 1:] > t[:, :-1])
    print("ok: equation (1) stays finite and ordered at overflow-scale r")


def test_criterion_skips_non_finite_fragment_without_raising():
    """One bad fragment must not take the whole batch down. subset_select_dp raises on a
    non-finite cost, that propagated to train.py which skipped the entire batch, and once
    it happened every batch the run froze while still reporting scores."""
    N = 32
    torch.manual_seed(0)
    logits = torch.randn(2, N, 3, requires_grad=True)
    t_hat = monotonic_times(torch.randn(2, N)).clone()
    t_hat[0, 5] = float('nan')
    targets = [{'classes': torch.tensor([BEAT, DOWNBEAT]),
                'times': torch.tensor([0.3, 0.7])}] * 2
    losses, stats = SubsetCriterion(b_scale=0.005, diagnostic_every=0)(logits, t_hat, targets)
    losses['total'].backward()
    assert stats['non_finite'] == 1, stats
    assert torch.isfinite(losses['total'])
    assert torch.isfinite(logits.grad).all()
    print("ok: a non-finite fragment is skipped and counted, batch survives")



def test_all_fragments_non_finite_still_backprops():
    """If EVERY fragment in a batch is skipped, the loss must still carry a grad_fn.

    The first version of this guard appended the background term inside the
    torch.no_grad() block, so the term was detached; with every fragment skipped the
    total had no grad_fn and backward() raised "element 0 of tensors does not require
    grad". That killed a run within 195 iterations."""
    crit = SubsetCriterion(b_scale=0.005, diagnostic_every=0)
    logits = torch.randn(2, 32, 3, requires_grad=True)
    t_hat = monotonic_times(torch.randn(2, 32)).clone()
    t_hat[:, 5] = float('nan')                      # every fragment non-finite
    targets = [{'classes': torch.tensor([BEAT, DOWNBEAT]),
                'times': torch.tensor([0.3, 0.7])}] * 2
    losses, stats = crit(logits, t_hat, targets)
    assert stats['non_finite'] == 2, stats
    assert losses['total'].grad_fn is not None, "loss must remain differentiable"
    losses['total'].backward()
    assert torch.isfinite(logits.grad).all()
    print("ok: an all-skipped batch stays differentiable instead of raising")


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nall tests passed" if failures == 0 else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
