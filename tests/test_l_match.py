"""L_match, vectorised against a double loop written straight off Algorithm 1.

build_l_match does every (event, candidate) pair at once; the algorithm is an if/else
inside two nested loops. The mixed fragment is the case that distinguishes them: a
branch taken per FRAGMENT instead of per EVENT drops its unlabelled rows.
"""
import pytest
import torch

from alignbeat.constants import CLASS_BEAT, CLASS_UNKNOWN, CLASS_DOWNBEAT
from alignbeat.training.criterion import SubsetCriterion

DATA_PRIOR = {"downbeat": 0.2853, "beat": 0.7147}   # fold-0 pi_data, measured
WINDOW_SECONDS = 30.0   # 1500 frames at 50 fps, the training excerpt these fixtures assume

LABELS = {
    "labelled": [CLASS_DOWNBEAT, CLASS_BEAT, CLASS_BEAT, CLASS_BEAT, CLASS_DOWNBEAT, CLASS_BEAT, CLASS_BEAT, CLASS_BEAT],
    "beat_only": [CLASS_UNKNOWN] * 8,
}


def l_match_loop(criterion, log_probabilities, t_hat, gt_class, gt_time):
    """Algorithm 1 lines 15-23, one (i, j) pair at a time, as the pseudocode reads."""
    l_match = torch.empty(len(gt_class), len(t_hat), dtype=log_probabilities.dtype)
    for i, (c_i, t_i) in enumerate(zip(gt_class, gt_time)):
        for j in range(len(t_hat)):
            # Labelled events pay their class NLL. Beat-only events pay the active
            # event mass when that paper term is enabled.
            if c_i == CLASS_UNKNOWN:
                class_cost = (-float(torch.logsumexp(
                    log_probabilities[j, [CLASS_DOWNBEAT, CLASS_BEAT]], dim=0))
                    if criterion.match_event_cost else 0.0)
            else:
                class_cost = -float(log_probabilities[j, c_i])
            l_match[i, j] = class_cost + criterion.lambda_l1 * abs(float(t_hat[j] - t_i))

    return l_match


@pytest.mark.parametrize("kind", list(LABELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("match_event_cost", [False, True])
def test_matches_the_double_loop(kind, seed, match_event_cost):
    g = torch.Generator().manual_seed(seed)
    log_probabilities = torch.log_softmax(
        torch.randn(25, 3, generator=g, dtype=torch.float64) * 3, dim=-1)
    t_hat = torch.sort(torch.rand(25, generator=g, dtype=torch.float64)).values
    gt_time = torch.sort(torch.rand(8, generator=g, dtype=torch.float64)).values
    gt_class = torch.tensor(LABELS[kind])

    criterion = SubsetCriterion(DATA_PRIOR, WINDOW_SECONDS, meter_prior={4: 1.0},
                                match_event_cost=match_event_cost)
    got = criterion.build_l_match(log_probabilities, t_hat, gt_class, gt_time)
    want = l_match_loop(criterion, log_probabilities, t_hat, gt_class, gt_time)

    assert got.shape == (len(gt_class), len(t_hat)), "L_match must be (M, N) for the DP"
    torch.testing.assert_close(got, want, rtol=1e-14, atol=2e-13,
                               msg=f"max |diff| = {(got - want).abs().max():.3e}")
