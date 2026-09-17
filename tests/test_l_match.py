"""L_match, vectorised against a double loop written straight off Algorithm 1.

build_l_match does every (event, candidate) pair at once; the algorithm is an if/else
inside two nested loops. The mixed fragment is the case that distinguishes them: a
branch taken per FRAGMENT instead of per EVENT drops its unlabelled rows.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            # Line 18 charges the observed class's NLL; line 20 has no class to charge.
            class_cost = (0.0 if c_i == CLASS_UNKNOWN
                          else -float(log_probabilities[j, c_i])) 
            l_match[i, j] = class_cost + criterion.lambda_l1 * abs(float(t_hat[j] - t_i))

    return l_match


@pytest.mark.parametrize("kind", list(LABELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_double_loop(kind, seed):
    g = torch.Generator().manual_seed(seed)
    log_probabilities = torch.log_softmax(
        torch.randn(25, 3, generator=g, dtype=torch.float64) * 3, dim=-1)
    t_hat = torch.sort(torch.rand(25, generator=g, dtype=torch.float64)).values
    gt_time = torch.sort(torch.rand(8, generator=g, dtype=torch.float64)).values
    gt_class = torch.tensor(LABELS[kind])

    criterion = SubsetCriterion(DATA_PRIOR, WINDOW_SECONDS, meter_prior={4: 1.0})
    got = criterion.build_l_match(log_probabilities, t_hat, gt_class, gt_time)
    want = l_match_loop(criterion, log_probabilities, t_hat, gt_class, gt_time)

    assert got.shape == (len(gt_class), len(t_hat)), "L_match must be (M, N) for the DP"
    assert torch.equal(got, want), f"max |diff| = {(got - want).abs().max():.3e}"
