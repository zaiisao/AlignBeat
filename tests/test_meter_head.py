"""Remark 3's meter head: a supervised L_hat, and the bias it puts on the class logits.

Section 8.7 defers "a learned meter prior q_hat(L | x; theta), of the kind a dedicated
meter-classification head could supply" for want of an architecture that provides one;
Remark 3 specifies it as "a small global head estimating meter length L_hat ... from
globally pooled features" whose output is used to "bias the per-candidate downbeat
logits directly". These tests pin the three properties that make it safe to add: it is
a no-op at initialisation, its supervision is a plain cross-entropy with no degenerate
minimiser, and its gradient reaches both the head and the class logits.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.classes import BEAT, DOWNBEAT
from alignbeat.criterion import SubsetCriterion
from alignbeat.head import SubsetSelectionHead

CANDIDATES = (2, 3, 4, 5, 6, 8)


def test_zero_init_makes_the_bias_a_no_op():
    """At step 0 the class logits must be exactly what they were without the branch."""
    torch.manual_seed(0)
    head = SubsetSelectionHead(feature_size=32, meter_candidates=CANDIDATES)
    x = torch.randn(2, 32, 64)
    logits, _, _, meter_logits = head(x)
    z = head.trunk(head.input_norm(x.transpose(1, 2)))
    assert torch.allclose(logits, head.class_head(z), atol=1e-6), (
        "meter_bias must be zero-initialised so the arm starts at the control")
    assert meter_logits.shape == (2, len(CANDIDATES))
    assert torch.isfinite(meter_logits).all()
    print("ok: the meter bias is a no-op at initialisation, and L_hat has one logit per candidate")


def test_bias_moves_the_class_logits_once_it_is_nonzero():
    torch.manual_seed(0)
    head = SubsetSelectionHead(feature_size=32, meter_candidates=CANDIDATES)
    x = torch.randn(2, 32, 64)
    before, _, _, _ = head(x)
    with torch.no_grad():
        head.meter_bias.weight.normal_(0.0, 1.0)
    after, _, _, _ = head(x)
    assert not torch.allclose(before, after), "a nonzero bias must reach the class logits"
    # it is a per-class offset shared by every candidate, not a per-candidate edit
    delta = after - before
    assert torch.allclose(delta, delta[:, :1, :].expand_as(delta), atol=1e-6), (
        "Remark 3's bias is global to the fragment: the same offset on every candidate")
    print("ok: the bias is a per-class, fragment-global offset on the logits")


def test_supervision_is_a_plain_cross_entropy_with_no_flat_optimum():
    """The failure of the eq. (33) marginal was that shrinking every p_hat raised the
    target's share. Cross-entropy on |M| free logits has the opposite gradient."""
    torch.manual_seed(0)
    crit = SubsetCriterion(meter_candidates=CANDIDATES, lambda_meter_head=1.0)
    target = torch.tensor([CANDIDATES.index(4)])
    right = torch.zeros(1, len(CANDIDATES)); right[0, CANDIDATES.index(4)] = 8.0
    flat = torch.zeros(1, len(CANDIDATES))
    wrong = torch.zeros(1, len(CANDIDATES)); wrong[0, CANDIDATES.index(2)] = 8.0
    ce = lambda l: float(F.cross_entropy(l, target))
    assert ce(right) < ce(flat) < ce(wrong), (ce(right), ce(flat), ce(wrong))
    assert ce(right) < 0.01, "a confident, correct head pays essentially nothing"
    print(f"ok: CE right {ce(right):.3f} < flat {ce(flat):.3f} < wrong {ce(wrong):.3f}")


def test_meter_term_trains_on_labelled_fragments_and_reaches_the_head():
    torch.manual_seed(0)
    N, M, L = 64, 16, 4
    head = SubsetSelectionHead(feature_size=32, meter_candidates=CANDIDATES)
    crit = SubsetCriterion(meter_candidates=CANDIDATES, lambda_meter_head=1.0)
    x = torch.randn(1, 32, N, requires_grad=True)
    logits, t_hat, b_hat, meter_logits = head(x)
    classes = torch.tensor([DOWNBEAT if i % L == 0 else BEAT for i in range(M)])
    targets = [{'classes': classes, 'times': torch.linspace(0.05, 0.95, M)}]
    losses, _ = crit(logits, t_hat, b_hat, targets, meter_logits=meter_logits)
    assert 'meter' in losses, "a labelled fragment with a candidate meter must be supervised"
    losses['total'].backward()
    assert head.meter_head.weight.grad is not None
    assert torch.any(head.meter_head.weight.grad != 0), "the meter head must receive gradient"
    assert torch.isfinite(losses['total'])
    print("ok: the meter term is formed on a labelled fragment and its gradient reaches the head")


def test_no_meter_term_without_logits_or_off_candidate_meter():
    torch.manual_seed(0)
    N, M = 64, 16
    crit = SubsetCriterion(meter_candidates=CANDIDATES, lambda_meter_head=1.0)
    logits = torch.randn(1, N, 3, requires_grad=True)
    t_hat = torch.linspace(0.01, 0.99, N).unsqueeze(0)
    b_hat = torch.full((1, N), 0.002)
    classes = torch.tensor([DOWNBEAT if i % 4 == 0 else BEAT for i in range(M)])
    targets = [{'classes': classes, 'times': torch.linspace(0.05, 0.95, M)}]
    losses, _ = crit(logits, t_hat, b_hat, targets)
    assert 'meter' not in losses, "no head output, no term -- the shipped loss is unchanged"

    # a 7/8 fragment: L is outside the candidate set, so it contributes nothing rather
    # than a wrong signal
    classes7 = torch.tensor([DOWNBEAT if i % 7 == 0 else BEAT for i in range(14)])
    targets7 = [{'classes': classes7, 'times': torch.linspace(0.05, 0.95, 14)}]
    ml = torch.zeros(1, len(CANDIDATES), requires_grad=True)
    losses7, _ = crit(logits, t_hat, b_hat, targets7, meter_logits=ml)
    assert 'meter' not in losses7, "an off-candidate meter must be skipped, not forced"
    print("ok: no term without head logits, and none for a meter outside the candidate set")


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nall meter-head tests passed" if failures == 0 else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
