"""Algorithm 3: does the metrical decode emit what section 3.1 promises?

3.1's whole argument for a second stage is a CONSTRUCTION guarantee, not a measured
improvement: every emitted label is c_i(omega_hat, L_hat) for one jointly-chosen
hypothesis, so an invalid pattern -- two downbeats with no beat between them, downbeat
gaps that disagree with each other -- cannot be produced at all. A guarantee is worth
asserting rather than hoping for, so that is what these check, on random logits where
the per-candidate rule violates it constantly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.classes import BACKGROUND, BEAT, DOWNBEAT
from alignbeat.criterion import SubsetCriterion
from alignbeat.decode import decode_events, decode_events_metrical
from alignbeat.head import monotonic_times

DATA_PRIOR = {"downbeat": 0.28532, "beat": 0.71468}
METER_PRIOR = {2: 0.09928, 3: 0.05, 4: 0.7943, 5: 0.01, 6: 0.03, 8: 0.01642}


def criterion():
    return SubsetCriterion(data_prior=DATA_PRIOR, meter_prior=METER_PRIOR)


def head_output(N=188, seed=0, event_bias=3.0):
    """Random logits, biased so a decent share of candidates clear the detector."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, 3, generator=g)
    logits[:, BACKGROUND] -= event_bias
    return logits, monotonic_times(torch.randn(N, generator=g))


def downbeat_gaps(classes):
    pos = (classes == DOWNBEAT).nonzero(as_tuple=False).flatten().numpy()
    return np.diff(pos)


@pytest.mark.parametrize("seed", range(8))
def test_output_is_a_single_meter_and_phase(seed):
    """The guarantee itself: downbeat gaps are all equal, i.e. one L throughout."""
    classes, _times, _s = decode_events_metrical(*head_output(seed=seed), criterion())
    gaps = downbeat_gaps(classes)
    if gaps.size:
        assert len(set(gaps.tolist())) == 1, f"mixed downbeat gaps: {sorted(set(gaps))}"


@pytest.mark.parametrize("seed", range(8))
def test_no_two_adjacent_downbeats(seed):
    """3.1's own example of an output no (omega, L) with L > 1 could ever produce."""
    classes, _t, _s = decode_events_metrical(*head_output(seed=seed), criterion())
    gaps = downbeat_gaps(classes)
    assert not (gaps == 1).any(), "two downbeats with no beat between them"


@pytest.mark.parametrize("seed", range(8))
def test_every_emitted_class_is_an_event(seed):
    """Stage 2 labels events; BACKGROUND must never survive into the output."""
    classes, _t, _s = decode_events_metrical(*head_output(seed=seed), criterion())
    assert (classes != BACKGROUND).all()


def test_the_per_candidate_rule_violates_what_this_guarantees():
    """The control. If decode_events also never produced an invalid pattern on this
    input, the tests above would pass vacuously and prove nothing about Algorithm 3."""
    violations = 0
    for seed in range(8):
        classes, _t, _s = decode_events(*head_output(seed=seed), tau=0.2)
        gaps = downbeat_gaps(classes.cpu())
        if gaps.size and ((gaps == 1).any() or len(set(gaps.tolist())) > 1):
            violations += 1
    assert violations > 0, "per-candidate argmax produced no invalid pattern to fix"


def test_detection_is_on_event_mass_not_the_winning_class():
    """Line 5 keeps j when 1 - p(empty) >= tau. A candidate split 0.4/0.4/0.2 across
    DB/B/empty is an event by that rule (0.8 >= 0.5) though its best class is only 0.4,
    which decode_events at the same tau would drop. Getting this backwards would make
    Algorithm 3 silently more conservative than the spec."""
    logits = torch.log(torch.tensor([[0.4, 0.4, 0.2]])).repeat(8, 1)
    t_hat = monotonic_times(torch.zeros(8))
    classes, times, _s = decode_events_metrical(logits, t_hat, criterion(), tau=0.5)
    assert times.numel() == 8, "line 5 should keep every candidate here"
    dropped, _t, _s = decode_events(logits, t_hat, tau=0.5)
    assert dropped.numel() == 0, "control: the per-candidate rule drops them all"


def test_no_detections_returns_empty():
    """A window the detector rejects entirely must decode to nothing, not crash."""
    logits, t_hat = head_output(seed=1, event_bias=-20.0)
    classes, times, scores = decode_events_metrical(logits, t_hat, criterion(), tau=0.5)
    assert classes.numel() == 0 and times.numel() == 0 and scores.numel() == 0


def test_too_few_events_for_any_meter_still_decodes():
    """Below the smallest candidate meter no hypothesis exists; the events are still
    real detections and must be emitted, not dropped."""
    crit = criterion()
    logits = torch.full((188, 3), -10.0)
    logits[:, BACKGROUND] = 10.0                          # everything is background...
    logits[:2] = torch.tensor([2.0, 1.0, -10.0])          # ...except exactly two events
    classes, times, _s = decode_events_metrical(logits, monotonic_times(torch.zeros(188)),
                                                crit, tau=0.5)
    assert times.numel() == 2
    assert (classes != BACKGROUND).all()


def test_pattern_matches_the_hypothesis_that_won():
    """The emitted labels must be the argmax hypothesis's own pattern, not a marginal
    or a re-derivation. Checked by scoring the returned pattern back through the
    criterion and confirming nothing beats it."""
    crit = criterion()
    logits, t_hat = head_output(seed=5)
    log_p = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    keep = (1.0 - torch.softmax(logits, -1)[:, BACKGROUND]) >= 0.5
    resolved = crit.infer_pattern(p[keep])
    assert resolved is not None
    classes, omega, meter = resolved

    blocks = crit._log_meter_phase_scores(crit._class_log_posterior(log_p[keep]))
    best = max(float(blocks[L].max()) for L in blocks)
    assert abs(float(blocks[meter][omega]) - best) < 1e-6, "returned a non-argmax pair"

    i0 = torch.arange(int(keep.sum()))
    assert torch.equal(classes, torch.where(((omega + i0) % meter) == 0,
                                            torch.full_like(i0, DOWNBEAT),
                                            torch.full_like(i0, BEAT)))


# ---------------------------------------------------------------------------
# Recovery: plant a known (omega, L) and check Algorithm 3 gets it back. The
# guarantee tests above only prove the output is SOME valid pattern; these prove
# it is the RIGHT one, which is a different claim.
# ---------------------------------------------------------------------------

def planted(M, omega, meter, confidence=0.9, N=188, seed=0):
    """Head output whose class probabilities encode c_i(omega, meter) at `confidence`,
    on the first M candidates, with the rest firmly background."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((N, 3), -20.0)
    logits[:, BACKGROUND] = 20.0
    i0 = torch.arange(M)
    is_db = ((omega + i0) % meter) == 0
    p_db = torch.where(is_db, confidence, 1.0 - confidence)
    probs = torch.stack([p_db, 1.0 - p_db, torch.full((M,), 1e-6)], dim=1)
    logits[:M] = torch.log(probs / probs.sum(1, keepdim=True))
    return logits, monotonic_times(torch.randn(N, generator=g))


@pytest.mark.parametrize("meter", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("omega", [0, 1, 2])
def test_recovers_the_planted_meter_and_phase(meter, omega):
    """Unambiguous evidence: the argmax must land on exactly what was planted."""
    if omega >= meter:
        pytest.skip("phase must be below the meter")
    M = 8 * meter
    logits, t_hat = planted(M, omega, meter)
    resolved = criterion().infer_pattern(torch.softmax(logits, dim=-1)[:M])
    assert resolved is not None
    classes, got_omega, got_meter = resolved
    assert (got_meter, got_omega) == (meter, omega)
    i0 = torch.arange(M)
    assert torch.equal(classes == DOWNBEAT, ((omega + i0) % meter) == 0)


@pytest.mark.parametrize("meter,omega", [(4, 0), (4, 2), (3, 1), (6, 5)])
def test_end_to_end_decode_recovers_the_pattern(meter, omega):
    """The same, through decode_events_metrical: detection, ordering and stage 2
    together must return the planted downbeat positions."""
    M = 8 * meter
    logits, t_hat = planted(M, omega, meter)
    classes, times, _s = decode_events_metrical(logits, t_hat, criterion(), tau=0.5)
    assert times.numel() == M, "stage 1 should detect exactly the planted events"
    i0 = torch.arange(M)
    assert torch.equal(classes == DOWNBEAT, ((omega + i0) % meter) == 0)


def test_recovery_survives_a_wrong_event():
    """One event's class evidence flipped must not move the argmax: line 17's product
    is over the whole fragment, so a single dissenting event should be outvoted."""
    M, meter, omega = 32, 4, 1
    logits, t_hat = planted(M, omega, meter)
    logits[5] = logits[5].flip(0) if False else torch.log(
        torch.tensor([0.9, 0.1, 1e-6]) / torch.tensor([0.9, 0.1, 1e-6]).sum())
    _c, got_omega, got_meter = criterion().infer_pattern(
        torch.softmax(logits, dim=-1)[:M])
    assert (got_meter, got_omega) == (meter, omega)


def test_first_downbeat_index_matches_the_spec():
    """The PDF states the first downbeat falls at i = (L - omega) mod L, not at omega.
    A sign slip in the phase convention would pass every validity test above and fail
    only here."""
    M, meter, omega = 24, 4, 3
    logits, _t = planted(M, omega, meter)
    classes, _o, _m = criterion().infer_pattern(torch.softmax(logits, -1)[:M])
    first = int((classes == DOWNBEAT).nonzero()[0])
    assert first == (meter - omega) % meter == 1
