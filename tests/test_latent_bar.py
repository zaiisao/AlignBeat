"""The bar ablation: what a beat-only annotation is allowed to contribute.

Three arms differ only in this. Arm 1 discards such a fragment, arm 2 lets it assert
that an event occurred, arm 3 marginalises it over the bar. These tests pin the two
properties the comparison rests on: that arm 2's term is a proper marginal likelihood,
and that arm 3's contains it exactly, so the arms are nested rather than merely similar.
"""
import math

import torch
import torch.nn.functional as F

from alignbeat.constants import CLASS_BEAT, CLASS_DOWNBEAT, CLASS_UNKNOWN
from alignbeat.training.criterion import SubsetCriterion
from alignbeat.training.priors import drop_beat_only_pieces

DATA_PRIOR = {"downbeat": 0.2853, "beat": 0.7147}
METER_PRIOR = {2: 0.0993, 3: 0.0947, 4: 0.7943, 5: 0.001, 6: 0.0093, 8: 0.0015}
WINDOW_SECONDS = 30.0
NUM_CANDIDATES, NUM_EVENTS = 188, 12


def criterion(latent_bar):
    return SubsetCriterion(DATA_PRIOR, WINDOW_SECONDS, gamma=1.0,
                           meter_prior=METER_PRIOR, match_event_cost=True,
                           latent_bar=latent_bar)


def fixture(seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(1, NUM_CANDIDATES, 3)
    t_hat = torch.sort(torch.rand(1, NUM_CANDIDATES)).values
    times = torch.sort(torch.rand(NUM_EVENTS)).values
    return logits, t_hat, times


def beat_only(times):
    return [{"times": times, "classes": torch.full((NUM_EVENTS,), CLASS_UNKNOWN)}]


def labelled(times):
    return [{"times": times, "classes": torch.tensor([0, 1, 1, 1] * (NUM_EVENTS // 4))}]


def test_event_only_term_is_the_negative_log_event_mass():
    """Arm 2 charges -log(1 - p(empty)), computed here from the probabilities directly
    rather than by calling the implementation back."""
    matched_log = F.log_softmax(torch.randn(NUM_EVENTS, 3).double(), dim=-1)
    probabilities = matched_log.exp()
    expected = -sum(math.log(float(probabilities[i, CLASS_DOWNBEAT]
                                   + probabilities[i, CLASS_BEAT]))
                    for i in range(NUM_EVENTS))
    got = float(criterion(latent_bar=False)._event_only_term(matched_log))
    assert abs(got - expected) < 1e-10


def test_arm_two_still_constrains_the_event_mass():
    """The trap this ablation has to avoid: dropping the class term outright would leave
    nothing pushing a matched beat-only candidate off empty, and the baseline would be a
    strawman. The term must be positive and finite, not zero."""
    logits, t_hat, times = fixture()
    losses, _ = criterion(latent_bar=False)(logits, t_hat, beat_only(times))
    assert torch.isfinite(losses["class"])
    assert float(losses["class"]) > 0.0


def test_the_arms_are_nested():
    """Arm 3's term is arm 2's plus a piece depending only on the DB/B split, because the
    head is read raw. Built here from the renormalised probabilities, independently of
    how the criterion computes either."""
    torch.manual_seed(3)
    matched_log = F.log_softmax(torch.randn(NUM_EVENTS, 3).double(), dim=-1)
    crit = criterion(latent_bar=True)
    class_columns = matched_log[:, [CLASS_DOWNBEAT, CLASS_BEAT]]

    scores = crit._log_meter_phase_scores((matched_log[:, CLASS_DOWNBEAT],
                                           matched_log[:, CLASS_BEAT]))
    flat = torch.cat([scores[meter] for meter in scores])
    pi = torch.softmax(flat, 0)
    full = float(crit._beat_only_term(matched_log, pi))

    event_only = float(crit._event_only_term(matched_log))
    normalised = class_columns - torch.logsumexp(class_columns, dim=-1, keepdim=True)
    events = torch.arange(NUM_EVENTS)
    split = []
    for meter in crit.meter_candidates:
        prior = crit.meter_prior.get(meter, 0.0) * (1.0 / meter)
        log_prior = math.log(prior) if prior > 0.0 else -float("inf")
        phases = torch.arange(meter)
        is_downbeat = (((phases[:, None] + events[None, :]) % meter) == 0).double()
        split.append(is_downbeat @ normalised[:, 0]
                     + (1.0 - is_downbeat) @ normalised[:, 1] + log_prior)
    split_term = float(-(pi * torch.cat(split)).sum())

    assert abs(full - (event_only + split_term)) < 1e-12


def test_fisher_identity_holds_for_the_event_only_term():
    """Arm 2 is EM, not a penalty that resembles one: the marginal NLL and the surrogate
    with q frozen at theta_old must share a gradient at theta_old."""
    torch.manual_seed(4)
    logits = torch.randn(NUM_EVENTS, 3, dtype=torch.float64, requires_grad=True)
    marginal = criterion(latent_bar=False)._event_only_term(F.log_softmax(logits, dim=-1))
    marginal_grad, = torch.autograd.grad(marginal, logits)

    with torch.no_grad():
        posterior = F.softmax(logits, dim=-1)[:, [CLASS_DOWNBEAT, CLASS_BEAT]]
        posterior = posterior / posterior.sum(-1, keepdim=True)
    log_probabilities = F.log_softmax(logits, dim=-1)
    surrogate = -(posterior
                  * log_probabilities[:, [CLASS_DOWNBEAT, CLASS_BEAT]]).sum()
    surrogate_grad, = torch.autograd.grad(surrogate, logits)

    assert torch.allclose(marginal_grad, surrogate_grad, atol=1e-12)


def test_a_labelled_fragment_does_not_notice_the_ablation():
    """Only the beat-only branch may move; a fragment carrying downbeats is an ordinary
    cross-entropy either way."""
    logits, t_hat, times = fixture(seed=5)
    on, _ = criterion(latent_bar=True)(logits, t_hat, labelled(times))
    off, _ = criterion(latent_bar=False)(logits, t_hat, labelled(times))
    for key in ("class", "time", "background", "total"):
        assert torch.equal(on[key], off[key]), key


def test_the_bar_actually_changes_a_beat_only_fragment():
    """Guards against the ablation silently not applying, which is how a null result gets
    manufactured."""
    logits, t_hat, times = fixture(seed=6)
    on, _ = criterion(latent_bar=True)(logits, t_hat, beat_only(times))
    off, _ = criterion(latent_bar=False)(logits, t_hat, beat_only(times))
    assert not torch.isclose(on["class"], off["class"])
    assert torch.equal(on["time"], off["time"])
    assert torch.equal(on["background"], off["background"])


class _Dataset:
    def __init__(self, items):
        self.items = items


def test_drop_beat_only_pieces_removes_exactly_the_unannotated():
    items = [{"name": "a", "downbeat_mask": True},
             {"name": "b", "downbeat_mask": False},
             {"name": "c", "downbeat_mask": True},
             {"name": "d", "downbeat_mask": False}]
    dataset = _Dataset(list(items))
    removed_count = drop_beat_only_pieces(dataset)
    assert removed_count == 2
    assert [item["name"] for item in dataset.items] == ["a", "c"]


def test_drop_beat_only_pieces_is_a_no_op_when_everything_is_annotated():
    dataset = _Dataset([{"name": "a", "downbeat_mask": True}])
    assert drop_beat_only_pieces(dataset) == 0
    assert len(dataset.items) == 1
