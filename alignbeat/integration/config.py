"""Where the subset head's settings live, and how they reach their three consumers."""
import inspect

from alignbeat.training.criterion import SubsetCriterion
from alignbeat.model.head import SubsetHead


# Architecture and decode settings ride in one `subset_kwargs` dict rather than in
# PLBeatThis's own signature. This is the single source of truth: the keys are the
# allowlist, the values the defaults applied when a checkpoint predates a knob.
SUBSET_ARCH_DEFAULTS = {
    "num_candidates": None,
    "train_length": 1500,
    "downsample_stages": None,
    "stitch_border": None,
    "attention_layers": 0,
    "time_param": "bounded",
    "tau": 0.2,           # decode_events: the winning class's own probability
    "detect_tau": 0.5,    # Algorithm 3 line 5: event mass, 1 - p(empty)
    "decode": "detect",
}

DECODE_RULES = ("argmax", "metrical", "detect")


def split_subset_kwargs(subset_kwargs):
    """Route subset_kwargs to its three consumers: head, Lightning module, criterion.

    Membership is read off each consumer's own signature, so adding a parameter to
    SubsetHead or SubsetCriterion routes it without editing a list here. Only the
    default value still has to be written down, in SUBSET_ARCH_DEFAULTS.
    """
    given = dict(subset_kwargs or {})
    criterion_keys = set(inspect.signature(SubsetCriterion.__init__).parameters)
    head_keys = set(inspect.signature(SubsetHead.__init__).parameters)

    arch = {k: v for k, v in given.items() if k not in criterion_keys}
    unknown = set(arch) - set(SUBSET_ARCH_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown subset_kwargs: {', '.join(sorted(unknown))}")
    arch = {**SUBSET_ARCH_DEFAULTS, **arch}

    if arch["decode"] not in DECODE_RULES:
        raise ValueError(f"decode must be one of {DECODE_RULES}, got {arch['decode']!r}")

    head = {k: v for k, v in arch.items() if k in head_keys}
    module = {k: v for k, v in arch.items() if k not in head_keys}
    criterion = {k: v for k, v in given.items() if k in criterion_keys}
    return head, module, criterion
