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
    "attention_layers": 0,
    "detect_tau": 0.5,    # stage 1: keep candidates whose event mass clears this
}



def split_subset_kwargs(subset_kwargs):
    """Route subset_kwargs to the head and the criterion, plus the detection threshold.

    Membership is read off each consumer's own signature, so adding a parameter to
    SubsetHead or SubsetCriterion routes it without editing a list here. Only the
    default value still has to be written down, in SUBSET_ARCH_DEFAULTS.
    Returns (head_kwargs, detect_tau, criterion_kwargs).
    """
    given = dict(subset_kwargs or {})
    criterion_keys = set(inspect.signature(SubsetCriterion.__init__).parameters)
    head_keys = set(inspect.signature(SubsetHead.__init__).parameters)

    arch = {k: v for k, v in given.items() if k not in criterion_keys}
    unknown = set(arch) - set(SUBSET_ARCH_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown subset_kwargs: {', '.join(sorted(unknown))}")
    arch = {**SUBSET_ARCH_DEFAULTS, **arch}


    head = {k: v for k, v in arch.items() if k in head_keys}
    criterion = {k: v for k, v in given.items() if k in criterion_keys}
    leftover = set(arch) - head_keys - {"detect_tau"}
    if leftover:
        raise TypeError(f"subset_kwargs no consumer takes: {', '.join(sorted(leftover))}")
    return head, arch["detect_tau"], criterion
