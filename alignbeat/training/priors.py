"""Per-fold priors read off the training split, and the annotation-coverage knob.

These live here rather than on BeatDataModule so beat_this/ stays at upstream. Each
takes the training dataset it measures, and downbeat_dropout is applied to the built
item list rather than threaded through the loader.
"""
import collections
import hashlib
import pathlib

import numpy as np


def apply_downbeat_dropout(train_dataset, rate):
    """Pretend a fraction of the downbeat-annotated pieces were never annotated.

    Both heads read the same downbeat_mask -- the dense arm masks its downbeat loss, the
    subset arm routes the fragment to the beat-only branch -- so one knob drives the
    comparison. Deterministic in the piece name, NOT resampled per epoch: a per-epoch
    draw would leak every piece's downbeats eventually and measure augmentation rather
    than coverage. Mutates the item dicts in place; oversampling repeats references to
    the same dict, so every copy of a piece moves together.
    """
    if not rate:
        return 0
    dropped = set()
    for item in train_dataset.items:
        # The loader hashed the dataset-relative item name; spect_path is that name with
        # "track.npy" appended, so strip it or the 30% falls on different pieces.
        name = str(pathlib.PurePath(item["spect_path"]).parent)
        if not item["downbeat_mask"]:
            continue
        digest = hashlib.md5(name.encode()).hexdigest()[:8]
        if int(digest, 16) / 0xFFFFFFFF < rate:
            item["downbeat_mask"] = False
            dropped.add(name)
    return len(dropped)


def get_train_class_prior(train_dataset, weight="stream"):
    """pi_data: the DB:B balance in the data the class head is trained on, this fold.

    Section 1.3's calibration assumption is that cross-entropy drives p_hat_j toward
    the training set's own posterior, whose implicit prior is this. Measured from
    train_dataset.items -- fold-excluded and oversampled, i.e. what the head
    actually saw -- so no validation or test track informs a prior the model uses.

    Only items carrying downbeat annotations are counted: those are the fragments
    trained by plain cross-entropy against an observed class. Beat-only fragments
    train the same head through the EM surrogate with soft r_i weights that move
    during training, so the head's true implicit prior is a mixture and this is an
    approximation of it -- a close one, labelled items being ~82% of the corpus.

    weight: "stream" counts items as the loader presents them, with the length-based
            oversampling; "track" counts each distinct piece once.
    """
    if weight not in ("stream", "track"):
        raise ValueError(f"weight must be 'stream' or 'track', got {weight!r}")
    downbeats = beats = 0
    seen = set()
    for item in train_dataset.items:
        if not item["downbeat_mask"]:
            continue
        if weight == "track":
            key = str(item["spect_path"])
            if key in seen:
                continue
            seen.add(key)
        values = np.asarray(item["beat_value"]).astype(int)
        downbeats += int((values == 1).sum())
        beats += len(values)
    if not beats:
        raise ValueError("no labelled training beats; cannot estimate pi_data")
    share = downbeats / beats
    return {"downbeat": share, "beat": 1.0 - share}


def get_train_meter_prior(train_dataset, candidates=(2, 3, 4, 5, 6, 8),
                          weight="stream", floor=1e-4):
    """pi_M(L): the meter distribution of THIS fold's own training split.

    weight:     "stream" counts train_dataset.items as they are, i.e. with the
                length-based oversampling the head actually sees; "track" counts
                each distinct piece once, as METER_DISTRIBUTION.md does.
    floor:      every candidate's share is raised to at least this before
                renormalising, so a meter absent from a fold is improbable rather
                than impossible (an unfloored zero gives log pi_M = -inf, which
                removes it from the support outright).

    Returns {L: probability} over `candidates`, summing to 1.
    """
    if weight not in ("stream", "track"):
        raise ValueError(f"weight must be 'stream' or 'track', got {weight!r}")

    counts = collections.Counter()
    seen = set()
    n_labelled = n_outside = 0
    for item in train_dataset.items:
        if not item["downbeat_mask"]:
            continue
        if weight == "track":
            key = str(item["spect_path"])
            if key in seen:
                continue
            seen.add(key)
        downbeats = np.flatnonzero(np.asarray(item["beat_value"]).astype(int) == 1)
        if len(downbeats) < 2:
            continue
        meter = int(np.bincount(np.diff(downbeats)).argmax())
        n_labelled += 1
        if meter in candidates:
            counts[meter] += 1
        else:
            n_outside += 1

    if not counts:
        raise ValueError(
            "no training track has a modal meter among "
            f"{tuple(candidates)}; cannot estimate pi_M")

    observed = sum(counts[L] for L in candidates)
    floored = {int(L): max(floor, counts[L] / observed) for L in candidates}
    total = sum(floored.values())
    prior = {L: p / total for L, p in floored.items()}
    return prior


def get_train_positive_weights(datamodule, widen_target_mask=3):
    """BeatDataModule's own positive weights, guarded for the all-dropped case.

    At --downbeat_dropout 1.0 no piece keeps a downbeat annotation, so the upstream
    ratio divides by zero. The dense head then has no positive downbeat example to
    weight and the subset head never reads this at all, so any finite value is inert;
    1 keeps the loss well defined.
    """
    weights = datamodule.get_train_positive_weights(widen_target_mask=widen_target_mask)
    downbeat = weights["downbeat"]
    if not np.isfinite(downbeat) or downbeat <= 0:
        weights = dict(weights, downbeat=1)
    return weights
