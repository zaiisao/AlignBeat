"""Single source of truth for turning collated interval annotations into event times.

Every ad-hoc re-implementation of this in a script has been wrong the same way:
make_intervals emits TWO chains - downbeat intervals (label 0) and beat intervals
(label 1) - and a downbeat appears in both, because dataloader.py writes downbeats
into the beat channel as well. Appending `left` for every label therefore counts each
downbeat twice in the beat reference. Measured on ballroom val: 225 duplicates in 1077
references, 20.9%. mir_eval cannot match a duplicated reference twice, so the extra
copies are unmatchable and the estimate is penalised on recall - which silently
depressed every number produced by train_bt_baseline.py and the /tmp diagnostics,
while beatfcos/beat_eval.py (which does this correctly) was unaffected. The two were
then compared to each other.

Mirrors evaluate_beat_f_measure_subset in beat_eval.py exactly, including the trailing
last-index event appended to each chain.
"""
import numpy as np

LABEL_DOWNBEAT, LABEL_BEAT, LABEL_BEAT_ONLY = 0, 1, 2


def intervals_to_times(annot, to_seconds):
    """(M,3) interval rows -> (beat_times, downbeat_times), both ascending.

    Beat times already INCLUDE downbeats: the beat chain is built from a channel that
    has downbeats set, so label 1 rows cover them. Do not add label 0 rows to beats.
    """
    beats, downbeats = [], []
    last_beat, last_downbeat = None, None
    for row in annot:
        label = int(row[2])
        if label < 0:
            continue  # collater padding
        left, right = int(row[0]), int(row[1])
        if label == LABEL_DOWNBEAT:
            downbeats.append(left * to_seconds)
            if last_downbeat is None or right > last_downbeat:
                last_downbeat = right
        elif label in (LABEL_BEAT, LABEL_BEAT_ONLY):
            # label 2 = beat-only dataset (SMC): certainly a beat, downbeat unlabelled.
            beats.append(left * to_seconds)
            if last_beat is None or right > last_beat:
                last_beat = right
    if last_beat is not None:
        beats.append(last_beat * to_seconds)
    if last_downbeat is not None:
        downbeats.append(last_downbeat * to_seconds)
    return np.sort(np.array(beats)), np.sort(np.array(downbeats))
