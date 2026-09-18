"""Targets in, detections out: Algorithm 10 and the annotation conversions."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from librosa.sequence import viterbi

from alignbeat.constants import CLASS_BACKGROUND, CLASS_BEAT, CLASS_UNKNOWN, CLASS_DOWNBEAT


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def targets_to_events(target, num_frames=None):
    """Frame-grid target (2, T) -> event list for one fragment."""
    if num_frames is None:
        num_frames = target.shape[-1]
    beat_frames = torch.nonzero(target[0] > 0, as_tuple=False).flatten()
    downbeat_frames = torch.nonzero(target[1] > 0, as_tuple=False).flatten()

    downbeat_set = set(downbeat_frames.tolist())
    beat_only = [f for f in beat_frames.tolist() if f not in downbeat_set]

    frames = sorted(downbeat_set.union(beat_only))
    if len(frames) == 0:
        return {
            'classes': torch.zeros(0, dtype=torch.long, device=target.device),
            'times': torch.zeros(0, dtype=torch.float32, device=target.device),
        }

    classes = torch.tensor(
        [CLASS_DOWNBEAT if f in downbeat_set else CLASS_BEAT for f in frames],
        dtype=torch.long, device=target.device)
    times = torch.tensor(frames, dtype=torch.float32, device=target.device) / float(num_frames)
    return {'classes': classes, 'times': times}


def intervals_to_events(annotations, num_frames):
    """Collated (M, 3) interval annotations -> event list. This is the path the real"""
    if annotations.dim() == 3:
        return [intervals_to_events(annotations[b], num_frames) for b in range(annotations.shape[0])]

    device = annotations.device
    empty = {
        'classes': torch.zeros(0, dtype=torch.long, device=device),
        'times': torch.zeros(0, dtype=torch.float32, device=device),
    }
    if annotations.numel() == 0:
        return empty

    valid = annotations[annotations[:, 2] >= 0]
    if valid.numel() == 0:
        return empty

    def endpoints(rows):
        if rows.numel() == 0:
            return torch.zeros(0, device=device)
        return torch.unique(torch.cat((rows[:, 0], rows[:, 1])))

    # class_id 2 marks a beat-only dataset (dataloader.CLASS_BEAT_ONLY): the event is
    # certainly a beat, but whether it is a downbeat was never annotated. Such a
    # fragment carries ONLY these rows, so handle it before the normal two-chain case.
    beat_only = endpoints(valid[valid[:, 2] == 2])
    if beat_only.numel() > 0:
        beat_only = beat_only[(beat_only >= 0) & (beat_only <= num_frames)]
        return {
            'classes': torch.full((beat_only.numel(),), CLASS_UNKNOWN,
                                  dtype=torch.long, device=device),
            'times': beat_only.float() / float(num_frames),
        }

    downbeat_frames = endpoints(valid[valid[:, 2] == CLASS_DOWNBEAT])
    beat_frames = endpoints(valid[valid[:, 2] == CLASS_BEAT])

    frames = torch.unique(torch.cat((downbeat_frames, beat_frames)))
    # Defensive: an annotation frame outside [0, num_frames] would produce an event
    # time outside (0, 1] that the criterion would silently accept (the cost and DP
    # are happy to match it, just badly). The dataloader's crop slices the frame grid
    # before make_intervals so this should not occur; drop rather than clamp if it
    # ever does, since a clamped time would be a fabricated event position.
    frames = frames[(frames >= 0) & (frames <= num_frames)]
    if frames.numel() == 0:
        return empty

    is_downbeat = torch.isin(frames, downbeat_frames)
    classes = torch.where(
        is_downbeat,
        torch.full_like(frames, CLASS_DOWNBEAT, dtype=torch.long),
        torch.full_like(frames, CLASS_BEAT, dtype=torch.long))

    return {'classes': classes, 'times': frames.float() / float(num_frames)}



def meter_phase_scores(class_probs, meter_candidates, meter_prior):
    """Algorithm 5 line 8's numerator: pi_M(L) pi_omega(omega) prod_i q_i(c_i(omega, L)),
    one row of phases per meter. The E-step's line 39 is the same quantity, so it calls
    this too -- note it passes the prior-corrected posterior where stage 2 passes the raw
    head, which is what the two documents each specify."""
    class_probs = class_probs.double()
    q_db = class_probs[:, CLASS_DOWNBEAT]
    q_b = class_probs[:, CLASS_BEAT]
    events = torch.arange(q_db.shape[0], device=q_db.device)
    scores_by_meter = {}

    for meter in meter_candidates:
        meter = int(meter)
        if meter <= 1:
            continue

        # pi_M(L) pi_omega(omega), the same under every phase of a given meter.
        prior = meter_prior.get(meter, 0.0) * (1.0 / meter)

        phases = torch.arange(meter, device=q_db.device)
        # is_db[p, i]: event i is a downbeat under omega = p, i.e. (p + i) % L == 0.
        is_db = ((phases[:, None] + events[None, :]) % meter) == 0

        # q_i(c_i(omega, L)) for every event, then the product over events.
        factors = torch.where(is_db, q_db[None, :], q_b[None, :])
        scores_by_meter[meter] = prior * factors.prod(dim=1)

    return scores_by_meter or None


def _map_pattern(p, meter_candidates, meter_prior):
    """Algorithm 5 lines 7-13: build Pi over every (omega, L), take its mode, and read
    every label off that one hypothesis. None when no meter candidate is viable, which
    means there is no hypothesis to maximise over -- not that the answer is beats."""
    scores_by_meter = meter_phase_scores(p, meter_candidates, meter_prior)
    if not scores_by_meter:
        return None
    meters = list(scores_by_meter)
    flat = torch.cat([scores_by_meter[meter] for meter in meters])
    joint_posterior = flat / flat.sum()

    # Each meter holds one entry per phase, in phase order, so the flat argmax
    # decomposes into (L_hat, omega_hat) by walking the concatenation.
    best = int(torch.argmax(joint_posterior))
    start = 0
    for meter in meters:
        width = scores_by_meter[meter].shape[0]
        if best < start + width:
            omega_hat, meter_hat = best - start, meter
            break
        start += width

    events = torch.arange(p.shape[0], device=p.device)
    is_downbeat = ((omega_hat + events) % meter_hat) == 0
    classes = torch.where(is_downbeat, torch.full_like(events, CLASS_DOWNBEAT),
                          torch.full_like(events, CLASS_BEAT))
    return classes, int(omega_hat), int(meter_hat)


def _local_period(times, half):
    """The beat period each gap should be judged against: a median of its neighbours.

    quantile rather than median: torch.median takes the lower of the two central values
    on an even-length window, which the windows at either end of the piece are.
    """
    gaps = times[1:] - times[:-1]
    return torch.stack([gaps[max(0, i - half):i + half + 1].quantile(0.5)
                        for i in range(gaps.shape[0])])


def _elapsed_beats(times, half, max_advance):
    """How many beats each step covers. A gap twice the local period covers two, which
    is how a missed detection costs one transition instead of inverting the phase for
    everything after it."""
    period = _local_period(times, half).clamp_min(1e-9)
    gaps = times[1:] - times[:-1]
    return (gaps / period).round().clamp(1, max_advance).long()


def _offset_transition(meter, eps):
    """How the bar phase moves relative to what the elapsed time already accounts for.

    Tracking the offset psi = (bar position - cumulative advance) mod L rather than the
    bar position itself makes the chain stationary: the deterministic part of the motion
    is absorbed into the coordinate, so all that is left is the slip. psi stays put with
    probability 1 - eps and moves one either way with eps/2, identically at every step,
    which is an ordinary homogeneous HMM and so librosa's own decoder can run it.

    += rather than =: at meter 2 the two slip directions are the same state, and the mass
    has to accumulate there instead of one write overwriting the other.
    """
    states = np.arange(meter)
    matrix = np.zeros((meter, meter))
    matrix[states, states] += 1.0 - eps
    matrix[states, (states + 1) % meter] += eps / 2.0
    matrix[states, (states - 1) % meter] += eps / 2.0
    return matrix / matrix.sum(axis=1, keepdims=True)


def decode(class_logits, times, meter_prior=None, tau=0.5, read_out="per_event",
           eps=0.03, max_advance=3, period_window=64):
    """Candidates in, labelled events out.

    per_event   each candidate takes its own argmax over {DB, B}
    map         Algorithm 5 lines 7-13: the joint mode of Pi over (omega, L), with every
                label read off that one hypothesis
    duration    bar position advances by however many beats the elapsed time says have
                passed, so a missed detection costs one transition instead
    """
    probabilities = F.softmax(class_logits, dim=-1)
    event_mass = 1.0 - probabilities[..., CLASS_BACKGROUND]
    index = torch.nonzero(event_mass >= tau, as_tuple=False).flatten()
    index = index[torch.argsort(times[index], stable=True)]
    p, event_times = probabilities[index], times[index]

    classes = None
    if read_out == "map":
        resolved = _map_pattern(p.double(), tuple(sorted(meter_prior)), meter_prior)
        if resolved is not None:
            classes, _omega_hat, _meter_hat = resolved
    elif read_out == "duration" and event_times.shape[0] >= 2:
        classes = _duration_classes(p, event_times, meter_prior, eps, max_advance,
                                    period_window)

    if classes is None:
        # per_event. CLASS_DOWNBEAT = 0 and CLASS_BEAT = 1, so the argmax over those two
        # columns is the class id itself. Also where map lands when no meter candidate
        # was viable, and duration on a piece too short to read a tempo from.
        classes = p[:, [CLASS_DOWNBEAT, CLASS_BEAT]].argmax(dim=-1)

    return classes, event_times, event_mass[index]


def _duration_classes(p, event_times, meter_prior, eps, max_advance, period_window):
    """The duration read-out: bar position over the whole piece, advancing by the beats
    the elapsed time accounts for. See decode()."""
    p = p.double()
    q = p[:, [CLASS_DOWNBEAT, CLASS_BEAT]]
    q = (q / q.sum(dim=1, keepdim=True)).cpu().numpy()
    downbeat_probability, beat_probability = q[:, 0], q[:, 1]

    advances = _elapsed_beats(event_times, period_window, max_advance).cpu().numpy()
    cumulative = np.concatenate([[0], np.cumsum(advances)])

    best_score, best_path, best_offset = -float("inf"), None, None
    for meter, prior in meter_prior.items():
        meter = int(meter)
        if meter <= 1 or prior <= 0:
            continue

        # Event i is a downbeat under whichever offset cancels the advances accumulated
        # up to it. The head names a downbeat or a beat and cannot say which beat of the
        # bar, so every other offset carries the beat probability.
        downbeat_offset = (-cumulative) % meter
        emission = np.tile(beat_probability, (meter, 1))
        emission[downbeat_offset, np.arange(len(event_times))] = downbeat_probability

        path, score = viterbi(np.clip(emission, 1e-300, None),
                              _offset_transition(meter, eps),
                              p_init=np.full(meter, 1.0 / meter), return_logp=True)
        score += math.log(prior)
        if score > best_score:
            best_score, best_path, best_offset = score, path, downbeat_offset

    return torch.as_tensor(
        np.where(best_path == best_offset, CLASS_DOWNBEAT, CLASS_BEAT),
        device=event_times.device)
