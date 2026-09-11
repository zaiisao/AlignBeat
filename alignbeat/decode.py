"""Targets in, detections out: Algorithm 10 and the annotation conversions."""
import numpy as np
import torch
import torch.nn.functional as F

from alignbeat.classes import BACKGROUND, BEAT, CLASS_UNKNOWN, DOWNBEAT


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
        [DOWNBEAT if f in downbeat_set else BEAT for f in frames],
        dtype=torch.long, device=target.device)
    times = torch.tensor(frames, dtype=torch.float32, device=target.device) / float(num_frames)
    return {'classes': classes, 'times': times}


def batch_targets_to_events(targets, num_frames=None):
    return [targets_to_events(targets[b], num_frames=num_frames) for b in range(targets.shape[0])]


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

    downbeat_frames = endpoints(valid[valid[:, 2] == DOWNBEAT])
    beat_frames = endpoints(valid[valid[:, 2] == BEAT])

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
        torch.full_like(frames, DOWNBEAT, dtype=torch.long),
        torch.full_like(frames, BEAT, dtype=torch.long))

    return {'classes': classes, 'times': frames.float() / float(num_frames)}


# ---------------------------------------------------------------------------
# Inference (section 9.2, Algorithm 10)
# ---------------------------------------------------------------------------

def estimate_beat_period(times, scores, threshold=0.2):
    """Equation (36)'s Delta_bar, estimated from PREDICTIONS rather than matches."""
    kept = times[scores >= threshold]
    if kept.shape[0] < 4:
        return None
    gaps = np.diff(np.sort(kept))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else None


def decode_events(class_logits, t_hat, tau=0.2):
    """Per-candidate argmax over {DB, B, empty}, kept when it clears tau."""
    probabilities = F.softmax(class_logits, dim=-1)
    scores, predicted = probabilities.max(dim=-1)
    keep = (predicted != BACKGROUND) & (scores >= tau)
    return predicted[keep], t_hat[keep], scores[keep]


def decode_events_metrical(class_logits, t_hat, criterion, tau=0.5):
    """Algorithm 3: detect events, then resolve ONE (omega, L) across all of them.

    Section 3.1's objection to decode_events is that per-candidate argmax can emit a
    pattern no (omega, L) could produce -- two adjacent downbeats, say -- because
    nothing in the head's own loss ties events to each other at inference. Here every
    emitted label is c_i(omega_hat, L_hat) for one jointly-chosen hypothesis, so a
    metrically valid output is guaranteed by construction rather than hoped for.

    Two differences from decode_events, both from the spec and both deliberate:
    line 5 thresholds the EVENT mass 1 - p(empty) rather than the winning class's own
    probability, so a candidate split evenly between DB and B still counts as an event;
    and tau defaults to 0.5, the value section 3's own note names, not 0.2.

    criterion supplies pi_data, pi_C, pi_M and pi_omega, all of which ride in the
    checkpoint. Returns (classes, times, scores) exactly as decode_events does.
    """
    probabilities = F.softmax(class_logits, dim=-1)
    event_mass = 1.0 - probabilities[..., BACKGROUND]
    keep = event_mass >= tau                                      # line 5
    # Line 7: relabel the kept candidates by increasing t_hat. monotonic_times is
    # strictly increasing in the candidate index in fp32, but NOT under autocast: at
    # N=188 fp16 collapses adjacent centres onto each other on ~6% of real fragments,
    # and training runs precision="16-mixed". Since c_i(omega, L) is indexed by
    # position, an inversion silently mislabels everything after it, so line 7 is
    # performed rather than assumed.
    index = torch.nonzero(keep, as_tuple=False).flatten()
    index = index[torch.argsort(t_hat[index], stable=True)]
    if index.numel() == 0:
        empty = index
        return empty, t_hat[empty], event_mass[empty]

    p = torch.softmax(class_logits, dim=-1)[index]
    resolved = criterion.infer_pattern(p)                         # lines 10-24
    if resolved is None:
        # Fewer detected events than the smallest candidate meter, so no hypothesis
        # exists to resolve. Fall back to the per-candidate call over {DB, B}: still a
        # detection, just with no metrical structure available to constrain it.
        classes = log_p[:, [DOWNBEAT, BEAT]].argmax(dim=-1)
    else:
        classes, _omega, _meter = resolved
    return classes, t_hat[index], event_mass[index]
