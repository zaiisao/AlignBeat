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


def decode_events_coupled(class_logits, t_hat, beat_period, gamma=0.5, mu=1.0,
                          threshold_beat=0.2, threshold_downbeat=0.2):
    """Decoding that couples neighbouring candidates, in place of Algorithm 10."""
    probabilities = F.softmax(class_logits, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    N = t_hat.shape[0]

    if beat_period is None or N == 0:
        return decode_events(class_logits, t_hat, threshold_beat, threshold_downbeat)

    times = t_hat.detach().cpu().numpy().astype(np.float64)
    logp = log_probabilities.detach().cpu().numpy().astype(np.float64)
    # Fire as whichever active class the candidate itself prefers; the spacing term is
    # about WHETHER a candidate fires, not which kind it is.
    fire_class = np.where(logp[:, DOWNBEAT] >= logp[:, BEAT], DOWNBEAT, BEAT)
    fire_cost = -logp[np.arange(N), fire_class]
    stay_cost = -gamma * logp[:, BACKGROUND]

    NONE = N                      # "nothing has fired yet" state
    INF = np.inf
    best = np.full(N + 1, INF)
    best[NONE] = 0.0
    back = np.full((N, N + 1), -1, dtype=np.int64)

    for j in range(N):
        gap = times[j] - np.concatenate([times, [0.0]])
        penalty = mu * (gap - beat_period) ** 2
        penalty[NONE] = 0.0       # the first event has no predecessor to be spaced from
        fire_from = best + fire_cost[j] + penalty
        source = int(np.argmin(fire_from))
        new = best + stay_cost[j]                     # stay: last-fired index unchanged
        back[j, :] = np.arange(N + 1)                 # provisional: everything stayed
        if fire_from[source] < new[j]:
            new[j] = fire_from[source]
            back[j, j] = source
        best = new

    fired = []
    k = int(np.argmin(best))
    for j in range(N - 1, -1, -1):
        if k == j:
            fired.append(j)
            k = int(back[j, j])
    fired = np.array(sorted(fired), dtype=np.int64)
    if fired.size == 0:
        return decode_events(class_logits, t_hat, threshold_beat, threshold_downbeat)

    keep = torch.as_tensor(fired, device=t_hat.device)
    classes = torch.as_tensor(fire_class[fired], device=t_hat.device, dtype=torch.long)
    scores = 1.0 - probabilities[keep, BACKGROUND]
    return classes, t_hat[keep], scores


def decode_events(class_logits, t_hat, threshold_beat=0.2, threshold_downbeat=0.2,
                  db_margin=0.0):
    """Algorithm 10 for one fragment: argmax per candidate, then threshold."""
    probabilities = F.softmax(class_logits, dim=-1)
    scores, predicted = probabilities.max(dim=-1)

    if db_margin != 0.0:
        log_p = torch.log_softmax(class_logits, dim=-1)
        relabelled = torch.where(
            log_p[:, DOWNBEAT] - log_p[:, BEAT] > db_margin,
            torch.full_like(predicted, DOWNBEAT), torch.full_like(predicted, BEAT))
        predicted = torch.where(predicted == BACKGROUND, predicted, relabelled)
        scores = probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)

    thresholds = torch.where(
        predicted == DOWNBEAT,
        torch.full_like(scores, threshold_downbeat),
        torch.full_like(scores, threshold_beat))
    keep = (predicted != BACKGROUND) & (scores >= thresholds)
    return predicted[keep], t_hat[keep], scores[keep]
