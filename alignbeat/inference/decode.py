"""Targets in, detections out: Algorithm 10 and the annotation conversions."""
import torch
import torch.nn.functional as F

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



def _stage1(class_logits, t_hat, tau):
    """Algorithm 3 lines 5-7: keep J = {j : 1 - p_j(empty) >= tau}, sorted by t_hat."""
    probabilities = F.softmax(class_logits, dim=-1)
    event_mass = 1.0 - probabilities[..., CLASS_BACKGROUND]             # line 5
    index = torch.nonzero(event_mass >= tau, as_tuple=False).flatten()
    index = index[torch.argsort(t_hat[index], stable=True)]       # line 7
    return probabilities, index, event_mass


def decode_events_detect(class_logits, t_hat, tau=0.5):
    """Algorithm 3 stage 1, then a per-candidate argmax over {DB, B}."""
    probabilities, index, event_mass = _stage1(class_logits, t_hat, tau)
    p = probabilities[index]
    # CLASS_DOWNBEAT = 0, CLASS_BEAT = 1, so the argmax over those two columns is the class id.
    classes = p[:, [CLASS_DOWNBEAT, CLASS_BEAT]].argmax(dim=-1)
    return classes, t_hat[index], event_mass[index]
