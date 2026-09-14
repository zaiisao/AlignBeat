"""Targets in, detections out: Algorithm 10 and the annotation conversions."""
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

def decode_events(class_logits, t_hat, tau=0.2):
    """Per-candidate argmax over {DB, B, empty}, kept when it clears tau."""
    probabilities = F.softmax(class_logits, dim=-1)
    scores, predicted = probabilities.max(dim=-1)
    keep = (predicted != BACKGROUND) & (scores >= tau)
    return predicted[keep], t_hat[keep], scores[keep]


def _stage1(class_logits, t_hat, tau):
    """Algorithm 3 lines 1-8: event detection, shared by both stage-1 consumers.

    Line 3's forward pass is the caller's; this receives its two outputs. Line 5 keeps
    J = { j : 1 - p_hat_j(empty) >= tau }, thresholding the EVENT mass rather than the
    winning class's own probability, so a candidate split evenly between DB and B still
    counts as an event. Line 7 relabels J by increasing t_hat: monotonic_times is
    strictly increasing in the candidate index in fp32 but NOT under autocast (at N=188
    fp16 collapses adjacent centres on ~6% of real fragments, and training runs
    precision="16-mixed"), and c_i(omega, L) is indexed by position, so an inversion
    would silently mislabel everything after it. The sort is performed, not assumed.

    Returns (probabilities, index, event_mass); index is j_0 < ... < j_{Mhat-1}.
    """
    probabilities = F.softmax(class_logits, dim=-1)
    event_mass = 1.0 - probabilities[..., BACKGROUND]             # line 5
    index = torch.nonzero(event_mass >= tau, as_tuple=False).flatten()
    index = index[torch.argsort(t_hat[index], stable=True)]       # line 7
    return probabilities, index, event_mass


def decode_events_detect(class_logits, t_hat, tau=0.5):
    """Algorithm 3's STAGE 1 ONLY: line 5's detection, then per-candidate labelling.

    DEVIATION from Algorithm 3, deliberate and measured. Lines 9-25 assign c_i(omega, L)
    by RANK, so one missed detection shifts every later index and inverts the phase for
    the rest of the fragment: 25.9 downbeat F1 per deleted event, measured on fragments
    whose detection was otherwise >= 99% correct, with the meter still resolved correctly
    92.3% of the time. Below ~99% detection the true labelling is not in that hypothesis
    class at all -- the oracle over every (omega, L) scores BELOW per-candidate labelling
    there -- so no resolver, prior or re-anchoring recovers it.

    Line 5 is kept because it is right: it is the Bayes rule for event-vs-empty, where
    decode_events' three-way argmax discards a candidate whose event mass is split across
    DB and B. Worth +0.29 beat F1 over argmax on the 8-fold protocol.

    Returns (classes, times, scores) exactly as decode_events does.
    """
    probabilities, index, event_mass = _stage1(class_logits, t_hat, tau)
    p = probabilities[index]
    # DOWNBEAT = 0, BEAT = 1, so the argmax over those two columns is the class id.
    classes = p[:, [DOWNBEAT, BEAT]].argmax(dim=-1)
    return classes, t_hat[index], event_mass[index]


def decode_events_metrical(class_logits, t_hat, criterion, tau=0.5):
    """Algorithm 3 Infer: detect events, then resolve one (omega, L) across all of them.

    Require: audio fragment x, trained theta, detection threshold tau, candidate meters
    M with prior pi_M, phase-offset prior pi_omega, class prior pi_C, training prior
    pi_data. The last five all ride on `criterion`, which carries them from the
    checkpoint. Ensure: predicted events (c_0, t_0), ..., (c_{Mhat-1}, t_{Mhat-1}).

    Section 3.1's objection to decode_events is that a per-candidate argmax can emit a
    pattern no (omega, L) could produce -- two adjacent downbeats, say -- because
    nothing in the head's own loss ties events to each other at inference. Here every
    emitted label is c_i(omega_hat, L_hat) for one jointly-chosen hypothesis, so a
    metrically valid output holds by construction rather than by hope.

    Line 5 thresholds the EVENT mass 1 - p(empty), not the winning class's own
    probability, so a candidate split evenly between DB and B still counts as an event;
    and tau defaults to 0.5, the value section 3's own note names, not decode_events'
    0.2. Returns (classes, times, scores) exactly as decode_events does.
    """
    # ---- Lines 1-8: Stage 1, in _stage1 --------------------------------------------
    probabilities, index, event_mass = _stage1(class_logits, t_hat, tau)
    num_events = index.numel()                                    # Mhat

    # ---- Line 9: Stage 2: metrical refinement { ----------------------------------
    # Lines 10-24 live in criterion.infer_pattern, on the kept candidates in line 7's
    # order:
    #   lines 11-14  P_hat(C = c | x, j_i) by Bayes, dividing pi_data and applying pi_C
    #   lines 16-18  Pi_{omega,L}, the joint posterior over (omega, L)
    #   line 20      (omega_hat, L_hat) <- argmax Pi_{omega,L}
    #   lines 22-24  c_i <- c_i(omega_hat, L_hat)
    # It is the E-step's own two functions underneath, so training and inference score a
    # hypothesis identically by construction.
    # infer_pattern reads Mhat back off p's own leading dimension, which is num_events
    # by construction, so lines 11 and 22's loop bounds are line 7's cardinality.
    p = probabilities[index]
    assert p.shape[0] == num_events
    resolved = criterion.infer_pattern(p)

    if resolved is None:
        # DEVIATION. Line 17's product runs over i = 0..Mhat-1 for every (L, omega), so
        # the spec always has a hypothesis for line 20's argmax. Our meter candidates are
        # only scored when Mhat admits them, so too few detections leaves the set empty.
        # Falling back to the per-candidate call over {DB, B} keeps line 5's detections
        # rather than dropping the fragment; it forfeits only line 9's metrical
        # guarantee, which had no hypothesis to enforce anyway.
        # p is already restricted to the kept candidates, and DOWNBEAT = 0, BEAT = 1, so
        # the argmax over those two columns is the class id itself.
        classes = p[:, [DOWNBEAT, BEAT]].argmax(dim=-1)
    else:
        classes, _omega_hat, _meter_hat = resolved
    # ---- Line 25: } --------------------------------------------------------------

    # Line 26: return (c_0, t_0), ..., (c_{Mhat-1}, t_{Mhat-1}). Line 23 pairs each class
    # with t_hat_{j_i}, so times carry line 7's ordering. The third element is this
    # rule's own confidence, the line 5 mass, standing where decode_events returns the
    # winning class probability.
    return classes, t_hat[index], event_mass[index]
