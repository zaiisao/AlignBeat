"""The alignment head's training and inference steps.

Free functions over the Lightning module rather than methods on it: PLBeatThis
calls them from four guard clauses, which keeps beat_this/ at a handful of edited
lines instead of carrying 150 of ours.
"""
import numpy as np
import torch

from alignbeat.constants import CLASS_UNKNOWN, CLASS_DOWNBEAT, CLASS_BEAT
from alignbeat.inference.decode import decode
from alignbeat.inference.stitching import predict_and_stitch_fragments
from alignbeat.training.criterion import SubsetCriterion



def subset_loss(criterion: SubsetCriterion, batch, model_prediction, fps):
    losses, _stats = criterion(
        model_prediction["class_logits"].float(),
        model_prediction["t_hat"].float(),
        subset_targets(batch, fps))
    return {"class": losses["class"], "time": losses["time"],
            "total": losses["total"]}


def subset_targets(batch, fps):
    """Ground-truth events for the alignment head, from this batch's own annotations."""
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / fps
    device = batch["spect"].device
    targets = []
    for index in range(len(batch["spect"])):
        beats = np.frombuffer(batch["truth_orig_beat"][index])
        downbeats = np.frombuffer(batch["truth_orig_downbeat"][index])
        has_downbeats = bool(batch["downbeat_mask"][index])

        # eq. (1) maps onto the half-open axis (0, 1], so a target at exactly 0 is
        # unreachable by construction and would be an unmatchable event.
        keep = (beats > 0) & (beats <= window_seconds)
        beats = np.unique(beats[keep])   # unique, not just sorted: Definition 1
        if has_downbeats:
            classes = np.where(np.isin(beats, downbeats), CLASS_DOWNBEAT, CLASS_BEAT)
        else:
            classes = np.full(len(beats), CLASS_UNKNOWN)
        # Normalize the period to the event-time units. Zero triggers the criterion's
        # fixed-scale fallback; .get() supports checkpoints without beat_period.
        period_seconds = float(batch["beat_period"][index]) if "beat_period" in batch else 0.0

        targets.append({
            "times": torch.as_tensor(beats / window_seconds,
                                     dtype=torch.float32, device=device),
            "classes": torch.as_tensor(classes, dtype=torch.long, device=device),
            "period": period_seconds / window_seconds,
        })
    return targets


def subset_decode(criterion, batch, model_prediction, fps, *, detect_tau,
                  read_out="per_event", stitch_border=None):
    """Inference per excerpt, returned as predicted TIMES in seconds.

    Every read-out runs here, including duration -- it estimates its tempo from whatever
    events the excerpt holds, so it is weaker than at piece scope but not ill-defined.

    stitch_border is accepted and ignored: an excerpt is already one fragment, so there
    are no seams to trim. It is in the signature because PLBeatThis splats one
    decode_kwargs into both this and subset_predict_piece, where it does apply.
    """
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / fps
    padding_mask = batch.get("padding_mask")
    beats, downbeats = [], []

    for index in range(len(batch["spect"])):
        classes, times, _ = decode(
            model_prediction["class_logits"][index].float(),
            model_prediction["t_hat"][index].float(), criterion.meter_prior,
            tau=detect_tau, read_out=read_out)

        seconds = (times * window_seconds).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()

        if padding_mask is not None:
            valid_seconds = float(padding_mask[index].sum()) / fps
            keep = seconds < valid_seconds
            seconds, classes = seconds[keep], classes[keep]

        beats.append(np.sort(seconds))
        downbeats.append(np.sort(seconds[classes == CLASS_DOWNBEAT]))

    return tuple(beats), tuple(downbeats)


def subset_predict_piece(model, criterion, batch, chunk_size, fps, *, detect_tau,
                         read_out="per_event", stitch_border=None):
    """Whole-piece decoding for the alignment head (Section 9.3).

    Returns (beats, downbeats) in seconds; the caller scores them, so nothing here
    reaches into the Lightning module.

    stitch_border is the frames trimmed at each fragment seam. None keeps section 9.3's
    D/N, one candidate cell; it is free precisely because the degradation it trims comes
    from truncated context at the edge, which is a property of the encoder rather than of
    the candidate spacing, so the two need not agree.
    """
    class_logits, frames_of_fragments, fragment = predict_and_stitch_fragments(
        batch["spect"][0], model, chunk_size, border_size=stitch_border)

    if read_out == "map":
        # Decode each fragment independently before concatenating its labelled events.
        parts = [decode(class_logits[fragment == f], frames_of_fragments[fragment == f],
                        criterion.meter_prior, tau=detect_tau, read_out=read_out)
                 for f in fragment.unique()]
        classes = torch.cat([part[0] for part in parts])
        frames_of_fragments = torch.cat([part[1] for part in parts])
    else:
        classes, frames_of_fragments, _scores = decode(class_logits, frames_of_fragments, criterion.meter_prior,
                                          tau=detect_tau, read_out=read_out)

    seconds = (frames_of_fragments / fps).detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()
    return (seconds,), (seconds[classes == CLASS_DOWNBEAT],)
