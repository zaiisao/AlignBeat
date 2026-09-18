"""The alignment head's training and inference steps.

Free functions over the Lightning module rather than methods on it: PLBeatThis
calls them from four guard clauses, which keeps beat_this/ at a handful of edited
lines instead of carrying 150 of ours.
"""
import numpy as np
import torch

from alignbeat.constants import CLASS_UNKNOWN, CLASS_DOWNBEAT, CLASS_BEAT
from alignbeat.inference.decode import decode_events_detect
from alignbeat.inference.stitching import stitch_piece
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

        targets.append({
            "times": torch.as_tensor(beats / window_seconds,
                                     dtype=torch.float32, device=device),
            "classes": torch.as_tensor(classes, dtype=torch.long, device=device),
        })
    return targets


def subset_decode(batch, model_prediction, fps, *, detect_tau):
    """Inference per excerpt, returned as predicted TIMES in seconds."""
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / fps
    padding_mask = batch.get("padding_mask")
    beats, downbeats = [], []

    for index in range(len(batch["spect"])):
        classes, times, _ = decode_events_detect(
            model_prediction["class_logits"][index].float(),
            model_prediction["t_hat"][index].float(), detect_tau)

        seconds = (times * window_seconds).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()

        if padding_mask is not None:
            valid_seconds = float(padding_mask[index].sum()) / fps
            keep = seconds < valid_seconds
            seconds, classes = seconds[keep], classes[keep]

        beats.append(np.sort(seconds))
        downbeats.append(np.sort(seconds[classes == CLASS_DOWNBEAT]))

    return tuple(beats), tuple(downbeats)


def subset_predict_piece(model, batch, chunk_size, fps, *, detect_tau):
    """Whole-piece decoding for the alignment head (Section 9.3).

    Returns (beats, downbeats) in seconds; the caller scores them, so nothing here
    reaches into the Lightning module.
    """
    def forward_fn(batch_mel):
        with torch.no_grad():
            out = model(batch_mel)
        return out["class_logits"].float(), out["t_hat"].float()

    # Section 9.3 leaves beta free but recommends tying it to the candidate spacing
    # D/N rather than to something outside the architecture: one cell is the
    # resolution at which this head can place an event at all.
    border = round(chunk_size / model.task_heads.num_candidates)

    classes, frames, _scores = stitch_piece(
        batch["spect"][0], forward_fn, chunk_size, border, detect_tau)

    seconds = (frames / fps).detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()
    return (seconds,), (seconds[classes == CLASS_DOWNBEAT],)

