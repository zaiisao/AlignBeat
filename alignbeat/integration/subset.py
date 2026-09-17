"""The alignment head's training and inference steps.

Free functions over the Lightning module rather than methods on it: PLBeatThis
calls them from four guard clauses, which keeps beat_this/ at a handful of edited
lines instead of carrying 150 of ours.
"""
import numpy as np
import torch

from alignbeat.constants import CLASS_UNKNOWN, CLASS_DOWNBEAT, CLASS_BEAT
from alignbeat.inference.decode import (decode_events, decode_events_detect,
                              decode_events_metrical)
from alignbeat.inference.stitching import stitch_piece



def subset_loss(module, batch, model_prediction):
    losses, _stats = module.subset_criterion(
        model_prediction["class_logits"].float(),
        model_prediction["t_hat"].float(),
        subset_targets(module, batch))
    return {"class": losses["class"], "time": losses["time"],
            "total": losses["total"]}


def subset_targets(module, batch):
    """Ground-truth events for the alignment head, from this batch's own annotations."""
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / module.fps
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


def subset_decode(module, batch, model_prediction, *, decode, tau, detect_tau):
    """Inference per excerpt, returned as predicted TIMES in seconds.

    Either decoding rule, selected by decode: the per-candidate argmax of
    decode_events, or Algorithm 3's two stages. Everything after the call -- the
    seconds conversion, the padding-mask restriction, the sort -- is shared, so the
    two rules differ in exactly one thing: which candidates are emitted with which
    classes."""
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / module.fps
    padding_mask = batch.get("padding_mask")
    beats, downbeats = [], []

    for index in range(len(batch["spect"])):
        logits = model_prediction["class_logits"][index].float()
        candidate_times = model_prediction["t_hat"][index].float()

        if decode == "metrical":
            # Algorithm 3. One fragment per call is exactly the condition section 3
            # states its stage 2 for: a single (omega, L) spans this window.
            classes, times, _ = decode_events_metrical(
                logits, candidate_times, module.subset_criterion, detect_tau)
        elif decode == "detect":
            classes, times, _ = decode_events_detect(
                logits, candidate_times, detect_tau)
        else:
            classes, times, _ = decode_events(
                logits, candidate_times, tau)

        seconds = (times * window_seconds).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()

        if padding_mask is not None:
            valid_seconds = float(padding_mask[index].sum()) / module.fps
            keep = seconds < valid_seconds
            seconds, classes = seconds[keep], classes[keep]

        beats.append(np.sort(seconds))
        downbeats.append(np.sort(seconds[classes == CLASS_DOWNBEAT]))

    return tuple(beats), tuple(downbeats)


def subset_predict_piece(module, batch, chunk_size, *,
                         
                         decode, tau, detect_tau, stitch_border):
    """Whole-piece decoding for the alignment head (Section 9.3)."""
    def forward_fn(batch_mel):
        with torch.no_grad():
            out = module.model(batch_mel)
        return out["class_logits"].float(), out["t_hat"].float()

    border = stitch_border
    if border is None:
        # A subset run has no beat_loss at all, and nn.Module.__getattr__ raises
        # before getattr's default can apply, so guard the module as well.
        border = 2 * getattr(getattr(module, "beat_loss", None), "tolerance", 3)

    # Same rule as subset_decode uses per excerpt, so whole-piece inference and
    # validation cannot disagree about how candidates are emitted.
    decode_fn = None
    if decode == "metrical":
        tau = detect_tau
        def decode_fn(logits, t_hat, tau):
            return decode_events_metrical(logits, t_hat, module.subset_criterion, tau)
    elif decode == "detect":
        tau = detect_tau
        decode_fn = decode_events_detect
    classes, frames, _scores = stitch_piece(
        batch["spect"][0], forward_fn, chunk_size, border, tau,
        decode_fn=decode_fn)

    seconds = (frames / module.fps).detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()
    beats = (seconds,)
    downbeats = (seconds[classes == CLASS_DOWNBEAT],)
    metrics = module._compute_metrics(batch, beats, downbeats, step="test")
    return metrics, None, batch["dataset"], batch["spect_path"]

