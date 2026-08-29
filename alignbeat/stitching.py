"""Piece-level inference by stitching overlapping fragments (Section 9.3).

The subset-selection head emits a fixed N candidates for a fixed-duration window, so
it cannot simply be handed a whole song. A piece is processed as
overlapping D-frame fragments, each decoded independently by Algorithm 5, and the
per-fragment detection lists are combined into one piece-level list.

Section 8.3's key observation: sigma_hat is a correspondence internal to one fragment's
candidate indexing and has no meaning across fragments, so there is no fragment-level
analogue of "discard the border frames". What does transfer is trimming and stitching
at the level of *decoded detections*, which are just (class, time) pairs on a shared
absolute axis once the fragment offset is known. That is what this module does.

Consecutive fragments overlap by 2*beta and each keeps only [o_k + beta, o_k + D - beta],
except that the first fragment keeps down to 0 and the last keeps up to the end - there
is no preceding or following fragment to prefer over the piece's own boundary. The keep
regions therefore partition the piece with no gap and no double-count: every time is the
responsibility of exactly one fragment.

What this does NOT fix (stated in section 8.3, and inherited rather than introduced):
trimming removes boundary *bookkeeping* errors, not boundary *context* errors. An event
inside fragment k's keep region is only reported if fragment k detects it; a neighbour
that saw it with better context cannot contribute, because its copy lies in a discarded
border. Beat This! carries the same residual risk.
"""
import torch

from alignbeat.subset_head import decode_events


def fragment_offsets(total_frames, window_frames, border_frames):
    """Offsets o_1 = 0, o_2 = D - 2*beta, ... covering [0, total_frames).

    Returns a list of (offset, keep_start, keep_end) in absolute frame units, where the
    keep bounds already carry the first/last fragment exceptions. keep_end of one
    fragment equals keep_start of the next, so the regions tile exactly.
    """
    if window_frames <= 2 * border_frames:
        raise ValueError(
            f"border_frames {border_frames} must be under half the window "
            f"{window_frames}; otherwise consecutive fragments cannot meet")

    stride = window_frames - 2 * border_frames
    offsets = []
    offset = 0
    while True:
        offsets.append(offset)
        if offset + window_frames >= total_frames:
            break
        offset += stride

    fragments = []
    for index, offset in enumerate(offsets):
        first, last = index == 0, index == len(offsets) - 1
        keep_start = 0 if first else offset + border_frames
        keep_end = total_frames if last else offset + window_frames - border_frames
        fragments.append((offset, keep_start, keep_end))
    return fragments


def stitch_piece(mel, forward_fn, window_frames, border_frames,
                 threshold_beat=0.2, threshold_downbeat=0.2, db_margin=0.0):
    """Section 9.3 over one piece.

    mel: (T, n_mels) log-mel for the whole piece (already on the right device).
    forward_fn: callable taking (B, window_frames, n_mels) and returning
        (class_logits (B, N, 3), t_hat (B, N)) - normally the model's eval path. It is
        called ONCE, with every fragment of the piece stacked into one batch.

    Returns (classes, frames, scores) as 1-D tensors on the absolute frame axis of the
    piece, ascending in time. Frames are float: a candidate's time is continuous within
    its fragment, not snapped to the frame grid.
    """
    total_frames, num_mels = mel.shape
    fragments = fragment_offsets(total_frames, window_frames, border_frames)

    # Build every fragment first, then run them through the model as ONE batch.
    # Decoding is per fragment either way, but the forward is not: one call at batch B
    # replaces B calls at batch 1, which on a transformer this size is the difference
    # between saturating the GPU and paying kernel-launch overhead B times over. This
    # is numerically identical -- the model is in eval mode and uses LayerNorm, so no
    # statistic crosses the batch axis.
    batch = []
    for offset, _keep_start, _keep_end in fragments:
        fragment = mel[offset:offset + window_frames]
        if fragment.shape[0] < window_frames:
            # Only the final fragment can be short. Zero-pad to the fixed window: the
            # dataloader pads short pieces the same way, and log1p(mel) == 0 is silence.
            fragment = torch.nn.functional.pad(
                fragment, (0, 0, 0, window_frames - fragment.shape[0]))
        batch.append(fragment)
    batched_logits, batched_t_hat = forward_fn(torch.stack(batch))

    all_classes, all_frames, all_scores = [], [], []
    for index, (offset, keep_start, keep_end) in enumerate(fragments):
        classes, times, scores = decode_events(
            batched_logits[index], batched_t_hat[index],
            threshold_beat, threshold_downbeat, db_margin=db_margin)
        if classes.numel() == 0:
            continue

        # t_hat is normalised to (0, 1] within the fragment -> absolute frames
        absolute = offset + times * window_frames
        # Interior seams are half-open [keep_start, keep_end) so a detection landing
        # exactly on a seam belongs to exactly one fragment (the paper's Alg. 4 writes
        # closed intervals, which would double-count seam detections - audit finding,
        # half-open kept deliberately). The LAST fragment's upper bound is inclusive:
        # keep_end there is the true piece end, and t_hat_N == 1.0 by equation (1)
        # guarantees a candidate can land exactly on it - with a strict < it was
        # always dropped (audit finding, confirmed).
        is_last = keep_end == total_frames
        upper_ok = (absolute <= keep_end) if is_last else (absolute < keep_end)
        inside = (absolute >= keep_start) & upper_ok
        if not bool(inside.any()):
            continue
        all_classes.append(classes[inside])
        all_frames.append(absolute[inside])
        all_scores.append(scores[inside])

    if not all_frames:
        empty_long = torch.zeros(0, dtype=torch.long, device=mel.device)
        empty_float = torch.zeros(0, device=mel.device)
        return empty_long, empty_float, empty_float

    classes = torch.cat(all_classes)
    frames = torch.cat(all_frames)
    scores = torch.cat(all_scores)
    order = torch.argsort(frames)
    return classes[order], frames[order], scores[order]
