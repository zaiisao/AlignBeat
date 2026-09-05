"""Piece-level inference by stitching overlapping fragments (Section 9.3)."""
import torch

from alignbeat.decode import decode_events


def fragment_offsets(total_frames, fragment_frames, border_frames):
    """Offsets o_1 = 0, o_2 = D - 2*beta, ... covering [0, total_frames)."""
    if fragment_frames <= 2 * border_frames:
        raise ValueError(
            f"border_frames {border_frames} must be under half the window "
            f"{fragment_frames}; otherwise consecutive fragments cannot meet")

    stride = fragment_frames - 2 * border_frames
    offsets = []
    offset = 0
    while True:
        offsets.append(offset)
        if offset + fragment_frames >= total_frames:
            break
        offset += stride

    # Zero padding in the LAST fragment is padding in the one fragment that decodes the
    # end of the piece, against an input the model never saw in training. Whenever the
    # tail is short of a full D, slide that fragment left to end exactly at the piece
    # end so it is all real audio -- the same strategy as beat_this.inference.split_piece
    # (avoid_short_end), which is what the dense arm has always done. The price is that
    # it then overlaps its predecessor by more than 2*beta, so the offsets are no longer
    # uniformly strided and the keep regions below have to be clamped to a high-water
    # mark to stay a partition. Padding survives only for a piece shorter than D, where
    # there is no earlier audio to slide into.
    if len(offsets) > 1 and total_frames - offsets[-1] < fragment_frames:
        offsets[-1] = total_frames - fragment_frames

    fragments = []
    covered_to = 0
    for index, offset in enumerate(offsets):
        first, last = index == 0, index == len(offsets) - 1

        # JA: offset + border_frames is the first beta frames of the fragment B.
        # offset + fragment_frames - border_frames is the last beta frames of the
        # fragment A.
        keep_start = 0 if first else max(offset + border_frames, covered_to)
        keep_end = total_frames if last else offset + fragment_frames - border_frames
        keep_end = max(keep_end, keep_start)
        covered_to = keep_end
        fragments.append((offset, keep_start, keep_end))
    return fragments


def stitch_piece(mel, forward_fn, fragment_frames, border_frames,
                 threshold_beat=0.2, threshold_downbeat=0.2, db_margin=0.0):
    """Section 9.3 over one piece."""
    total_frames, num_mels = mel.shape
    fragments = fragment_offsets(total_frames, fragment_frames, border_frames)

    # Build every fragment first, then run them through the model as ONE batch.
    # Decoding is per fragment either way, but the forward is not: one call at batch B
    # replaces B calls at batch 1, which on a transformer this size is the difference
    # between saturating the GPU and paying kernel-launch overhead B times over. This
    # is numerically identical -- the model is in eval mode and uses LayerNorm, so no
    # statistic crosses the batch axis.
    batch = []
    for offset, _keep_start, _keep_end in fragments:
        fragment = mel[offset:offset + fragment_frames]
        if fragment.shape[0] < fragment_frames:
            # Only the final fragment can be short. Zero-pad to the fixed window: the
            # dataloader pads short pieces the same way, and log1p(mel) == 0 is silence.
            fragment = torch.nn.functional.pad(
                fragment, (0, 0, 0, fragment_frames - fragment.shape[0]))

        batch.append(fragment)
    batched_class_logits, batched_t_hat = forward_fn(torch.stack(batch))

    all_classes, all_frames, all_scores = [], [], []
    for index, (offset, keep_start, keep_end) in enumerate(fragments):
        classes, times, scores = decode_events(
            batched_class_logits[index], batched_t_hat[index],
            threshold_beat, threshold_downbeat, db_margin=db_margin)

        if classes.numel() == 0:
            continue

        # t_hat is normalised to (0, 1] within the fragment -> absolute frames
        absolute = offset + times * fragment_frames
        # Algorithm 11 line 9 writes a CLOSED keep region for every fragment, and at an
        # interior seam keep_end == the next fragment's keep_start exactly, so a detection
        # landing on a seam is claimed by both and appended twice -- contradicting the
        # paper's own claim that the keep regions make every time the responsibility of
        # exactly one fragment. This is not a measure-zero worry: candidates sit on a
        # regular grid (t_hat_j = j/N under equation (1) for uniform increments) and the
        # seams are integer frames, so exact hits are routine, not accidental -- a closed
        # test duplicates real detections in test_every_event_reported_exactly_once.
        # Interior seams are therefore half-open [keep_start, keep_end), which assigns the
        # seam to the fragment on its right.
        #
        # The LAST fragment keeps the paper's closed upper bound: keep_end there is the
        # piece end, no neighbour can claim it, and t_hat_N == 1.0 exactly by equation (1)
        # means a candidate always lands on it -- a strict < dropped the final candidate of
        # every piece (audit finding, confirmed).
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
