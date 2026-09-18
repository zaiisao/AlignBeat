"""Piece-level inference by stitching overlapping fragments (Section 9.3)."""
import torch

from alignbeat.inference.decode import decode_events_detect


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


def stitch_piece(mel, forward_fn, fragment_frames, border_frames, tau=0.5):
    """Section 9.3 over one piece."""
    total_frames, num_mels = mel.shape
    fragments = fragment_offsets(total_frames, fragment_frames, border_frames)

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
        classes, times, scores = decode_events_detect(
            batched_class_logits[index], batched_t_hat[index], tau)

        if classes.numel() == 0:
            continue

        # t_hat is normalised to (0, 1] within the fragment -> absolute frames
        absolute = offset + times * fragment_frames

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
