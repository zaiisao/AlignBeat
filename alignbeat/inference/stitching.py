"""Piece-level inference by stitching overlapping fragments (Section 9.3)."""
import torch


def get_fragment_offsets(total_frames, fragment_size, border_size):
    """Offsets o_1 = 0, o_2 = D - 2*beta, ... covering [0, total_frames)."""
    if fragment_size <= 2 * border_size:
        raise ValueError(
            f"border_size {border_size} must be under half the window "
            f"{fragment_size}; otherwise consecutive fragments cannot meet")

    stride = fragment_size - 2 * border_size
    offsets = []
    offset = 0
    while True:
        offsets.append(offset)
        if offset + fragment_size >= total_frames:
            break
        offset += stride

    if len(offsets) > 1 and total_frames - offsets[-1] < fragment_size:
        offsets[-1] = total_frames - fragment_size

    fragments = []
    covered_to = 0
    for index, offset in enumerate(offsets):
        first, last = index == 0, index == len(offsets) - 1

        # Keep each fragment's owned interior; covered_to absorbs overlap when the
        # final fragment shifts back to the end of the piece.
        keep_start = 0 if first else max(offset + border_size, covered_to)
        keep_end = total_frames if last else offset + fragment_size - border_size
        keep_end = max(keep_end, keep_start)
        covered_to = keep_end
        fragments.append((offset, keep_start, keep_end))
    return fragments


def _get_fragment_batch(mel, fragment_offsets, fragment_size):
    """Cut one piece into the overlapping fragments the model is given."""
    fragment_batch = []
    for offset, _keep_start, _keep_end in fragment_offsets:
        fragment = mel[offset:offset + fragment_size]
        if fragment.shape[0] < fragment_size:
            # Only the final fragment can be short. Zero-pad to the fixed window: the
            # dataloader pads short pieces the same way, and log1p(mel) == 0 is silence.
            fragment = torch.nn.functional.pad(
                fragment, (0, 0, 0, fragment_size - fragment.shape[0]))

        fragment_batch.append(fragment)

    return fragment_batch


def predict_and_stitch_fragments(mel, model, fragment_size, border_size=None):
    """Every candidate the piece produces, trimmed to its owning fragment and ordered.

    Candidates rather than decoded events, so a read-out needing the whole piece at once
    -- a local tempo, a bar phase carried across fragment seams -- can run after the
    stitching rather than inside it. Returns which fragment each candidate came from as
    well, since Algorithm 5 requires a fragment and so has to be decoded within one.
    """
    total_frames, _num_mels = mel.shape

    if border_size is None:
        # Section 9.3 leaves the border beta free but recommends tying it to the
        # candidate spacing D/N rather than to something outside the architecture: one
        # cell is the resolution at which this head can place an event at all.
        border_size = round(fragment_size / model.task_heads.num_candidates)

    fragment_offsets = get_fragment_offsets(total_frames, fragment_size, border_size)
    fragment_batch = _get_fragment_batch(mel, fragment_offsets, fragment_size)

    with torch.no_grad():
        fragments_prediction = model(torch.stack(fragment_batch))

    fragments_class_logits = fragments_prediction["class_logits"].float()
    fragments_t_hat = fragments_prediction["t_hat"].float()

    kept_logits, kept_frames, kept_fragment = [], [], []
    for index, (offset, keep_start, keep_end) in enumerate(fragment_offsets):
        absolute = offset + fragments_t_hat[index] * fragment_size
        inside = _inside(absolute, keep_start, keep_end, total_frames)
        kept_logits.append(fragments_class_logits[index][inside])
        kept_frames.append(absolute[inside])
        kept_fragment.append(torch.full((int(inside.sum()),), index,
                                        dtype=torch.long, device=mel.device))

    frames_of_fragments = torch.cat(kept_frames)
    order = torch.argsort(frames_of_fragments)
    return (torch.cat(kept_logits)[order], frames_of_fragments[order],
            torch.cat(kept_fragment)[order])


def _inside(absolute, keep_start, keep_end, total_frames):
    """The half-open interior seam: the final fragment owns its own last frame."""
    is_last = keep_end == total_frames
    upper_ok = (absolute <= keep_end) if is_last else (absolute < keep_end)
    return (absolute >= keep_start) & upper_ok
