"""Piece-level inference by stitching overlapping fragments (Section 9.3)."""
import torch


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


def _forward_fragments(mel, model, fragment_frames, border_frames):
    """Run the model over the overlapping fragments covering one piece.

    Section 9.3 leaves the border beta free but recommends tying it to the candidate
    spacing D/N rather than to something outside the architecture: one cell is the
    resolution at which this head can place an event at all.
    """
    if border_frames is None:
        border_frames = round(fragment_frames / model.task_heads.num_candidates)

    total_frames, _num_mels = mel.shape
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

    with torch.no_grad():
        out = model(torch.stack(batch))

    return fragments, out["class_logits"].float(), out["t_hat"].float()


def stitch_candidates(mel, model, fragment_frames, border_frames=None):
    """Every candidate the piece produces, trimmed to its owning fragment and ordered.

    Candidates rather than decoded events, so a read-out needing the whole piece at once
    -- a local tempo, a bar phase carried across fragment seams -- can run after the
    stitching rather than inside it. Returns which fragment each candidate came from as
    well, since Algorithm 5 requires a fragment and so has to be decoded within one.
    """
    fragments, batched_class_logits, batched_t_hat = _forward_fragments(
        mel, model, fragment_frames, border_frames)
    total_frames = mel.shape[0]

    kept_logits, kept_frames, kept_fragment = [], [], []
    for index, (offset, keep_start, keep_end) in enumerate(fragments):
        absolute = offset + batched_t_hat[index] * fragment_frames
        inside = _inside(absolute, keep_start, keep_end, total_frames)
        kept_logits.append(batched_class_logits[index][inside])
        kept_frames.append(absolute[inside])
        kept_fragment.append(torch.full((int(inside.sum()),), index,
                                        dtype=torch.long, device=mel.device))

    frames = torch.cat(kept_frames)
    order = torch.argsort(frames)
    return (torch.cat(kept_logits)[order], frames[order],
            torch.cat(kept_fragment)[order])


def _inside(absolute, keep_start, keep_end, total_frames):
    """The half-open interior seam: the final fragment owns its own last frame."""
    is_last = keep_end == total_frames
    upper_ok = (absolute <= keep_end) if is_last else (absolute < keep_end)
    return (absolute >= keep_start) & upper_ok
