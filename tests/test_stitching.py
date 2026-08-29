"""Tests for piece-level stitching (Algorithm 4).

The property that matters: keep regions must partition the piece exactly, so a planted
event is reported once and only once regardless of where fragment boundaries fall.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignbeat.stitching import fragment_offsets, stitch_piece  # noqa: E402
from alignbeat.subset_head import BACKGROUND, BEAT, DOWNBEAT  # noqa: E402


def test_keep_regions_partition_exactly():
    """No gap, no overlap, full coverage - for many piece lengths including ones that
    do not divide evenly by the stride."""
    for total in (1280, 1281, 2000, 4096, 4097, 5000, 12345):
        for window, border in ((1280, 8), (1280, 64), (500, 10)):
            if total < window:
                continue
            fragments = fragment_offsets(total, window, border)
            assert fragments[0][1] == 0, "first fragment must keep down to 0"
            assert fragments[-1][2] == total, "last fragment must keep up to the end"
            for (_, _, end), (_, start, _) in zip(fragments, fragments[1:]):
                assert end == start, f"gap/overlap at {end} vs {start}"
            for offset, keep_start, keep_end in fragments:
                assert keep_start < keep_end, "empty keep region"
                assert keep_start >= offset, "keep region starts before its fragment"
    print("ok: keep regions partition [0, l) exactly for every tested length")


def test_border_must_be_under_half_window():
    try:
        fragment_offsets(1000, 100, 50)
    except ValueError as exc:
        assert "border" in str(exc)
        print("ok: a border >= half the window is rejected")
        return
    raise AssertionError("expected ValueError")


def test_every_event_reported_exactly_once():
    """Plant one detection per fragment at a known absolute position via a fake model,
    and check the stitched output recovers each exactly once.

    The fake model reports a detection at every candidate slot, so each fragment emits
    N detections spread across its own window; after trimming, the union over fragments
    must contain each absolute position once.
    """
    window, border, num_candidates = 400, 20, 40
    total = 1500

    def fake_forward(fragment):
        # every candidate is a confident beat, uniformly spaced over the window
        B = fragment.shape[0]
        logits = torch.full((B, num_candidates, 3), -10.0)
        logits[:, :, BEAT] = 10.0
        t_hat = (torch.arange(1, num_candidates + 1, dtype=torch.float32) / num_candidates)
        return logits, t_hat.unsqueeze(0).expand(B, -1).contiguous()

    mel = torch.zeros(total, 128)
    classes, frames, scores = stitch_piece(mel, fake_forward, window, border)

    assert torch.all(frames[1:] > frames[:-1]), "output must be sorted and duplicate-free"
    assert torch.all(frames >= 0) and torch.all(frames <= total), (
        "the piece end itself is a legal detection time (t_hat_N == 1 exactly)")
    assert torch.all(classes == BEAT)

    # Each fragment contributes exactly the candidates landing in its keep region, and
    # the keep regions tile [0, total) - so the count must equal the number of distinct
    # candidate positions falling inside the piece.
    expected = 0
    for offset, keep_start, keep_end in fragment_offsets(total, window, border):
        positions = offset + (torch.arange(1, num_candidates + 1).float() / num_candidates) * window
        # mirror the stitcher: interior seams half-open, last fragment inclusive so a
        # detection exactly at the piece end is kept, not dropped
        upper = positions <= keep_end if keep_end == total else positions < keep_end
        expected += int(((positions >= keep_start) & upper).sum())
    assert len(frames) == expected, f"{len(frames)} != {expected}"
    print(f"ok: {len(frames)} detections, each reported exactly once")


def test_short_piece_is_padded_not_dropped():
    """A piece shorter than one window must still be decoded, with the pad ignored."""
    window, border, num_candidates = 400, 20, 40

    def fake_forward(fragment):
        assert fragment.shape[1] == window, "model must always see a full window"
        B = fragment.shape[0]
        logits = torch.full((B, num_candidates, 3), -10.0)
        logits[:, :, BEAT] = 10.0
        t_hat = (torch.arange(1, num_candidates + 1, dtype=torch.float32) / num_candidates)
        return logits, t_hat.unsqueeze(0).expand(B, -1).contiguous()

    classes, frames, _ = stitch_piece(torch.zeros(150, 128), fake_forward, window, border)
    assert len(frames) > 0
    assert torch.all(frames <= 150), "detections must be clipped to the real piece length (end inclusive)"
    print(f"ok: short piece padded to a full window, {len(frames)} detections kept in range")


def test_background_only_model_returns_nothing():
    window, border, num_candidates = 400, 20, 40

    def fake_forward(fragment):
        B = fragment.shape[0]
        logits = torch.full((B, num_candidates, 3), -10.0)
        logits[:, :, BACKGROUND] = 10.0
        t_hat = (torch.arange(1, num_candidates + 1, dtype=torch.float32) / num_candidates)
        return logits, t_hat.unsqueeze(0).expand(B, -1).contiguous()

    classes, frames, scores = stitch_piece(torch.zeros(1500, 128), fake_forward, window, border)
    assert len(classes) == 0 and len(frames) == 0 and len(scores) == 0
    print("ok: an all-background model yields an empty detection list, not a crash")


def test_downbeat_class_survives_stitching():
    window, border, num_candidates = 400, 20, 40

    def fake_forward(fragment):
        B = fragment.shape[0]
        logits = torch.full((B, num_candidates, 3), -10.0)
        logits[:, 0::4, DOWNBEAT] = 10.0
        logits[:, 1::4, BEAT] = 10.0
        logits[:, 2::4, BEAT] = 10.0
        logits[:, 3::4, BEAT] = 10.0
        t_hat = (torch.arange(1, num_candidates + 1, dtype=torch.float32) / num_candidates)
        return logits, t_hat.unsqueeze(0).expand(B, -1).contiguous()

    classes, frames, _ = stitch_piece(torch.zeros(1500, 128), fake_forward, window, border)
    assert int((classes == DOWNBEAT).sum()) > 0 and int((classes == BEAT).sum()) > 0
    print("ok: both classes survive stitching")


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nall tests passed" if failures == 0 else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
