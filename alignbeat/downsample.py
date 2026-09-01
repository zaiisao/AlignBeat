"""Encoder output (B, T, d) -> candidate features (B, N, d)."""
import math

import torch.nn.functional as F
from torch import nn

# Fastest tempo the corpus contains, in BPM. Only an upper bound is needed:
# overshooting costs a few extra candidates that get classified as background, which
# the formulation expects anyway, while undershooting is unrecoverable -- an
# order-preserving injection needs N >= M, and SubsetCriterion skips any fragment where
# it does not hold. 15 of 5555 tracks exceed this, all of them asap, whose MIDI-derived
# annotations are note-level rather than beat-level (the densest has a 0.005 s gap,
# i.e. 11521 BPM); those are bad annotations, not fast music.
BPM_MAX = 340.0


def n_candidates_from_tempo(window_frames: int, fps: float,
                            bpm_max: float = BPM_MAX) -> int:
    """N := BPM_max * D, the most events a window of D minutes can contain."""
    return math.ceil(bpm_max * (window_frames / fps) / 60.0)


class Downsample(nn.Module):
    """(B, T, d) -> (B, N, d) by one strided operator."""

    def __init__(self, d_model: int, num_candidates: int, mode: str = "learned",
                 window_frames: int = None):
        super().__init__()

        if mode not in ("learned", "avg", "max"):
            raise ValueError(f"mode must be learned|avg|max, got {mode!r}")

        self.mode = mode
        self.num_candidates = num_candidates
        self.conv = None

        if mode == "learned":
            if window_frames is None:
                raise ValueError("mode='learned' needs window_frames (T) to size the "
                                 "stride; avg/max are adaptive and do not")

            # Halve the length once per layer (1500 -> 750 -> 375) instead of
            # collapsing T/N frames in a single stride: the same receptive field is
            # reached with fewer parameters and a nonlinearity in between.
            self.num_stages = max(1, round(math.log2(window_frames / num_candidates)))
            self.kernel = 2
            self.padded_length = num_candidates * 2 ** self.num_stages

            layers = []
            for i in range(self.num_stages):
                if i > 0:
                    layers.append(nn.GELU())
                layers.append(nn.Conv1d(d_model, d_model, kernel_size=2, stride=2))
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)

        if self.mode == "avg":
            z = F.adaptive_avg_pool1d(x, self.num_candidates)
        elif self.mode == "max":
            z = F.adaptive_max_pool1d(x, self.num_candidates)
        elif self.mode == "learned":
            if x.shape[-1] != self.padded_length:
                if x.shape[-1] < self.padded_length:
                    x = F.pad(x, (0, self.padded_length - x.shape[-1]))
                else:
                    x = x[..., :self.padded_length]

            z = self.conv(x)
        else:
            raise NotImplementedError

        return z.transpose(1, 2)                 # (B, N, d)


def choose_num_candidates(window_frames: int, fps: float,
                          bpm_max: float = BPM_MAX) -> int:
    """Smallest N at or above the tempo floor that divides the window exactly."""
    floor = n_candidates_from_tempo(window_frames, fps, bpm_max)
    for n in range(floor, window_frames + 1):
        if window_frames % n == 0:
            return n
    return floor
