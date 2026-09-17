"""Encoder output (B, T, d) -> candidate features (B, N, d)."""
import math

import torch.nn.functional as F
from torch import nn

from alignbeat.constants import BPM_MAX


def n_candidates_from_tempo(fragment_frames: int, fps: float,
                            bpm_max: float = BPM_MAX) -> int:
    """N := BPM_max * D, the most events a window of D minutes can contain."""
    return math.ceil(bpm_max * (fragment_frames / fps) / 60.0)


class Downsample(nn.Module):
    """(B, T, d) -> (B, N, d) by one strided operator."""

    def __init__(self, d_model: int, num_candidates: int,
                 mode: str = "learned", fragment_frames: int = None,
                 stages: int = None):
        super().__init__()

        if mode not in ("learned", "avg", "max"):
            raise ValueError(f"mode must be learned|avg|max, got {mode!r}")

        self.mode = mode
        self.num_candidates = num_candidates
        self.conv = None

        if fragment_frames is None:
            raise ValueError("mode='learned' needs fragment_frames (T) to size the "
                                "stride; avg/max are adaptive and do not")

        if stages is not None:
            # JA: We apply half downsampling three times assuming the original
            # length of the spectrogram is 1500.
            self.strides = [2] * int(stages)
        else:
            self.strides = factor_strides(fragment_frames, num_candidates)

        # JA: The size of the sequence to be downsampled. It is the original
        # length plus padding size
        self.padded_length = num_candidates * math.prod(self.strides)

        if mode == "learned":
            # JA: Reduce the length in several small strided steps (1500 -> 750 -> 250)
            # instead of collapsing T/N frames in one
            layers = []
            for i, stride in enumerate(self.strides):
                if i > 0: layers.append(nn.GELU())
                layers.append(nn.Conv1d(d_model, d_model, kernel_size=stride,
                                        stride=stride))

            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)

        length = x.shape[-1]
        if length > self.padded_length:
            # Truncating would drop ground-truth events entirely, and an event the
            # candidates cannot reach is unmatchable rather than merely mispredicted.
            raise ValueError(
                f"input of {length} frames exceeds the {self.padded_length} this "
                f"Downsample was built for; rebuild it with fragment_frames={length} "
                f"or split the input")

        if length < self.padded_length:
            x = F.pad(x, (0, self.padded_length - length))

        if self.mode == "avg":
            z = F.adaptive_avg_pool1d(x, self.num_candidates)
        elif self.mode == "max":
            z = F.adaptive_max_pool1d(x, self.num_candidates)
        elif self.mode == "learned":
            z = self.conv(x)
        else:
            raise NotImplementedError

        return z.transpose(1, 2)                 # (B, N, d)

    def time_scale(self, input_frames: int) -> float:
        """Frames-of-input per unit of t_hat, as a fraction of the input."""
        if input_frames == self.padded_length:
            return 1.0

        return self.padded_length / float(input_frames)

def halved_candidates(fragment_frames: int, stages: int) -> int:
    """N after `stages` halvings of T, rounding up at each odd length."""
    n = int(fragment_frames)
    for _ in range(stages):
        n = -(-n // 2)
    return n

def stages_from_tempo(fragment_frames: int, fps: float,
                      bpm_max: float = BPM_MAX) -> tuple:
    """Most halvings of T whose N still covers the tempo floor.

    Returns (stages, N, tempo_floor)."""
    tempo_floor = n_candidates_from_tempo(fragment_frames, fps, bpm_max)
    stages = 0
    while halved_candidates(fragment_frames, stages + 1) >= tempo_floor:
        stages += 1

    return stages, halved_candidates(fragment_frames, stages), tempo_floor

def factor_strides(fragment_frames: int, num_candidates: int) -> list:
    """ceil(T/N) split into per-stage strides, smallest first, multiplying to it exactly.

    T/N need not be a power of two (1500/250 = 6), so "halve every layer" cannot be
    taken literally: 6 becomes (2, 3) rather than 8. Rounding up to 8 would pad the
    window with 500 frames of silence and silently rescale what t_hat = 1 means. When N
    divides T -- which choose_num_candidates() ensures -- the strides reach exactly T
    and no padding happens at all.
    """
    factor = -(-fragment_frames // num_candidates)          # ceil
    strides = []
    divisor = 2
    while divisor * divisor <= factor:
        while factor % divisor == 0:
            strides.append(divisor)
            factor //= divisor
        divisor += 1
    if factor > 1:
        strides.append(factor)
    return strides or [1]
