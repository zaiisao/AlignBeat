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

    def __init__(self, d_model: int, num_candidates: int,
                 downsampled_seq_size: int,
                 mode: str = "learned", window_frames: int = None,
                 stages: int = None):
        super().__init__()

        if mode not in ("learned", "avg", "max"):
            raise ValueError(f"mode must be learned|avg|max, got {mode!r}")

        self.mode = mode
        self.num_candidates = num_candidates
        self.conv = None
        self.downsampled_seq_size = downsampled_seq_size

        if mode == "learned":
            if window_frames is None:
                raise ValueError("mode='learned' needs window_frames (T) to size the "
                                 "stride; avg/max are adaptive and do not")

            # Reduce the length in several small strided steps (1500 -> 750 -> 250)
            # instead of collapsing T/N frames in one: the same receptive field with
            # fewer parameters and a nonlinearity in between. The strides must MULTIPLY
            # to T/N exactly -- rounding to powers of two instead would pad the window
            # with silence and silently rescale what t_hat = 1 means.
            if stages is not None:
                # JA: We apply half downsampling three times assuming the original
                # length of the spectrogram is 1500.
                self.strides = [2] * int(stages)
            else:
                self.strides = factor_strides(window_frames, num_candidates)
            self.num_stages = len(self.strides)
            self.kernel = self.strides[0]

            # JA: The size of the sequence to be downsampled. It is the original
            # length plus padding size
            self.padded_length = downsampled_seq_size * math.prod(self.strides)
            self.window_frames = int(window_frames)

            layers = []
            for i, stride in enumerate(self.strides):
                if i > 0:
                    layers.append(nn.GELU())
                layers.append(nn.Conv1d(d_model, d_model, kernel_size=stride,
                                        stride=stride))
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)

        if self.mode == "avg":
            z = F.adaptive_avg_pool1d(x, self.downsampled_seq_size)
        elif self.mode == "max":
            z = F.adaptive_max_pool1d(x, self.downsampled_seq_size)
        elif self.mode == "learned":
            length = x.shape[-1]
            if length > self.padded_length:
                # Truncating would drop ground-truth events entirely, and an event the
                # candidates cannot reach is unmatchable rather than merely mispredicted.
                raise ValueError(
                    f"input of {length} frames exceeds the {self.padded_length} this "
                    f"Downsample was built for; rebuild it with window_frames={length} "
                    f"or split the input")

            if length < self.padded_length:
                x = F.pad(x, (0, self.padded_length - length))

            z = self.conv(x)
        else:
            raise NotImplementedError

        return z.transpose(1, 2)                 # (B, N, d)

    def time_scale(self, input_frames: int) -> float:
        """Frames-of-input per unit of t_hat, as a fraction of the input.

        The candidate grid spans padded_length frames, so a head reading it emits times
        relative to THAT, not to the caller's input. Whenever the two differ -- a short
        final chunk, a piece below one window -- t_hat must be multiplied by this to
        become input-relative, and candidates landing beyond 1.0 sit in the padding and
        are not real detections.
        """
        if self.mode != "learned" or input_frames == self.padded_length:
            return 1.0
        return self.padded_length / float(input_frames)


def halved_candidates(window_frames: int, stages: int) -> int:
    """N after `stages` halvings of T, rounding up at each odd length."""
    n = int(window_frames)
    for _ in range(stages):
        n = -(-n // 2)
    return n

def stages_from_tempo(window_frames: int, fps: float,
                      bpm_max: float = BPM_MAX) -> tuple:
    """Most halvings of T whose N still covers the tempo floor.

    N only has to be >= the densest event count a window can hold; beyond that every
    extra candidate is a background classification. So halve until one more halving
    would drop below that floor: T=1500 at 50 fps floors at 170, and 750 -> 375 -> 188
    all clear it while 94 does not, giving 3 stages. Returns (stages, N).
    """
    num_candidates = n_candidates_from_tempo(window_frames, fps, bpm_max)
    stages = 0
    while halved_candidates(window_frames, stages + 1) >= num_candidates:
        stages += 1
    return stages, halved_candidates(window_frames, stages), num_candidates

def factor_strides(window_frames: int, num_candidates: int) -> list:
    """ceil(T/N) split into per-stage strides, smallest first, multiplying to it exactly.

    T/N need not be a power of two (1500/250 = 6), so "halve every layer" cannot be
    taken literally: 6 becomes (2, 3) rather than 8. Rounding up to 8 would pad the
    window with 500 frames of silence and silently rescale what t_hat = 1 means. When N
    divides T -- which choose_num_candidates() ensures -- the strides reach exactly T
    and no padding happens at all.
    """
    factor = -(-window_frames // num_candidates)          # ceil
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


def choose_num_candidates(window_frames: int, fps: float,
                          bpm_max: float = BPM_MAX) -> int:
    """Smallest N at or above the tempo floor that divides the window exactly."""
    # JA: This computes the max N; for example, assuming T is 1500, this is 170
    min_num_candidates = n_candidates_from_tempo(window_frames, fps, bpm_max)
    for n in range(min_num_candidates, window_frames + 1):
        if window_frames % n == 0:
            return n
    return min_num_candidates
