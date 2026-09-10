"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import NUM_CLASSES


# ---------------------------------------------------------------------------
# Prediction architecture (section 3)
# ---------------------------------------------------------------------------


class SubsetSelectionHead(nn.Module):
    """Encoder features -> N candidates -> (class logits, monotone times)."""

    def __init__(self, feature_size=256, hidden_size=256,
                 window_seconds=30.0):
        super(SubsetSelectionHead, self).__init__()

        self.window_seconds = float(window_seconds)

        self.input_norm = nn.LayerNorm(feature_size)

        # JA: Trunk is from the tree metaphor: one shared trunk, then branches. Here
        # self.trunk is the single shared path every candidate feature goes through,
        # and class_head and regression_head are the two branches
        # that split off it
        self.trunk = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.class_head = nn.Linear(hidden_size, 3) # JA: 3 classes: downbeat, beat, background
        self.regression_head = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Re-runnable: BeatThis applies a generic init after building the heads, which
        would otherwise overwrite every deliberate choice below."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)
        nn.init.zeros_(self.class_head.weight)

    def forward(self, x):
        """x: (B, C, N) candidate features from Downsample, one token per candidate."""
        z = self.input_norm(x.transpose(1, 2))      # (B, N, C)
        z = self.trunk(z)

        # JA: regression_head reduces 256-dim features to 1-d
        r = self.regression_head(z).squeeze(dim=2)      # (B, N)
        t_hat = monotonic_times(r)

        class_logits = self.class_head(z)               # (B, N, 3)
        return class_logits, t_hat


# Largest displacement of a clock from its cell centre, as a fraction of one cell.
# Below 1/2 no two clocks can cross, so ordering is architectural. At 0.45 a clock
# reaches +-72 ms at N=188, past the cell's own half-width, and the seam between two
# neighbours' reaches is 0.1 cell = 16 ms -- 8 ms from a clock, well inside the
# 70 ms tolerance.
MAX_OFFSET = 0.45


def monotonic_times(r, max_offset=MAX_OFFSET):
    """Each clock is its cell centre plus a bounded offset read from its own feature.

        t_hat_j = (j + 1/2) / N  +  max_offset * tanh(r_j) / N

    Consecutive centres are 1/N apart and every offset lies in (-max_offset/N,
    +max_offset/N), so t_hat_{j+1} - t_hat_j > (1 - 2 max_offset)/N > 0 for any r:
    strictly increasing by construction, not learned, and with no state that can
    fail it (tanh is bounded; there is no normaliser to underflow).

    This replaces the cumsum of normalised increments. That form coupled every clock
    to every increment before it: moving one clock 80 ms toward a beat meant its
    neighbour giving up ~17 ms (measured corr -0.61), and the reach saturated at ~68 ms
    however far the beat was (0.62 of the need beyond 100 ms). Here beat i's time
    gradient reaches r_j and nothing else, and the reach is max_offset of a cell.
    The price is that clocks can no longer bunch up inside one cell, which nothing
    measured used.
    """
    N = r.shape[-1]
    centre = (torch.arange(N, device=r.device, dtype=r.dtype) + 0.5) / N
    return centre + max_offset * torch.tanh(r) / N
