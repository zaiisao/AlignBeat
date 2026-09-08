"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import F_MEASURE_TOLERANCE, NUM_CLASSES


# ---------------------------------------------------------------------------
# Prediction architecture (section 3)
# ---------------------------------------------------------------------------

def softplus_inverse(y):
    """The u with softplus(u) = y. The head emits u; the criterion uses b = softplus(u)."""
    return math.log(math.expm1(y))


class SubsetSelectionHead(nn.Module):
    """Encoder features -> N candidates -> (class logits, monotone times)."""

    def __init__(self, feature_size=256, hidden_size=256,
                 window_seconds=30.0,
                 class_attention_layers=0, class_attention_heads=4,
                 class_attention_pos="none", class_attention_final_norm=False):
        super(SubsetSelectionHead, self).__init__()

        self.window_seconds = float(window_seconds)

        self.input_norm = nn.LayerNorm(feature_size)

        # JA: Trunk is from the tree metaphor: one shared trunk, then branches. Here
        # self.trunk is the single shared path every candidate feature goes through,
        # and class_head, regression_head and precision_head are the three branches
        # that split off it
        self.trunk = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.candidate_attention = None
        if class_attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=class_attention_heads,
                dim_feedforward=hidden_size * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            # norm_first=True normalizes each sublayer's INPUT; the residual stream
            # itself is never normalized, so what class_head reads leaves the block ~4x
            # longer than it entered (measured: token norm 11.7 -> 46.5 at one layer)
            # and the scale compounds with depth. The canonical pre-LN transformer ends
            # in a LayerNorm for exactly this reason. Opt-in, not default: adding the
            # parameters unconditionally would stop every existing checkpoint loading.
            final_norm = nn.LayerNorm(hidden_size) if class_attention_final_norm else None
            self.candidate_attention = nn.TransformerEncoder(
                layer, class_attention_layers, norm=final_norm)

        # Self-attention is permutation-equivariant, so without this the classifier can
        # see WHAT the other candidates look like but not WHERE they are -- and bar phase,
        # (p + i - 1) mod L, is a statement about position. Two ways to supply it:
        #   index: sinusoids over the candidate's ordinal j, i.e. its beat number
        #   time:  sinusoids over t_hat_j, i.e. when it actually is
        # The first counts beats, which is what phase needs; the second knows spacing,
        # which is what tempo needs. They are separable, so they are separate options.
        if class_attention_pos not in ("none", "index", "time"):
            raise ValueError(f"class_attention_pos must be none|index|time, got "
                             f"{class_attention_pos!r}")
        self.class_attention_pos = class_attention_pos

        self.class_head = nn.Linear(hidden_size, 3) # JA: 3 classes: downbeat, beat, background
        self.regression_head = nn.Linear(hidden_size, 1)
        self.precision_head = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Re-runnable: BeatThis applies a generic init after building the heads, which
        would otherwise overwrite every deliberate choice below."""
        attention_modules = set()
        if self.candidate_attention is not None:
            attention_modules = {id(m) for m in self.candidate_attention.modules()}

        for m in self.modules():
            if id(m) in attention_modules:
                continue

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
        nn.init.zeros_(self.precision_head.weight)

        # t_hat and the targets live on (0, 1] over the window, so the tolerance has to
        # cross into that unit before it can be a scale: 0.07 s of a 30 s window is
        # 0.00233. The head emits u and the criterion takes softplus(u), so invert it.
        b_initial = F_MEASURE_TOLERANCE / self.window_seconds
        nn.init.constant_(self.precision_head.bias, softplus_inverse(b_initial))

        # The bias sets b's overall scale, and the 70 ms init is only its starting point:
        # both bias and weight train from step 0 so the scale can follow the data. (It
        # used to be frozen, which held b_hat near twice the true residual.)

    def forward(self, x):
        """x: (B, C, N) candidate features from Downsample, one token per candidate."""
        z = self.input_norm(x.transpose(1, 2))      # (B, N, C)
        z = self.trunk(z)

        # Regression reads z directly and is therefore unaffected by section 10.2's
        # attention pass; only the classifier sees the contextualised features.

        # JA: regression_head reduces 256-dim features to 1-d
        r = self.regression_head(z).squeeze(dim=2)      # (B, N)
        t_hat = monotonic_times(r)

        z_class = z
        if self.candidate_attention is not None:
            if self.class_attention_pos == "index":
                z_class = z + sinusoidal(
                    torch.arange(z.shape[1], device=z.device, dtype=z.dtype)
                    .unsqueeze(0).expand(z.shape[0], -1), z.shape[2])
            elif self.class_attention_pos == "time":
                # t_hat is on (0, 1] over the window; scale to candidate-index units so
                # both encodings live at the same frequency range and are comparable.
                z_class = z + sinusoidal(t_hat * z.shape[1], z.shape[2])

            z_class = self.candidate_attention(z_class)

        class_logits = self.class_head(z_class)         # (B, N, 3)

        # b_j reads the trunk but never trains it: the precision term has no path into
        # the features the class and regression heads share (section 4.1.2's concern).
        b_hat_logit = self.precision_head(z.detach()).squeeze(dim=2)     # (B, N)
        b_hat = nn.functional.softplus(b_hat_logit)

        # Raw output u_j; the criterion applies b_j = b_min + softplus(u_j).
        return class_logits, t_hat, b_hat


def sinusoidal(position, dim):
    """Standard sinusoidal features of a (B, N) real position -> (B, N, dim).

    position need not be integral: `time` mode passes t_hat scaled into index units, so
    the same frequencies describe both "which beat" and "when".
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=position.device, dtype=position.dtype)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    angles = position.unsqueeze(-1) * freqs
    out = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if out.shape[-1] < dim:                       # odd dim: pad the last column
        out = torch.cat([out, out[..., :1] * 0], dim=-1)
    return out[..., :dim]


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
