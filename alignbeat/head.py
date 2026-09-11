"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import F_MEASURE_TOLERANCE


# ---------------------------------------------------------------------------
# Prediction architecture (section 3)
# ---------------------------------------------------------------------------


class SubsetSelectionHead(nn.Module):
    """Encoder features -> N candidates -> (class logits, monotone times)."""

    def __init__(self, feature_size=256, hidden_size=256, attention_layers=0,
                 attention_heads=4, time_param="bounded",
                 window_seconds=30.0):
        super(SubsetSelectionHead, self).__init__()

        self.window_seconds = float(window_seconds)
        if time_param not in ("paper", "bounded", "floored"):
            raise ValueError(f"time_param must be paper|bounded|floored, got {time_param!r}")
        self.time_param = time_param

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

        # Candidate self-attention. The head is otherwise a per-candidate MLP, so
        # candidate j cannot see what j+1 is doing -- and 82% of false fires are within
        # two cells of a real one, i.e. the same beat claimed twice. DETR avoids NMS
        # precisely because its queries attend to each other and can back off; this is
        # that mechanism, on an anchored grid.
        #
        # The final LayerNorm is NOT optional here. norm_first normalises each
        # sublayer's INPUT, leaving the residual stream itself unnormalised, so what
        # class_head reads leaves the block several times longer than it entered
        # (measured previously: token norm 11.7 -> 46.5 at one layer, compounding with
        # depth). 42 of the 44 arms that ever ran this had it OFF. It is on here.
        self.candidate_attention = None
        if attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=attention_heads,
                dim_feedforward=hidden_size * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            self.candidate_attention = nn.TransformerEncoder(
                layer, attention_layers, norm=nn.LayerNorm(hidden_size))

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
        if self.time_param == "paper":
            t_hat = paper_times(r)                       # eq. (1), coupled, no gap floor
        elif self.time_param == "floored":
            t_hat = floored_times(r, self.window_seconds)   # coupled, gap >= 2 * tolerance
        else:
            t_hat = monotonic_times(r)                   # bounded per-candidate offset

        # Only the classifier sees the contextualised features; regression already read
        # z above, so timing is unaffected by the attention pass.
        z_class = z
        if self.candidate_attention is not None:
            # Self-attention is permutation-equivariant, so without a position signal
            # the classifier could see WHAT the other candidates look like but not
            # WHERE they are -- and "is my neighbour claiming this beat" is a question
            # about position. t_hat is already monotone in the index, so the index IS
            # the time order; sinusoidal features of it are the cheaper of the two.
            z_class = z + sinusoidal(
                torch.arange(z.shape[1], device=z.device, dtype=z.dtype)
                .unsqueeze(0).expand(z.shape[0], -1), z.shape[2])
            z_class = self.candidate_attention(z_class)

        class_logits = self.class_head(z_class)         # (B, N, 3)
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


def sinusoidal(position, dim):
    """Standard sinusoidal features of a (B, N) real position -> (B, N, dim)."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=position.device, dtype=position.dtype)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    angles = position.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)[..., :dim]


# Minimum gap between consecutive candidates, as a fraction of the window. Two
# detections can only be matched to the same reference beat if they lie within 2 x the
# metric tolerance of each other, so a floor of 2 x 70 ms makes that impossible by
# construction. 0.218% of annotated inter-beat intervals in the corpus are below it.
MIN_GAP_SECONDS = 2 * F_MEASURE_TOLERANCE


def floored_times(r, window_seconds, min_gap_seconds=MIN_GAP_SECONDS):
    """Equation (1) with a floor on each increment.

    Section 9.2 argues no de-duplication is needed because eq. (1) guarantees
    t_0 < t_1 < ... < t_{N-1}, so "no two reported detections can ever coincide or
    cross in time". That is true, and it is not the property the metric needs:
    mir_eval matches one estimate per reference inside a +-70 ms window, so two
    detections 22 ms apart are distinct, correctly ordered, and still cost a false
    positive. Measured on a trained model: duplicate pairs sit a median 22 ms apart,
    every one strictly ordered, and they are 9.2% of all fires.

    Strict ordering is a statement about points; the metric scores intervals. This
    extends the guarantee from ordering to tolerance-disjointness, in the same style --
    architectural, not penalised:

        t_j = sum_{k<=j} [ delta + (1 - N delta) softmax(r)_k ],   delta = min_gap/window

    Every increment is at least delta, so no two candidates can fall inside one
    tolerance window, so no reference beat can absorb two detections. Still a cumulative
    sum of strictly positive increments, still strictly increasing for every r, still
    normalised to (0, 1].

    The cost is a recall ceiling: true beats closer together than min_gap can never both
    be emitted. At 140 ms that is 0.218% of the corpus.
    """
    # float32 cumsum over N terms loses ~1.2 us of the floor by the last candidate,
    # which would leave the guarantee short of 2 x tolerance by a hair -- enough to be
    # false in principle, which defeats the purpose. Carry a 0.1 ms margin so the
    # realised gap clears the tolerance in float32 too.
    N = r.shape[-1]
    delta = (float(min_gap_seconds) + 1e-4) / float(window_seconds)
    slack = 1.0 - N * delta
    if slack <= 0:
        raise ValueError(
            f"N={N} candidates at a {min_gap_seconds:g}s floor need "
            f"{N * min_gap_seconds:g}s > {window_seconds:g}s of window")
    return torch.cumsum(delta + slack * torch.softmax(r, dim=-1), dim=-1)


def paper_times(r):
    """Equation (1) of Beat_DP_matching_final: the cumulative-sum reparameterisation.

        t_hat_j = sum_{k<=j} softplus(r_k) / sum_{k<N} softplus(r_k)

    Strictly increasing for every r since softplus > 0, and normalised to (0, 1].
    This is what monotonic_times replaced in 757d931. Kept so the three
    parameterisations can be compared on one recipe rather than across sessions.

    Note it has NO minimum gap: softplus(r) -> 0 is reachable, so two candidates can sit
    arbitrarily close while remaining strictly ordered.
    """
    inc = F.softplus(r)
    return torch.cumsum(inc, dim=-1) / inc.sum(dim=-1, keepdim=True)
