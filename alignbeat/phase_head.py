"""Continuous-phase prediction architecture: the third head, beside dense and subset.

Where SubsetSelectionHead emits a 3-way class posterior per candidate, this emits a
continuous bar phase in [0, 1). A downbeat is phase 0, the k-th beat of an L-beat bar is
k/L, and no meter is needed at the head at all -- L is inferred at decode time. The
timing branch is unchanged: equation (1)'s monotone cumulative-softplus construction,
shared verbatim with the subset head.

Ported from the standalone phase-time-regression arm. Two of its modules are dropped in
favour of what this repo already has, because they were reimplementations rather than
differences: its EncoderToCandidates/DownsampleStep pair is Downsample, and its
monotonic_time_reparam is monotonic_times (cumsum(softplus)/Z and cumsum(softplus/Z) are
the same expression).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import F_MEASURE_TOLERANCE
from alignbeat.downsample import Downsample
from alignbeat.head import monotonic_times, sinusoidal, softplus_inverse

# The Laplace scale the timing branch starts at, as a fraction of the window: one
# F-measure tolerance, the same quantity SubsetSelectionHead's precision bias encodes.
B_MIN = 1e-3 * (F_MEASURE_TOLERANCE / 30.0)


class SharedProjection(nn.Module):
    """One bottleneck feeding both branches, not a per-head reduction.

    The same shape as SubsetSelectionHead's trunk -- Linear, nonlinearity, LayerNorm --
    and for the same reason: without it every head is an affine read of the downsample
    output, and the downsample is itself linear. Placed before the branch splits, so the
    encoder and downsample keep their full d_model and only the heads' shared input is
    reduced.
    """

    def __init__(self, d_model: int, reduced_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, reduced_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(reduced_dim)

    def forward(self, z):
        return self.norm(self.act(self.proj(z)))                # (B, N, reduced_dim)


class PhaseHead(nn.Module):
    """(u_j, v_j) in R^2 -> phi_hat_j = atan2(v, u) / 2pi, always in [0, 1).

    The atan2 construction is what makes the output meter-free and wrap-free: no
    parameterisation can emit an out-of-range phase, and 0.99 and 0.01 are adjacent
    rather than a unit apart.
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, tilde_z):
        u, v = self.net(tilde_z).unbind(-1)
        return (torch.atan2(v, u) % (2 * math.pi)) / (2 * math.pi)      # (B, N)


class RegHead(nn.Module):
    """Raw per-candidate score r_j. The time itself is monotonic_times(r), a GLOBAL
    normalisation over all N together, so this module alone does not produce it."""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)                                   # (B, N)


class ScaleHead(nn.Module):
    """b_e = softplus(ScaleHead(z_bar)) + b_min, the learned Laplace timing scale.

    Per-FRAGMENT, not per-candidate: it reads the mean-pooled candidate features, on the
    argument that timing spread is a property of the whole fragment. That is the one
    place this arm deliberately has less freedom than SubsetSelectionHead's per-candidate
    b_j, and the diagnostic that motivated it is that localisation is already saturated
    (A = B = 98.1% on fold 0), so the extra freedom was not being used.
    """

    def __init__(self, d_model: int, hidden: int = 256, b_min: float = B_MIN,
                 b_0: float = None):
        super().__init__()
        self.b_min = b_min
        self.b_0 = b_0
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.reset_scale()

    def reset_scale(self):
        """Start at exactly b_0, so the warm start's release is continuous.

        The warm start pins b_e = b_0 with no gradient path to this head, so nothing fits
        it until it is released -- and a default-initialised last layer emits softplus(~0)
        ~ 0.69, about 280x b_0. Measured on the reference run: b_e = 0.589 the epoch after
        release, at which the timing weight 1/b falls from 428 to ~1.5 and the DP matches
        98% of events to a non-nearest candidate. Zero weight plus a bias at
        softplus_inverse(b_0 - b_min) makes release a no-op, the same construction
        SubsetSelectionHead uses for its precision bias.
        """
        if self.b_0 is None:
            return
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, softplus_inverse(max(self.b_0 - self.b_min, 1e-8)))

    def forward(self, z_bar):
        return F.softplus(self.net(z_bar).squeeze(-1)) + self.b_min      # (B,)


class PhaseSelectionHead(nn.Module):
    """Candidate features -> (phi_hat, t_hat, b_e)."""

    def __init__(self, feature_size=512, reduced_dim=256, hidden_size=256,
                 window_seconds=30.0,
                 phase_attention_layers=1, phase_attention_heads=4,
                 phase_attention_pos="none", phase_attention_final_norm=False,
                 b_min=B_MIN, warmup_epochs=5):
        super().__init__()
        self.window_seconds = float(window_seconds)
        # b_0 is one F-measure tolerance expressed in the window's own units, the same
        # constant SubsetSelectionHead's precision bias inverts through softplus.
        self.b_0 = F_MEASURE_TOLERANCE / self.window_seconds
        self.warmup_epochs = int(warmup_epochs)

        self.shared_proj = SharedProjection(feature_size, reduced_dim)

        # Only the phase branch is contextualised; the regression branch reads the
        # projected features directly, mirroring SubsetSelectionHead, where the
        # classifier sees the attention pass and the regressor does not.
        self.candidate_attention = None
        if phase_attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=reduced_dim, nhead=phase_attention_heads,
                dim_feedforward=reduced_dim * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            final_norm = nn.LayerNorm(reduced_dim) if phase_attention_final_norm else None
            self.candidate_attention = nn.TransformerEncoder(
                layer, phase_attention_layers, norm=final_norm)

        # Self-attention is permutation-equivariant: without this the phase branch can
        # see WHAT the other candidates look like but not WHERE they are, and bar phase
        # is a statement about position -- beats 2 and 3 of a bar sound alike; what tells
        # them apart is how many beats since the downbeat. Same two options, same reason,
        # as SubsetSelectionHead's class_attention_pos.
        if phase_attention_pos not in ("none", "index", "time"):
            raise ValueError(f"phase_attention_pos must be none|index|time, got "
                             f"{phase_attention_pos!r}")
        self.phase_attention_pos = phase_attention_pos

        self.phase_head = PhaseHead(reduced_dim, hidden_size)
        self.regression_head = RegHead(reduced_dim, hidden_size)
        self.precision_head = ScaleHead(reduced_dim, hidden_size, b_min, b_0=self.b_0)

        self._initialize_weights()

    def _initialize_weights(self):
        """Re-runnable: BeatThis applies a generic init after building the heads."""
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # t_hat starts as the uniform grid, exactly as in SubsetSelectionHead: r = 0
        # makes every softplus increment equal, so the candidates are evenly spaced and
        # the head begins from no timing claim at all.
        nn.init.zeros_(self.regression_head.net[-1].weight)
        nn.init.zeros_(self.regression_head.net[-1].bias)
        self.precision_head.reset_scale()

    def forward(self, x, epoch=None):
        """x: (B, C, N) candidate features, channel-first as the other heads take them."""
        z = self.shared_proj(x.transpose(1, 2))                          # (B, N, d)

        t_hat = monotonic_times(self.regression_head(z))                 # (B, N)

        tilde_z = z
        if self.candidate_attention is not None:
            if self.phase_attention_pos == "index":
                tilde_z = z + sinusoidal(
                    torch.arange(z.shape[1], device=z.device, dtype=z.dtype)
                    .unsqueeze(0).expand(z.shape[0], -1), z.shape[2])
            elif self.phase_attention_pos == "time":
                tilde_z = z + sinusoidal(t_hat * z.shape[1], z.shape[2])
            tilde_z = self.candidate_attention(tilde_z)
        phi_hat = self.phase_head(tilde_z)                               # (B, N)

        # Warm start: b is held at b_0 with no gradient path until warmup_epochs have
        # passed, so the timing branch is fitted at a fixed, known scale before the scale
        # itself becomes free. epoch=None means "already trained" (inference).
        z_bar = z.mean(dim=1)                                            # (B, d)
        if epoch is not None and epoch <= self.warmup_epochs:
            b_e = torch.full((z.shape[0],), self.b_0,
                             device=z.device, dtype=z.dtype)
        else:
            b_e = self.precision_head(z_bar)                             # (B,)

        return phi_hat, t_hat, b_e


class PhaseTimeHead(nn.Module):
    """Progressive downsample T -> N, then the continuous-phase head.

    The counterpart of SubsetHead, and it shares that class's Downsample verbatim, so
    the two arms read an identical candidate grid and differ only in what sits on top.
    """

    def __init__(self, input_dim, num_candidates,
                 downsample_mode="learned", train_length=1500, fps=50,
                 downsample_stages=None, reduced_dim=256, hidden_size=256,
                 phase_attention_layers=1, phase_attention_heads=4,
                 phase_attention_pos="none", phase_attention_final_norm=False,
                 warmup_epochs=5):
        super().__init__()

        self.downsample = Downsample(input_dim, num_candidates, downsample_mode,
                                     fragment_frames=train_length,
                                     stages=downsample_stages)
        self.num_candidates = self.downsample.num_candidates

        self.head = PhaseSelectionHead(
            feature_size=input_dim, reduced_dim=reduced_dim,
            hidden_size=hidden_size,
            window_seconds=train_length / float(fps),
            phase_attention_layers=phase_attention_layers,
            phase_attention_heads=phase_attention_heads,
            phase_attention_pos=phase_attention_pos,
            phase_attention_final_norm=phase_attention_final_norm,
            warmup_epochs=warmup_epochs)

    def forward(self, x, epoch=None):
        z = self.downsample(x)                          # (B, T, d) -> (B, N, d)
        phi_hat, t_hat, b_e = self.head(z.transpose(1, 2), epoch=epoch)
        return {"phi_hat": phi_hat, "t_hat": t_hat, "b_e": b_e,
                # Keys the dense readers expect; this arm has no framewise output.
                "beat": None, "downbeat": None}
