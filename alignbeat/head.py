"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn

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
                 class_prior=(0.10, 0.30, 0.60), window_seconds=30.0,
                 class_attention_layers=0, class_attention_heads=4,
                 class_attention_pos="none"):
        super(SubsetSelectionHead, self).__init__()

        self.window_seconds = float(window_seconds)
        self.class_prior = class_prior

        self.input_norm = nn.LayerNorm(feature_size)
        self.trunk = nn.Linear(feature_size, hidden_size)

        self.candidate_attention = None
        if class_attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=class_attention_heads,
                dim_feedforward=hidden_size * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            self.candidate_attention = nn.TransformerEncoder(layer, class_attention_layers)

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

        prior = torch.tensor(self.class_prior, dtype=torch.float32)
        with torch.no_grad():
            self.class_head.bias.copy_(torch.log(prior / prior.sum()))

        # t_hat and the targets live on (0, 1] over the window, so the tolerance has to
        # cross into that unit before it can be a scale: 0.07 s of a 30 s window is
        # 0.00233. The head emits u and the criterion takes softplus(u), so invert it.
        b_initial = F_MEASURE_TOLERANCE / self.window_seconds
        nn.init.constant_(self.precision_head.bias, softplus_inverse(b_initial))

        # JA: Both weights and bias are frozen
        self.precision_head.requires_grad_(False)

    def forward(self, x):
        """x: (B, C, N) candidate features from Downsample, one token per candidate."""
        z = self.input_norm(x.transpose(1, 2))      # (B, N, C)
        z = self.trunk(z)

        # Regression reads z directly and is therefore unaffected by section 10.2's
        # attention pass; only the classifier sees the contextualised features.
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

        b_hat_logit = self.precision_head(z).squeeze(dim=2)
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


def monotonic_times(r, alpha=1e-3):
    """Equation (1) with the gap floor made explicit."""

    # JA: t_hat is in (0, 1] and of float32 type, in which any two values with less than
    # 1e-7 (~2^-23) difference are truncated to the same value. softplus(r) = e^r can
    # produce increments that are too small to be represented in float32
    N = r.shape[-1]
    inc = (1 - alpha) * torch.softmax(r, dim=-1) + alpha / N
    return torch.cumsum(inc, dim=-1)
