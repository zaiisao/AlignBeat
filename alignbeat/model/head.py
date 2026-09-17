"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn

from alignbeat.constants import NUM_CLASSES
from alignbeat.model.downsample import Downsample

MAX_OFFSET = 0.45

def monotonic_times(r):
    """t_hat_j = (j + 1/2)/N + MAX_OFFSET * tanh(r_j)/N.

    Strictly increasing for any r, since centres are 1/N apart and offsets are bounded
    by MAX_OFFSET/N < 1/2N. Replaces eq. (1)'s cumsum of normalised increments, which
    coupled every candidate to every increment before it (moving one 80 ms cost its
    neighbour 17 ms, corr -0.61) and saturated its reach at ~68 ms.
    """
    N = r.shape[-1]
    centre = (torch.arange(N, device=r.device, dtype=r.dtype) + 0.5) / N
    return centre + MAX_OFFSET * torch.tanh(r) / N

def sinusoidal(position, dim):
    """Standard sinusoidal features of a (B, N) real position -> (B, N, dim)."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=position.device, dtype=position.dtype)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    angles = position.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)[..., :dim]

class SubsetSelectionHead(nn.Module):
    """Encoder features -> N candidates -> (class logits, monotone times)."""

    def __init__(self, feature_size=256, hidden_size=256,
                 attention_layers=0, attention_heads=4):
        super(SubsetSelectionHead, self).__init__()

        self.input_norm = nn.LayerNorm(feature_size)

        # JA: Trunk is from the tree metaphor: one shared trunk, then branches. Here
        # self.trunk is the single shared path every candidate feature goes through,
        # and class_head and regression_head are the two branches that split off it
        self.trunk = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.candidate_attention = None
        if attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=attention_heads,
                dim_feedforward=hidden_size * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            self.candidate_attention = nn.TransformerEncoder(
                layer, attention_layers, norm=nn.LayerNorm(hidden_size))

        self.class_head = nn.Linear(hidden_size, NUM_CLASSES)
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

        z_class = z
        if self.candidate_attention is not None:
            z_class = z + sinusoidal(
                torch.arange(z.shape[1], device=z.device, dtype=z.dtype)
                .unsqueeze(0).expand(z.shape[0], -1), z.shape[2])
            z_class = self.candidate_attention(z_class)

        class_logits = self.class_head(z_class)         # (B, N, 3)
        return class_logits, t_hat

class SubsetHead(nn.Module):
    """Progressive downsample T -> N, then the order-preserving alignment head."""

    def __init__(self, input_dim, num_candidates, attention_layers=0,
                 train_length=1500, downsample_stages=None):
        super().__init__()

        self.downsample = Downsample(input_dim, num_candidates,
                                     fragment_frames=train_length,
                                     stages=downsample_stages)
        self.head = SubsetSelectionHead(
            feature_size=input_dim, attention_layers=attention_layers)

        self.num_candidates = self.downsample.num_candidates

    def forward(self, x):
        z = self.downsample(x) # (B, T, dim) -> (B, N, dim)
        out = self.head(z.transpose(1, 2))     # the head wants channel-first

        downsample_factor = self.downsample.time_scale(x.shape[1])
        t_hat = out[1] if downsample_factor == 1.0 else out[1] * downsample_factor

        return {"class_logits": out[0], "t_hat": t_hat}
