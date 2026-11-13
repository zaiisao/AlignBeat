"""Prediction architecture (section 3): candidates, and equation (1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.constants import NUM_CLASSES
from alignbeat.model.downsample import Downsample

MAX_OFFSET = 0.45


def librosa_logmel_onset_strength(spectrogram):
    """Librosa-style positive spectral flux for (B, frames, mel) log-mel input.

    Matches librosa.onset.onset_strength(S=spectrogram.T, sr=22050,
    hop_length=441, n_fft=1024, center=True): positive first differences are
    averaged over mel bands and padded by the librosa framing offset.
    """
    if spectrogram.ndim != 3:
        raise ValueError(f"expected (batch, frames, mel), got {tuple(spectrogram.shape)}")
    x = spectrogram.float()
    flux = F.relu(x[:, 1:, :] - x[:, :-1, :]).mean(dim=-1)
    # librosa pads by lag + n_fft // (2 * hop_length) == 2 frames here.
    flux = F.pad(flux, (2, 0))[:, :x.shape[1]]
    return flux


def sample_onset_strength(onset_strength, times):
    """Sample a locally-maxed, 95th-percentile-normalized flux at candidate times."""
    if onset_strength.ndim != 2 or times.ndim != 2:
        raise ValueError("onset_strength and times must both have shape (batch, length)")
    if onset_strength.shape[0] != times.shape[0]:
        raise ValueError("onset_strength and times batch sizes differ")
    env = onset_strength.float().unsqueeze(1)
    env = F.max_pool1d(F.pad(env, (2, 2), mode="replicate"), 5, stride=1).squeeze(1)
    scale = torch.quantile(env, 0.95, dim=-1, keepdim=True).clamp_min(1e-6)
    env = torch.log1p(env / scale)

    position = times.float().clamp(0, 1) * (env.shape[-1] - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(env.shape[-1] - 1)
    fraction = position - left
    return env.gather(1, left) * (1 - fraction) + env.gather(1, right) * fraction


def calibrate_event_logits(class_logits, onset_feature, calibrator):
    """Adjust event-vs-background odds while preserving DB-vs-beat probabilities."""
    if class_logits.shape[-1] != 3 or onset_feature.shape != class_logits.shape[:-1]:
        raise ValueError("expected three class logits and one onset feature per candidate")
    logits = class_logits.float()
    event_logit = torch.logsumexp(logits[..., :2], dim=-1) - logits[..., 2]
    calibration_input = torch.stack((event_logit, onset_feature.float()), dim=-1)
    adjusted_event_logit = calibrator(calibration_input).squeeze(-1)

    log_conditional_class = F.log_softmax(logits[..., :2], dim=-1)
    log_event = F.logsigmoid(adjusted_event_logit)
    log_background = F.logsigmoid(-adjusted_event_logit)
    return torch.cat((log_event.unsqueeze(-1) + log_conditional_class,
                      log_background.unsqueeze(-1)), dim=-1)

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
                 train_length=1500, downsample_stages=None,
                 onset_adjustment=False):
        super().__init__()

        self.downsample = Downsample(input_dim, num_candidates,
                                     fragment_frames=train_length,
                                     stages=downsample_stages)
        self.head = SubsetSelectionHead(
            feature_size=input_dim, attention_layers=attention_layers)

        self.onset_adjustment = bool(onset_adjustment)
        self.event_calibrator = nn.Linear(2, 1) if self.onset_adjustment else None
        if self.onset_adjustment:
            self.initialize_event_calibrator()

        self.num_candidates = self.downsample.num_candidates

    def initialize_event_calibrator(self):
        """Start as an exact identity adjustment to the original event probabilities."""
        if self.event_calibrator is None:
            return
        nn.init.zeros_(self.event_calibrator.weight)
        nn.init.zeros_(self.event_calibrator.bias)
        with torch.no_grad():
            self.event_calibrator.weight[0, 0] = 1.0

    def forward(self, x, onset_strength=None):
        z = self.downsample(x) # (B, T, dim) -> (B, N, dim)
        out = self.head(z.transpose(1, 2))     # the head wants channel-first

        downsample_factor = self.downsample.time_scale(x.shape[1])
        t_hat = out[1] if downsample_factor == 1.0 else out[1] * downsample_factor

        class_logits = out[0]
        if self.onset_adjustment:
            if onset_strength is None:
                raise ValueError("onset_adjustment requires a framewise onset-strength input")
            onset_at_candidates = sample_onset_strength(onset_strength, t_hat)
            class_logits = calibrate_event_logits(
                class_logits, onset_at_candidates, self.event_calibrator)

        return {"class_logits": class_logits, "t_hat": t_hat}
