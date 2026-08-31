"""Prediction architecture (section 3): candidates, and equation (1)."""
import torch
import torch.nn as nn

from alignbeat.classes import NUM_CLASSES


# ---------------------------------------------------------------------------
# Prediction architecture (section 3)
# ---------------------------------------------------------------------------

class SubsetSelectionHead(nn.Module):
    """Encoder features -> N candidates -> (class logits, monotone times)."""

    def __init__(self, feature_size=256, hidden_size=256,
                 class_prior=(0.10, 0.30, 0.60),
                 class_attention_layers=0, class_attention_heads=4,
                 predict_precision=False):
        super(SubsetSelectionHead, self).__init__()

        self.input_norm = nn.LayerNorm(feature_size)
        self.trunk = nn.Linear(feature_size, hidden_size)

        self.candidate_attention = None
        if class_attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=class_attention_heads,
                dim_feedforward=hidden_size * 2, dropout=0.0,
                batch_first=True, norm_first=True)
            self.candidate_attention = nn.TransformerEncoder(layer, class_attention_layers)

        self.class_head = nn.Linear(hidden_size, 3) # JA: 3 classes: downbeat, beat, background
        self.regression_head = nn.Linear(hidden_size, 1)
        self.precision_head = nn.Linear(hidden_size, 1) if predict_precision else None

        self._initialize_weights(class_prior)

    def _initialize_weights(self, class_prior):
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

        prior = torch.tensor(class_prior, dtype=torch.float32)
        self.class_head.bias.data = torch.log(prior / prior.sum())

    def forward(self, x):
        """x: (B, C, N) candidate features from Downsample, one token per candidate."""
        z = self.input_norm(x.transpose(1, 2))      # (B, N, C)
        z = self.trunk(z)

        # Regression reads z directly and is therefore unaffected by section 10.2's
        # attention pass; only the classifier sees the contextualised features.
        r = self.regression_head(z).squeeze(dim=2)      # (B, N)
        t_hat = monotonic_times(r)

        z_class = z if self.candidate_attention is None else self.candidate_attention(z)
        class_logits = self.class_head(z_class)         # (B, N, 3)

        if self.precision_head is None:
            return class_logits, t_hat

        # Raw output u_j; the criterion applies b_j = b_min + softplus(u_j).
        return class_logits, t_hat, self.precision_head(z).squeeze(dim=2)


def monotonic_times(r, alpha=1e-3):
    """Equation (1) with the gap floor made explicit."""

    # JA: t_hat is in (0, 1] and of float32 type, in which any two values with less than
    # 1e-7 (~2^-23) difference are truncated to the same value. softplus(r) = e^r can
    # produce increments that are too small to be represented in float32
    N = r.shape[-1]
    inc = (1 - alpha) * torch.softmax(r, dim=-1) + alpha / N
    return torch.cumsum(inc, dim=-1)
