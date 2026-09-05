"""
Model definitions for the Beat This! beat tracker.
"""

import contextlib

from alignbeat.downsample import Downsample
from alignbeat.head import SubsetSelectionHead
from collections import OrderedDict

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from beat_this.model import roformer
from beat_this.utils import replace_state_dict_key


class BeatThis(nn.Module):
    """A neural network model for beat tracking. It is composed of three main components:"""

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
        sum_head: bool = True,
        partial_transformers: bool = True,
        head_type: str = "dense",
        num_candidates: int = None,
        downsample_mode: str = "learned",
        train_length: int = 1500,
        fps: int = 50,
        downsample_stages: int = None,
        class_attention_layers: int = 0,
        class_attention_heads: int = 4,
        class_attention_pos: str = "none",
        class_attention_final_norm: bool = False,
    ):
        super().__init__()
        # shared rotary embedding for frontend blocks and transformer blocks
        rotary_embed = RotaryEmbedding(head_dim)

        # create the frontend
        # - stem
        stem = self.make_stem(spect_dim, stem_dim)
        spect_dim //= 4  # frequencies were convolved with stride 4
        # - three frontend blocks
        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                self.make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers,
                    head_dim,
                    rotary_embed,
                    dropout["frontend"],
                )
            )
            dim *= 2
            spect_dim //= 2  # frequencies were convolved with stride 2
        frontend_blocks = nn.Sequential(*frontend_blocks)
        # - linear projection to transformer dimensionality
        concat = Rearrange("b c f t -> b t (c f)")
        linear = nn.Linear(dim * spect_dim, transformer_dim)
        self.frontend = nn.Sequential(
            OrderedDict(stem=stem, blocks=frontend_blocks, concat=concat, linear=linear)
        )

        # create the transformer blocks
        assert (
            transformer_dim % head_dim == 0
        ), "transformer_dim must be divisible by head_dim"
        n_heads = transformer_dim // head_dim
        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )

        # create the output heads
        if head_type == "subset":
            if num_candidates is None:
                raise ValueError(
                    "head_type='subset' needs num_candidates; launch_scripts/train.py "
                    "derives it from --bpm_max and --train_length")
            # JA: This is the bridge to our AlignBeat architecture
            self.task_heads = SubsetHead(
                transformer_dim, num_candidates=num_candidates,
                downsample_mode=downsample_mode, train_length=train_length, fps=fps,
                downsample_stages=downsample_stages,
                class_attention_layers=class_attention_layers,
                class_attention_heads=class_attention_heads,
                class_attention_pos=class_attention_pos,
                class_attention_final_norm=class_attention_final_norm)
        elif sum_head:
            self.task_heads = SumHead(transformer_dim)
        else:
            self.task_heads = Head(transformer_dim)

        # init all weights
        self.apply(self._init_weights)

        # ...then restore the subset head's own initialisation, which the generic pass
        # above would otherwise overwrite: the class prior on the classifier bias, the
        # zeroed regression/class/precision weights, and the precision head's scale.
        if isinstance(self.task_heads, SubsetHead):
            self.task_heads.head._initialize_weights()

    @staticmethod
    def make_stem(spect_dim: int, stem_dim: int) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                rearrange_tf=Rearrange("b t f -> b f t"),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(
                    in_channels=1,
                    out_channels=stem_dim,
                    kernel_size=(4, 3),
                    stride=(4, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                bn2d=nn.BatchNorm2d(stem_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        partial_transformers: bool = True,
        head_dim: int | None = 32,
        rotary_embed: RotaryEmbedding | None = None,
        dropout: float = 0.1,
    ) -> nn.Module:
        if partial_transformers and (head_dim is None or rotary_embed is None):
            raise ValueError(
                "Must specify head_dim and rotary_embed for using partial_transformers"
            )
        return nn.Sequential(
            OrderedDict(
                partial=(
                    PartialFTTransformer(
                        dim=in_dim,
                        dim_head=head_dim,
                        n_head=in_dim // head_dim,
                        rotary_embed=rotary_embed,
                        dropout=dropout,
                    )
                    if partial_transformers
                    else nn.Identity()
                ),
                # conv block
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                # out_channels : 64, 128, 256
                # freqs : 16, 8, 4 (due to the stride=2)
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.transformer_blocks(x)
        x = self.task_heads(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class PartialRoformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block either only across frequencies or only across time.
    Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        direction: str,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        self.direction = direction[0].lower()
        if self.direction not in "ft":
            raise ValueError(f"direction must be F or T, got {direction}")
        self.attn = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ff = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        if self.direction == "f":
            pattern = "(b t) f c"
        elif self.direction == "t":
            pattern = "(b f) t c"
        x = rearrange(x, f"b c f t -> {pattern}")
        x = x + self.attn(x)
        x = x + self.ff(x)
        x = rearrange(x, f"{pattern} -> b c f t", b=b)
        return x


class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time. Same
    as applying two PartialRoformer() in sequence, but encapsulated in a single
    module. Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = roformer.FeedForward(dim, dropout=dropout)
        # time directed partial transformer
        self.attnT = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x


class SubsetHead(nn.Module):
    """Progressive downsample T -> N, then the order-preserving alignment head."""

    def __init__(self, input_dim, num_candidates,
                 downsample_mode="learned", train_length=1500, fps=50,
                 downsample_stages=None,
                 class_attention_layers=0, class_attention_heads=4,
                 class_attention_pos="none", class_attention_final_norm=False):
        super().__init__()

        self.downsample = Downsample(input_dim, num_candidates,
                                     downsample_mode,
                                     fragment_frames=train_length,
                                     stages=downsample_stages)

        # One token out of the downsample is one candidate into the heads, so there is a
        # single N. The halvings decide it (1500 -> 188) and the tempo floor is only a
        # lower bound they must clear, not a target to pool down to -- tempo augmentation
        # can push a 30 s window past the floor's 170 events, so the slack above it is
        # useful rather than waste. The criterion reads N from the logits' shape.
        self.num_candidates = self.downsample.num_candidates

        self.head = SubsetSelectionHead(
            feature_size=input_dim,
            window_seconds=train_length / float(fps),
            class_attention_layers=class_attention_layers,
            class_attention_heads=class_attention_heads,
            class_attention_pos=class_attention_pos,
            class_attention_final_norm=class_attention_final_norm)

    def forward(self, x):
        z = self.downsample(x) # (B, T, dim) -> (B, N, dim)
        out = self.head(z.transpose(1, 2))     # the head wants channel-first

        # The candidate grid spans padded_length frames, so on a short input t_hat is
        # relative to the padding rather than to x. Rescale so callers can keep reading
        # it as a fraction of what they passed in; candidates past 1.0 sit in the pad.
        downsample_factor = self.downsample.time_scale(x.shape[1])
        t_hat = out[1] if downsample_factor == 1.0 else out[1] * downsample_factor
        b_hat = out[2]

        return {"class_logits": out[0], "t_hat": t_hat, "b_hat": b_hat}


class SumHead(nn.Module):
    """
    A PyTorch module that produces the final beat and downbeat prediction logits.
    The beats are a sum of all beats and all downbeats predictions, to reduce the prediction
    of downbeats which are not beats.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x):
        beat_downbeat = self.beat_downbeat_lin(x)
        # separate beat from downbeat
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)
        # aggregate beats and downbeats prediction
        # autocast to float16 disabled to avoid numerical issues causing NaNs
        if hasattr(
            torch.amp, "is_autocast_available"
        ) and not torch.amp.is_autocast_available(beat.device.type):
            # but do not try disabling if the device does not support autocast
            disable_autocast = contextlib.nullcontext()
        else:
            disable_autocast = torch.autocast(beat.device.type, enabled=False)
        with disable_autocast:
            beat = beat.float() + downbeat.float()
        return {"beat": beat, "downbeat": downbeat}


class Head(nn.Module):
    """
    A PyToch module that produces the final beat and downbeat prediction logits with independent linear layers outputs.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x):
        beat_downbeat = self.beat_downbeat_lin(x)
        # separate beat from downbeat
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)
        return {"beat": beat, "downbeat": downbeat}
