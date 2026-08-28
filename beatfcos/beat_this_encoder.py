"""
Beat This encoder wrapper (backbone_type="beat_this").

Only the frontend + transformer_blocks portion is taken as-is from BeatThis
(beat_this/model/beat_tracker.py); task_heads (SumHead, dense per-frame output)
is not used. Per the paper's Figure 1, this class's forward covers Stem ->
Frontend Block x3 -> Concat+Linear -> Transformer Block x6, and everything
after that (Task Heads) is replaced with our own downsample + subset head.
"""
from collections import OrderedDict

import torch
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from beatfcos import roformer
from beatfcos.beat_this_frontend import make_stem, make_frontend_block


class BeatThisEncoder(nn.Module):
    """
    Input : (B, T, spect_dim) log-mel spectrogram
    Output: (B, T, transformer_dim) -- T is preserved exactly (neither the
            frontend nor the transformer ever reduces the time axis; only the
            frequency axis inside the frontend shrinks).

    The BeatThis encoder with the task_heads block removed. Takes a
    (B, T, 128) log-mel spectrogram, passes it through the stem + 3 frontend
    blocks (the CNN + partial-attention part that only compresses frequency)
    to get (B, T, 512), then through a 6-layer standard self-attention
    transformer to finally output (B, T, 512).

    The time axis T is never reduced anywhere in this class -- it is kept
    exactly as-is; only the frequency axis inside the frontend shrinks. The
    T -> N downsampling we build afterward happens outside this class.
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
        partial_transformers: bool = True,
    ):
        super().__init__()

        # The RoPE object is created exactly once here, and this single object
        # is shared by both the frontend (F/T-direction attention) and the
        # 6-layer transformer_blocks.
        rotary_embed = RotaryEmbedding(head_dim)

        stem = make_stem(spect_dim, stem_dim)  # (B,T,128) -> (B,32,32,T)
        # The stem shrinks frequency from 128 -> 32, so this variable tracks
        # the "current number of frequency bands" for the frontend-block loop
        # below (the original reused/overwrote the spect_dim variable; here
        # it's kept as a separate freq_bands variable to avoid confusion --
        # logic is identical).
        freq_bands = spect_dim // 4

        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers,
                    head_dim,
                    rotary_embed,
                    dropout["frontend"],
                )
            )
            dim *= 2       # channels: 32 -> 64 -> 128
            freq_bands //= 2  # frequency: 32 -> 16 -> 8 -> 4
        frontend_blocks = nn.Sequential(*frontend_blocks)  # chain the 3 blocks into one module

        # after the 3 blocks, flatten (B,256,4,T) into (B,T,1024) (concat),
        # then compress with Linear(1024,512)
        concat = Rearrange("b c f t -> b t (c f)")
        linear = nn.Linear(dim * freq_bands, transformer_dim)
        self.frontend = nn.Sequential(
            # name stem/blocks/concat/linear and bundle them into one module;
            # self.frontend(x) runs the whole chain in forward().
            OrderedDict(stem=stem, blocks=frontend_blocks, concat=concat, linear=linear)
        )

        assert transformer_dim % head_dim == 0, "transformer_dim must be divisible by head_dim"
        n_heads = transformer_dim // head_dim  # 512 / 32 = 16 heads
        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,       # 6 layers
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,  # shares the same object as the frontend
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )

        # Equivalent to BeatThis.__init__'s self.apply(self._init_weights).
        # Without this, PyTorch's default init (Kaiming uniform for both
        # Linear/Conv2d) would be used instead. Applied recursively to every
        # submodule inside frontend + transformer_blocks.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # initialize with a normal distribution, mean 0, std 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # initialize with kaiming_normal (for relu, fan_out mode)
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # The original also had an nn.Embedding branch; dropped here since
        # this codebase has no Embedding layers.

    def forward(self, x):
        x = self.frontend(x)            # (B,T,128) -> (B,T,512)
        x = self.transformer_blocks(x)  # (B,T,512) -> (B,T,512), T unchanged
        return x
        # Identical to the original BeatThis.forward() minus the task_heads(x) call.
