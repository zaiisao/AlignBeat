"""
Frontend (Stem + 3 Frontend Blocks) for the Beat This encoder, copied from
CPJKU/beat_this (beat_this/model/beat_tracker.py) as part of the Beat This
encoder port (backbone_type="beat_this"). No logic changes.

make_stem / make_frontend_block were @staticmethod on the BeatThis class in the
original; ported here as plain functions so they can be reused to build our own
encoder wrapper without the rest of BeatThis (task_heads etc.).
"""

from collections import OrderedDict

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from beatfcos import roformer

"""
Why BatchNorm1d comes first: each of the 128 frequency bands has a completely
different energy distribution (low bands vs. high bands). Normalizing "per
band" to homogenize them before the conv is what the paper calls "homogenise
them" (Beat This, Section 3.1.1 "Frontend").

kernel = (4, 3), stride = (4, 1): the first number is the frequency axis, the
second is the time axis. The frequency axis is aggressively shrunk with
stride 4 (128 -> 32), while the time axis uses stride 1, i.e. T is never
touched -- Beat This never reduces the time axis.

Result: (B, T, 128) -> (B, 32, 32, T) -- channels=32, freq=32, time T unchanged.
"""
def make_stem(spect_dim: int, stem_dim: int) -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            rearrange_tf=Rearrange("b t f -> b f t"), # (B,T,128) -> (B,128,T): channel axis must sit in the second dimension by convention.
            bn1d=nn.BatchNorm1d(spect_dim), # once axes are aligned, normalize each of the 128 frequency bands independently
            add_channel=Rearrange("b f t -> b 1 f t"), # (B, 128, T) -> (B, 1, 128, T): add the channel dimension Conv2d expects
            conv2d=nn.Conv2d(
                in_channels=1,
                out_channels=stem_dim,
                kernel_size=(4, 3),
                stride=(4, 1),
                padding=(0, 1),
                bias=False,
            ), # compress only the frequency axis by 4 (128 -> 32), channels 1 -> 32
            bn2d=nn.BatchNorm2d(stem_dim),
            activation=nn.GELU(),
        )
    )

"""
Attention runs first, then the conv compresses -- i.e. "consolidate context
at the current resolution, then compress." Since compressing loses
information, attention mixes in context beforehand so it isn't lost.

n_head is computed as in_dim // head_dim. Because channels double at each
block (32 -> 64 -> 128), the head count automatically grows 1 -> 2 -> 4.

The conv also uses stride=(2, 1): again the time axis is untouched and only
the frequency axis is halved. Called 3 times: (32, 32, T) -> (64, 16, T) ->
(128, 8, T) -> (256, 4, T)
"""
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


class PartialRoformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block either only across frequencies or only across time.
    Returns a tensor of the same shape as the input.

    The input is (B, C, F, T), while roformer.Attention/FeedForward expect the
    usual (batch, seq_len, dim) shape NLP transformers take. So which axis
    counts as "seq_len" is swapped in and out via rearrange each time,
    letting the same attention code be reused.

    Attending over F and T jointly would be too expensive; splitting the axes
    and attending separately is much cheaper. Also, the F direction carries
    harmony/timbre information while the T direction carries
    rhythm/periodicity -- different enough in character that learning them
    separately helps.
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
        self.direction = direction[0].lower()  # "f" or "t", picked once at construction
        if self.direction not in "ft":
            raise ValueError(f"direction must be F or T, got {direction}")
        self.attn = roformer.Attention(  # single direction's attention -- roformer.py's generic engine
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ff = roformer.FeedForward(dim, dropout=dropout)  # this direction's dedicated FeedForward

    # Not used by make_frontend_block (which calls PartialFTTransformer directly,
    # doing both directions in one class) -- kept as the single-direction building
    # block PartialFTTransformer's docstring refers back to.

    """
    In the frequency direction, batch and time (T) are folded together into
    a "fake batch," with frequency (F) used as the sequence length and
    channels (C) as the attention feature dimension -- i.e. each time frame
    is treated as an independent sample, and only the 32 frequency bands
    within it attend to each other.

    In the time direction, batch and frequency are folded together instead,
    with time (T) used as the sequence -- each frequency band is treated as
    an independent sample, and the time flow within it attends to itself.
    """
    def forward(self, x):
        b = len(x)
        if self.direction == "f":
            pattern = "(b t) f c"  # fold batch+time, frequency becomes the sequence
        elif self.direction == "t":
            pattern = "(b f) t c"  # fold batch+frequency, time becomes the sequence
        x = rearrange(x, f"b c f t -> {pattern}")
        x = x + self.attn(x)  # residual attention
        x = x + self.ff(x)    # residual FeedForward
        x = rearrange(x, f"{pattern} -> b c f t", b=b)  # restore original (B,C,F,T) shape
        return x


class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time. Same
    as applying two PartialRoformer() in sequence, but encapsulated in a single
    module. Returns a tensor of the same shape as the input.

    A block that applies attention to a (B,C,F,T) input in two stages. First
    along the F axis: the 32 frequency bands within the same time frame
    attend to each other, capturing "which frequencies are sounding together
    right now" (harmony/timbre relationships). Then along the T axis: the
    1500 time frames within the same frequency band attend to each other,
    capturing "how this frequency component changes over time"
    (rhythm/periodicity). The two directions are attended to separately
    rather than jointly because of compute cost, and because the F and T axes
    carry different kinds of information. Once both directions are done, the
    tensor is reshaped back to (B,C,F,T) for the next stage, the
    frequency-compressing conv2d.
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
        self.attnF = roformer.Attention( # F-direction (frequency axis) attention -- dedicated parameters
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = roformer.FeedForward(dim, dropout=dropout) # F-direction dedicated FeedForward
        # time directed partial transformer
        self.attnT = roformer.Attention( # T-direction (time axis) attention
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = roformer.FeedForward(dim, dropout=dropout) # T-direction dedicated FeedForward

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c") # fold batch + time into a "fake batch", frequency becomes the sequence
        x = x + self.attnF(x) # F-direction attention, residual -- captures "harmony/timbre relationships"
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b) # fold batch + frequency, time becomes the sequence
        x = x + self.attnT(x) # T-direction attention, residual -- captures "rhythm/periodicity"
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x
