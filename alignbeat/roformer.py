"""
Transformer with rotary position embedding, adapted from Phil Wang's repository
at https://github.com/lucidrains/BS-RoFormer (under MIT License).

Copied verbatim from CPJKU/beat_this (beat_this/model/roformer.py) as part of the
Beat This encoder port (backbone_type="beat_this"). No logic changes.

A generic building-block module holding only the "engine parts" of a
self-attention transformer -- the piece of the Beat This architecture
responsible for "how attention is computed" throughout.

RMSNorm: pre-norm (normalizes input before each attention/FeedForward block)
FeedForward: the usual MLP after attention (expand channels -> GELU -> shrink)
Attend: computes the actual attention score from Q, K, V
Attention: applies RoPE (positional info) to Q/K and includes gating -- one
    complete self-attention layer
Transformer: stacks Attention + FeedForward in residual blocks across layers

Has no notion of what the attention is applied to (frequency axis vs. time
axis, frontend vs. main transformer) -- purely reusable attention logic.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module, ModuleList

# helper functions


def exists(val):
    return val is not None


# norm


class RMSNorm(Module):
    def __init__(self, size, dim=-1):
        super().__init__()
        self.scale = size**0.5  # sqrt(dim) rescaling factor (RMSNorm convention)
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))  # learnable per-channel scale
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim) * self.scale * self.gamma  # L2-normalize along `dim`, then rescale


# feedforward


class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
        dim_out=None,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        dim_inner = int(dim * mult)  # hidden size = dim * mult (paper: "four times the channel count")
        self.activation = nn.GELU()
        self.net = nn.Sequential(
            RMSNorm(dim),                    # pre-norm
            nn.Linear(dim, dim_inner),        # expand
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),    # shrink back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# attention


class Attend(nn.Module):
    def __init__(self, dropout=0.0, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale  # optional override of the default 1/sqrt(d) attention scale

    def forward(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)  # rescale q so the effective attention scale becomes self.scale

        # Flash SDP CUDA kernel fails when batch > 65535 (PartialFTTransformer reshapes to B*T).
        # For small seq_len (freq axis, seq=32), math backend is fine memory-wise.
        if q.shape[0] > 65535:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                return F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0
                )
        return F.scaled_dot_product_attention(  # standard scaled dot-product attention (Q,K,V -> weighted sum)
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        gating=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head  # total width across all heads

        self.rotary_embed = rotary_embed  # shared RoPE object, applied to Q/K only

        self.attend = Attend(dropout=dropout)

        self.norm = RMSNorm(dim)  # pre-norm before Q/K/V projection
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)  # one projection, split into Q,K,V

        if gating:
            self.to_gates = nn.Linear(dim, heads)  # per-head sigmoid gate (Section 3.1.1, "following Band Split RoFormer")
        else:
            self.to_gates = None

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)  # merge heads back to `dim`
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads  # split one projection into Q,K,V per head
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)  # inject relative position info into Q
            k = self.rotary_embed.rotate_queries_or_keys(k)  # ...and K (V is left untouched)

        out = self.attend(q, k, v)

        if exists(self.to_gates):
            gates = self.to_gates(x)  # per-head gate value, computed from the pre-norm input
            out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()  # scale each head's output by its own sigmoid gate

        out = rearrange(out, "b h n d -> b n (h d)")  # concat heads back together
        return self.to_out(out)


# Roformer


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=32,
        heads=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        gating=True,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):  # one (Attention, FeedForward) pair per layer, `depth` layers total
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            gating=gating,
                        ),
                        ff,
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()  # optional final norm after the last layer

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # residual: attention output added back to input
            x = ff(x) + x    # residual: feedforward output added back to input
        x = self.norm(x)
        return x
