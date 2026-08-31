"""
T -> N progressive downsampling (dp-short-1.pdf Section 2, eq. 1-3).

Without an FPN, takes the single (B,T,d_model) output of the encoder's last
layer (BeatThisEncoder) and progressively shrinks its length via
patch-merging into (B,N,d_model). N is not a fixed hyperparameter but a value
derived from N_min and the reduction ratio r (eq.1).
"""
import math

import torch
from torch import nn


def compute_schedule(T: int, N_min: int, r: float = 0.5):
    """eq.(1): T_0:=T, T_s:=ceil(r*T_{s-1}), S:=max{s : T_s >= N_min}, N:=T_S.

    Returns the list [T_1, T_2, ..., T_S] (T_0=T itself is not included,
    since that's just the input length).

    - Section 2 "Architecture" -

    compute_schedule implements the paper's eq.(1). eq.(1) is the formula
    that precomputes "how many steps (S) it takes to shrink T down to N, and
    what the final length N ends up being." At each step the length is
    roughly halved (r=1/2), but it keeps repeating only as long as it can
    stay at or above the floor N_min; the length at each step is returned as
    a list.

    In other words, if you keep halving T, there's no guarantee the result
    ever lands exactly on some arbitrarily chosen N. Instead, only N_min (the
    minimum number of candidates that must exist -- BPM_max x D_min = 100) is
    fixed, and the strategy is: "halve as many times as possible while
    staying at or above N_min, then take whatever length remains as N."

    The reason this shrinks over several steps instead of all at once: if
    1500 were compressed straight down to 188 in one step, each step would
    have to summarize an average of 8 (1500/188) positions at once, causing
    heavy information loss. Halving exactly in half each time keeps the
    amount of information any single step has to handle much smaller, at the
    cost of needing several steps to progressively summarize.
    """
    schedule = []
    T_s = T
    while True:
        next_T = math.ceil(r * T_s) # candidate length for this step
        if next_T < N_min: # if it would drop below N_min
            break # don't include this step, stop here
        T_s = next_T # confirm the candidate length
        schedule.append(T_s) # record it
    return schedule 


class FixedPoolStep(nn.Module):
    """Parameter-free downsample step: merge each adjacent pair by mean or max.

    Section 3 permits "any fixed downsampling operator ... for instance a strided
    convolution or a strided average-pooling layer", so a parameter-free operator is
    fully faithful -- and it removes 2.36M of the head's 3.29M parameters, which is 72%
    of a head whose actual decision layers are only 1,028 parameters (dense's are 1,026).

    mean vs max is not a neutral choice for this task. Beats are SPARSE salient events:
    averaging 8 frames dilutes a sharp onset by ~8x, while max preserves it. Against
    that, channel-wise max takes each of the 512 dimensions' argmax independently, so
    the pooled vector need not correspond to any single time position. Both are worth
    measuring rather than assuming.

    (Note this is NOT the same operation as Beat This's max-pooling, which is stride-1
    and length-preserving -- used for shift tolerance in the loss and for peak-picking
    at decode, not for reducing sequence length.)
    """

    def __init__(self, mode: str):
        super().__init__()
        assert mode in ("mean", "max")
        self.mode = mode

    def forward(self, x):
        B, T_prev, D = x.shape
        n = T_prev // 2
        pair = x[:, : 2 * n, :].reshape(B, n, 2, D)
        merged = pair.mean(2) if self.mode == "mean" else pair.max(2).values
        if T_prev % 2 == 1:                      # lone tail element passes through
            merged = torch.cat([merged, x[:, -1:, :]], dim=1)
        return merged


class DownsampleStep(nn.Module):
    """eq.(3): merges two adjacent positions into one via concat + linear
    projection. If the input length is odd, the last leftover position is
    handled on its own by a separate learnable linear projection (W'_s).

    Adjacent pairs within the step's input sequence (1&2, 3&4, ...) are
    concatenated, then a learnable linear projection W_s merges each pair
    into one. E.g. for a 750-length input: (g_1, g_2) -> g_1', (g_3, g_4) ->
    g_2', repeated 375 times to shrink to 375. Each step (1500 -> 750,
    750 -> 375, 375 -> 188) has its own independent W_s, b_s -- i.e. the
    three steps are separate DownsampleStep instances with separate
    parameters.

    Odd-length handling only triggers when the step's input length (T_{s-1})
    is odd. E.g. in the 375 -> 188 step, since 375 is odd, 187 pairs (374
    elements) go through the normal concat+W_s path, and the single leftover
    375th element goes through its own W'_s. The result is 187 + 1 = 188
    (even).

    This handling makes the resulting length exactly match eq.(1)'s schedule.
    """

    def __init__(self, d_model: int, init_as_pooling: bool = False):
        super().__init__()
        self.merge = nn.Linear(2 * d_model, d_model)  # W_s, b_s: learnable params that merge a pair into one
        self.odd_proj = nn.Linear(d_model, d_model)   # W'_s, b'_s (only used when odd): separate params for the odd leftover
        if init_as_pooling:
            # Start this step as strided average pooling: merge([g_a ; g_b]) = (g_a+g_b)/2
            # and odd_proj(g) = g. Section 3 names "a strided average-pooling layer" as an
            # acceptable Downsample, so this initialises the learned operator AT one of the
            # operators the document sanctions, and lets training depart from it.
            #
            # Why it may matter: this module is 2.36M of the head's 3.29M parameters and
            # sits on the ONLY gradient path from the loss to the encoder. Left at default
            # init it is three stacked random projections, so at step 0 the classifier reads
            # a scrambled view of the encoder and back-propagates scrambled gradient into it.
            # Measured context: the subset arm delivers 5.4x MORE per-parameter gradient to
            # the encoder than the dense arm and yet produces a WORSE encoder (end-to-end
            # 0.842 downbeat vs 0.895 with the encoder frozen) -- a large, mis-directed
            # signal, which is the shape a random bottleneck would produce.
            with torch.no_grad():
                eye = torch.eye(d_model)
                self.merge.weight.copy_(torch.cat([eye, eye], dim=1) * 0.5)
                self.merge.bias.zero_()
                self.odd_proj.weight.copy_(eye)
                self.odd_proj.bias.zero_()

    def forward(self, x):
        # x: (B, T_prev, d_model)
        B, T_prev, D = x.shape
        n_pairs = T_prev // 2

        paired = x[:, : 2 * n_pairs, :].reshape(B, n_pairs, 2 * D)  # [g_2i-1 ; g_2i] concat
        merged = self.merge(paired)  # (B, n_pairs, D)

        if T_prev % 2 == 1: # if odd, the last element has no partner
            last = self.odd_proj(x[:, -1:, :])  # (B, 1, D)
            merged = torch.cat([merged, last], dim=1)  # (B, n_pairs+1, D), appended at the end

        return merged  # (B, ceil(T_prev/2), D)


class ProgressiveDownsample(nn.Module):
    """encoder output (B,T,d_model) -> candidate feature (B,N,d_model).

    N is precomputed and fixed at __init__ time via compute_schedule(T, N_min, r),
    and one DownsampleStep with its own parameters is created per step
    T_1..T_S (eq.3: "W_s ... learned per step" -- independent weights per step).

    ProgressiveDownsample first precomputes how many steps are needed via
    compute_schedule (eq.1), builds that many independent DownsampleStep
    instances (eq.3), and runs them in order in forward -- the full pipeline
    that shrinks T down to N.
    """

    def __init__(self, d_model: int, T: int, N_min: int, r: float = 0.5,
                 init_as_pooling: bool = False, pool_mode: str = ""):
        super().__init__()
        self.schedule = compute_schedule(T, N_min, r) # precompute the schedule via eq.(1) -> [750, 375, 188]
        if not self.schedule:
            # N_min >= ceil(T/2): eq.(1)'s schedule is empty, so there is NO downsampling
            # step and z = h by identity, N = T. This is a legitimate configuration of the
            # formulation, not a degenerate one -- nothing in the correspondence problem,
            # the DP, or the loss requires N < T. It is also the only configuration in
            # which section 3's orthogonality claim holds vacuously: with no operator to
            # choose, no choice can matter. It isolates the formulation's actual thesis --
            # DP matching to exactly M events, with eq.(1) continuous times -- against
            # per-frame BCE at identical architecture and identical resolution.
            self.N = T
        else:
            # the last value of the schedule is the final N
            self.N = self.schedule[-1]
        # create one independent DownsampleStep per schedule entry (S=3)
        if pool_mode:      # parameter-free: frozen mean/max pooling, 0 params
            self.steps = nn.ModuleList([FixedPoolStep(pool_mode) for _ in self.schedule])
        else:
            self.steps = nn.ModuleList([DownsampleStep(d_model, init_as_pooling)
                                        for _ in self.schedule])

    def forward(self, x):
        for step in self.steps:
            x = step(x)
        # runs through the DownsampleSteps in order: 1500 -> 750 -> 375 -> 188
        return x  # (B, self.N, d_model)
